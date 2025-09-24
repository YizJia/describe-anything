import os
import json
import cv2
import numpy as np
from pathlib import Path
import pycocotools.mask as maskUtils
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2
from typing import List, Dict, Any
import argparse
import shutil
import tempfile
from PIL import Image

VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']

def extract_frames_from_video(video_path):
    """Extract frames from a video file and save them to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # 获取基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)                    # 帧率 (每秒帧数)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))        # 视频宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))      # 视频高度
    # duration = frame_count / fps if fps > 0 else 0        # 视频时长(秒)
    
    # 其他属性
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))            # 编码格式
    # pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)            # 当前帧的时间戳(毫秒)
    # pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))    # 当前帧位置
    # pos_avi_ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)  # 当前帧位置(相对于总帧数的比例)

    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        raise ValueError("No frames were extracted from the video.")
    
    return frame_paths, temp_dir, fps

def convert_sam2_mask_to_rle(mask):
    """
    将 SAM2 产生的二值 mask 转换为 RLE 格式
    
    Args:
        mask: numpy array, 二值掩码 (0 或 1)，shape 为 (H, W)
    
    Returns:
        dict: RLE 格式的掩码
    """
    # 确保 mask 是二值的 (0 或 1)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 转换为 RLE 格式
    rle = maskUtils.encode(np.asfortranarray(binary_mask))
    
    # 将 bytes 转换为字符串以便 JSON 序列化
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle

def convert_masks_to_rle_batch(masks):
    """
    批量转换多个 mask 为 RLE 格式

    Args:
        masks: list of numpy arrays 或 numpy array with shape (N, H, W)

    Returns:
        list: RLE 格式的掩码列表
    """
    rle_masks = []

    if isinstance(masks, np.ndarray) and len(masks.shape) == 3:
        # 如果是 (N, H, W) 格式
        for i in range(masks.shape[0]):
            rle = convert_sam2_mask_to_rle(masks[i])
            rle_masks.append(rle)
    else:
        # 如果是 mask 列表
        for mask in masks:
            rle = convert_sam2_mask_to_rle(mask)
            rle_masks.append(rle)

    return rle_masks

def apply_sam2(image_files, points=None, box=None, normalized_coords=False):
    """Apply SAM2 to video frames using points or box on first frame"""

    # If coordinates are normalized, convert them to absolute coordinates
    if normalized_coords:
        # Read first frame to get dimensions
        first_frame = cv2.imread(image_files[0])
        height, width = first_frame.shape[:2]
        
        if points is not None:
            points = np.array(points, dtype=np.float32)
            points[:, 0] *= width
            points[:, 1] *= height
        elif box is not None:
            box = np.array(box, dtype=np.float32)
            box[0] *= width  # x1
            box[1] *= height # y1
            box[2] *= width  # x2
            box[3] *= height # y2

    # Initialize inference state
    inference_state = predictor.init_state(video_path=image_files)
    predictor.reset_state(inference_state)

    # Add points or box on first frame
    ann_frame_idx = 0
    ann_obj_id = 1

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if points is not None:
            # Convert points to numpy array and add labels (all positive)
            points = np.array(points, dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels
            )
        elif box is not None:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box
            )

        # Propagate through video and collect masks
        masks = []
        rle_masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            masks.append(mask)

            # convert mask to RLE
            rle = convert_masks_to_rle_batch(mask)
            rle_masks.append(rle)

    return masks, rle_masks

def load_annotations(json_file: str) -> List[Dict[str, Any]]:
    """Load video annotations from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_frames_by_event(video_data: Dict[str, Any], base_path: str, output_dir: str):
    """Extract and organize frames by event"""
    video_root = video_data["image_root"]

    for ext in VIDEO_EXTENSIONS:
        video_name = f"{video_root}{ext}"
        video_path = os.path.join(base_path, video_name)
        if Path(video_path).exists():
            # Create output directory for this video
            video_output_dir = os.path.join(output_dir, video_root)
            os.makedirs(video_output_dir, exist_ok=True)
            break
    
    event_frames = []

    # First. extract all frames from the video
    all_frame_paths, temp_frame_dir, fps = extract_frames_from_video(video_path)
    print(f"Extracted {len(all_frame_paths)} frames from {video_path} at {fps} FPS.")

    sample_interval = 10  # You can adjust this value to sample frames at different intervals
    seg_frames = {}
    # Second. process frames according to each event
    for annotation, time_stamp in zip(video_data["annotations"], video_data["timestamps"]):
        event_id = annotation["event_id"]
        start_time, end_time = time_stamp
        
        # Calculate frame indices for this event
        start_frame_idx = int(start_time * fps)
        end_frame_idx = int(end_time * fps)
        
        # Create event directory
        event_dir = os.path.join(video_output_dir, f"event_{event_id}")
        os.makedirs(event_dir, exist_ok=True)
        
        # Copy relevant frames to event directory
        event_frame_paths = []
        for frame_idx in range(start_frame_idx, min(end_frame_idx + 1, len(all_frame_paths))):
            if (frame_idx - start_frame_idx) % sample_interval == 0 or frame_idx+1 == min(end_frame_idx + 1, len(all_frame_paths)):
                src_frame = all_frame_paths[frame_idx]
                dst_frame = os.path.join(event_dir, f"frame_{frame_idx:05d}.jpg")
                
                # Copy frame to event directory
                shutil.copy2(src_frame, dst_frame)
                event_frame_paths.append(dst_frame)

                seg_frames[frame_idx] = src_frame
        
        event_frames.append(
            {
                "event_id": event_id,
                "frames": event_frame_paths,
                "annotation": annotation,
                "event_dir": event_dir
            }
        )
    
    # Third. segment each frames according to the first bbox in annotation
    event_segmentations = []

    bbox = video_data["annotations"][0].get("box", None)
    assert bbox is not None, "No bbox found in the first annotation"
    # Remove duplicates and sort the seg_frames list
    seg_input_frames = list(seg_frames.values())
    seg_input_frames = sorted(list(set(seg_input_frames)))
    mask_files, rle_masks = process_video_segmentation(seg_input_frames, bbox[0], temp_frame_dir)
    
    # Fourth. assign masks to each event
    for annotation, time_stamp in zip(video_data["annotations"], video_data["timestamps"]):
        event_id = annotation["event_id"]
        start_time, end_time = time_stamp
        
        # Calculate frame indices for this event
        start_frame_idx = int(start_time * fps)
        end_frame_idx = int(end_time * fps)
        
        # Create event mask directory
        event_mask_dir = os.path.join(video_output_dir, f"event_{event_id}_masks")
        os.makedirs(event_mask_dir, exist_ok=True)
        
        # Copy relevant frames to event directory
        event_mask_paths = []
        for frame_idx in range(start_frame_idx, min(end_frame_idx + 1, len(all_frame_paths))):
            if (frame_idx - start_frame_idx) % sample_interval == 0 or frame_idx+1 == min(end_frame_idx + 1, len(all_frame_paths)):
                ind_frame = seg_frames.get(frame_idx, None)
                assert ind_frame is not None, f"No source frame found for index {frame_idx}"

                src_frame = mask_files[os.path.basename(ind_frame)]
                dst_frame = os.path.join(event_mask_dir, f"frame_{frame_idx:05d}_mask.png")
                
                # Copy frame to event directory
                shutil.copy2(src_frame, dst_frame)
                event_mask_paths.append(dst_frame)

        event_segmentations.append(
            {
                "event_id": event_id,
                "rle_masks": rle_masks,
                "mask_files": event_mask_paths,
            }
        )

    # Clean up temporary frames
    shutil.rmtree(temp_frame_dir)
    
    return event_frames, event_segmentations

def process_video_segmentation(frames: List[str], bbox: List[float], frame_root: str):
    """Process video segmentation for each frames"""
    results = []
    
    bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    # Apply SAM2 segmentation
    masks, rle_masks = apply_sam2(frames, box=bbox_xyxy)
    assert len(masks) == len(frames), "Number of masks should match number of frames"

    mask_dir = os.path.join(frame_root, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    mask_files = {}

    for (mask, frame) in zip(masks, frames):
        # mask_file = os.path.join(mask_dir, f"mask_{i:05d}.png")
        mask_file = os.path.join(mask_dir, os.path.basename(frame))
        # Convert boolean mask to uint8
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_file)
        mask_files[os.path.basename(frame)] = mask_file

    return mask_files, rle_masks

def save_rel_format(video_root: str, video_data: Dict[str, Any], event_frames: List[Dict], event_segmentations: List[Dict]):
    """Save results in REL format"""
    rel_data = {
        "type": video_data["type"],
        "image_root": video_data["image_root"],
        "annotations": []
    }
    
    ori_annotations = video_data["annotations"]
    for ori_anno, event_frame, event_seg in zip(ori_annotations, event_frames, event_segmentations):
        assert ori_anno["event_id"] == event_frame["event_id"], "Event ID mismatch"
        assert ori_anno["event_id"] == event_seg["event_id"], "Event ID mismatch"

        new_annotation = ori_anno.copy()
        new_annotation["frames"] = [str(Path(frame).relative_to(Path(video_root))) for frame in event_frame["frames"]]
        new_annotation["segmentations"] = event_seg["rle_masks"]
        
        rel_data["annotations"].append(new_annotation)

    return rel_data

def main():
    parser = argparse.ArgumentParser(description="Process video data with segmentation")
    parser.add_argument("--json_file", help="Path to JSON annotation file", default="data/merged_output.json")
    parser.add_argument("--base_path", help="Base path for video frames", default='data/PAM_ActivityNetCap_Subset/videos')
    parser.add_argument("--output_dir", help="Output directory for processed data", default='data/processed')
    parser.add_argument("--normalized_coords", action="store_true", help="Whether coordinates are normalized")
    parser.add_argument("--device", default="cuda", help="Device for SAM2")
    
    args = parser.parse_args()
    
    # Initialize SAM2
    global predictor, device
    device = args.device
    
    sam2_env_path = os.path.dirname(os.path.dirname(sam2.__file__))
    sam2_checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    sam2_checkpoint = os.path.join(sam2_env_path, sam2_checkpoint_path)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # Load annotations
    annotations = load_annotations(args.json_file)

    new_annotations = []
    
    # Process each video
    for video_data in annotations:
    # for i in range(1):
        # video_data = annotations[i]
        video_root = video_data["image_root"]
        print(f"Processing video: {video_root}")
        
        
        # Extract frames and segmentations by event
        event_frames, event_segmentations = extract_frames_by_event(video_data, args.base_path, args.output_dir)
        
        # Save results in REL format
        new_entry = save_rel_format(args.output_dir, video_data, event_frames, event_segmentations)
        new_annotations.append(new_entry)

        print(f"Completed processing: {video_root}")

    # Save all new annotations
    with open(os.path.join(args.output_dir, "new_annotations_masks.json"), 'w', encoding='utf-8') as f:
        json.dump(new_annotations, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()