import os
import json
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import sam2
from typing import List, Dict, Any
import argparse

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
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            masks.append(mask)

    return masks

def load_annotations(json_file: str) -> List[Dict[str, Any]]:
    """Load video annotations from JSON file"""
    annotations = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations

def extract_frames_by_event(video_data: Dict[str, Any], base_path: str, output_dir: str):
    """Extract and organize frames by event"""
    image_root = video_data["image_root"]
    video_path = os.path.join(base_path, image_root)
    
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, image_root)
    os.makedirs(video_output_dir, exist_ok=True)
    
    event_frames = {}
    
    for annotation in video_data["annotations"]:
        event_id = annotation["event_id"]
        frames = annotation["frames"]
        
        # Create event directory
        event_dir = os.path.join(video_output_dir, f"event_{event_id}")
        os.makedirs(event_dir, exist_ok=True)
        
        # Copy frames to event directory
        frame_paths = []
        for frame in frames:
            src_path = os.path.join(video_path, frame)
            dst_path = os.path.join(event_dir, frame)
            
            if os.path.exists(src_path):
                # Copy frame
                img = cv2.imread(src_path)
                cv2.imwrite(dst_path, img)
                frame_paths.append(dst_path)
        
        event_frames[event_id] = {
            "frames": frame_paths,
            "annotation": annotation,
            "event_dir": event_dir
        }
    
    return event_frames

def process_video_segmentation(event_frames: Dict[int, Dict], args):
    """Process video segmentation for each event"""
    results = {}
    
    for event_id, event_data in event_frames.items():
        frames = event_data["frames"]
        annotation = event_data["annotation"]
        event_dir = event_data["event_dir"]
        
        if "box" in annotation and annotation["box"]:
            bbox = annotation["box"][0]  # Take first box
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            
            # Apply SAM2 segmentation
            masks = apply_sam2(frames, box=bbox_xyxy, normalized_coords=args.normalized_coords)
            
            # Save masks
            mask_dir = os.path.join(event_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            
            mask_files = []
            for i, mask in enumerate(masks):
                mask_file = os.path.join(mask_dir, f"mask_{i:05d}.png")
                # Convert boolean mask to uint8
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(mask_file, mask_img)
                mask_files.append(mask_file)
            
            results[event_id] = {
                "masks": masks,
                "mask_files": mask_files,
                "annotation": annotation
            }
    
    return results

def save_rel_format(video_data: Dict[str, Any], segmentation_results: Dict, output_file: str):
    """Save results in REL format"""
    rel_data = {
        "type": video_data["type"],
        "image_root": video_data["image_root"],
        "annotations": []
    }
    
    for annotation in video_data["annotations"]:
        event_id = annotation["event_id"]
        
        new_annotation = annotation.copy()
        
        if event_id in segmentation_results:
            # Add mask information
            new_annotation["mask_files"] = segmentation_results[event_id]["mask_files"]
            new_annotation["has_segmentation"] = True
        else:
            new_annotation["has_segmentation"] = False
        
        rel_data["annotations"].append(new_annotation)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rel_data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Process video data with segmentation")
    parser.add_argument("--json_file", required=True, help="Path to JSON annotation file")
    parser.add_argument("--base_path", required=True, help="Base path for video frames")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
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
    
    # Process each video
    for video_data in annotations:
        print(f"Processing video: {video_data['image_root']}")
        
        # Extract frames by event
        event_frames = extract_frames_by_event(video_data, args.base_path, args.output_dir)
        
        # Process segmentation
        segmentation_results = process_video_segmentation(event_frames, args)
        
        # Save results in REL format
        output_file = os.path.join(args.output_dir, f"{video_data['image_root']}_rel.json")
        save_rel_format(video_data, segmentation_results, output_file)
        
        print(f"Completed processing: {video_data['image_root']}")

if __name__ == "__main__":
    main()