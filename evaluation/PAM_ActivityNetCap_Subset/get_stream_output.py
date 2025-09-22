# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This script is used to segment objects in a video using SAM2 and then describe the segmented objects using DAM. 
# This script uses SAM (v2.1) and requires localization for the first frame.

import argparse
import ast
import torch
import numpy as np
import math
from PIL import Image
import pycocotools.mask as maskUtils
from dam import DescribeAnythingModel, disable_torch_init
import cv2
import glob
import os
import tempfile
import json
from tqdm import tqdm
import shutil
from torch.utils.data import Dataset, DataLoader
from sam2.build_sam import build_sam2_video_predictor
# from mm_utils import load_video

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
    # video_dir = os.path.dirname(image_files[0])
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

def print_streaming(text):
    """Helper function to print streaming text with flush"""
    print(text, end="", flush=True)

def annToMask(mask_ann, h=None, w=None):
    if isinstance(mask_ann, list):
        rles = maskUtils.frPyObjects(mask_ann, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class ActivityNet_Cap_Dataset(Dataset):
    def __init__(self, data_list, mode, video_folder):
        self.data_list = data_list
        self.mode = mode
        self.video_folder = video_folder
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_type = line.get("type", "unknown")
        assert video_type=="video", f"Only 'video' type is supported, but got {video_type}"
        video_path = os.path.join(self.video_folder, line['image_root'])
        assert os.path.exists(video_path), f"Video file {video_path} does not exist."

        annotations = line.get("annotations", [])
        assert len(annotations) > 0, f"No annotations found in line {idx}"

        return {
            'video_type': video_type,
            'video_path': video_path,
            'annotations': annotations,
        }

def collate_fn(batch):
    vty = [x['video_type'] for x in batch]
    vpt = [x['video_path'] for x in batch]
    annos = [x['annotations'] for x in batch]
    return vty, vpt, annos

def build_ActivityNet_Cap_eval(args):
    # convert parquet to json
    try:
        with open(args.question_file, "r") as f:
            questions = [json.loads(line) for line in f.readlines()]
    except Exception as e:
        print(f"Error reading {args.question_file}: {e}")
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = ActivityNet_Cap_Dataset(questions, args.mode, args.video_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --points '[[1824, 397]]' --output_image_dir videos/1_visualization
    # Example: python examples/dam_video_with_sam2.py --video_file videos/1.mp4 --points '[[1824, 397]]' --output_image_dir videos/1_visualization

    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --box '[1612, 364, 1920, 430]' --output_image_dir videos/1_visualization
    
    parser = argparse.ArgumentParser(description="Describe Anything script")

    parser.add_argument('--model_path', type=str, default='nvidia/DAM-3B-Video', help='Path to the model checkpoint')
    parser.add_argument('--video-folder', help='Path to the video file.', default='data/PAM_ActivityNetCap_Subset/frames')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='data/ActivityNet-Cap-Subset/input_full_md.jsonl')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='evaluation/model_outputs_cache/DAM-3B-ActivityNet-Caps_Subset.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    parser.add_argument('--query', type=str, default='\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.', help='Prompt for the model')
    
    parser.add_argument('--prompt_mode', type=str, default='focal_prompt', help='Prompt mode')
    parser.add_argument('--conv_mode', type=str, default='v1', help='Conversation mode')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.5, help='Top-p for sampling')
    parser.add_argument('--normalized_coords', action='store_true', 
                       help='Interpret coordinates as normalized (0-1) values')
    parser.add_argument('--no_stream', action='store_true', help='Disable streaming output')

    args = parser.parse_args()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DAM model and get description
    disable_torch_init()

    prompt_modes = {
        "focal_prompt": "full+focal_crop",
    }
    
    dam = DescribeAnythingModel(
        model_path=args.model_path,
        conv_mode=args.conv_mode,
        prompt_mode=prompt_modes.get(args.prompt_mode, args.prompt_mode),
    ).to(device)

    import sam2
    sam2_env_path = os.path.dirname(os.path.dirname(sam2.__file__))
    sam2_checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    # sam2_checkpoint = "/raid/jia_yizhen/code/sam2/checkpoints/sam2.1_hiera_large.pt"
    sam2_checkpoint = os.path.join(sam2_env_path, sam2_checkpoint_path)
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    
    val_loader = build_ActivityNet_Cap_eval(args)
 
    all_results = []
    for i, (video_type, video_path, annos) in enumerate(tqdm(val_loader)):
        video_name = video_type[0]
        video_path = video_path[0]
        annos = annos[0]

        print(f"\n=== Processing video {i + 1}: {video_name} ===")
        video_result = {
            "type": video_type,
            "image_root": video_path,
            "annotations": []
        }

        for anno_idx, anno in enumerate(annos):
            # get event details
            event_id = anno.get("event_id", "unknown_event")
            frames = anno.get("frames", [])
            assert len(frames) > 0, f"No frames found for event {event_id}"
            bbox_xywh = anno.get("box", None)  # [x, y, w, h]
            assert bbox_xywh is not None, f"No box found for event {event_id}"
            bbox_xywh = bbox_xywh[0]
            gt_caption = anno.get("gt", "")

            current_frames = []
            for frame in frames:
                frame_path = os.path.join(video_path, frame)
                current_frames.append(frame_path)
                assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."

            x1, y1 = int(bbox_xywh[0]), int(bbox_xywh[1])
            x2, y2 = int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3])
            bbox_xyxy = [x1, y1, x2, y2]

            # Process video with SAM2
            masks = apply_sam2(current_frames, box=bbox_xyxy, normalized_coords=args.normalized_coords)

            # Convert frames to PIL images
            processed_images = [Image.open(f).convert('RGB') for f in current_frames]
            processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in masks]

            question = 'Video: ' + '<image>'*len(current_frames) + args.query

            try:
                outputs = dam.get_description(processed_images, processed_masks, question, 
                                            temperature=args.temperature, top_p=args.top_p, 
                                            num_beams=1, max_new_tokens=512)
                print("Descriptions: ", outputs)
                history = outputs
            except Exception as e:
                print(f"Error processing video {video_name}: {e}")
                outputs = "Error in processing"

            event_result = {
                "event_id": event_id,
                "gt": gt_caption,
                "pred": outputs
            }
            video_result["annotations"].append(event_result)

        ans_file.write(json.dumps(video_result, ensure_ascii=False) + '\n')
        all_results.append(video_result)

    ans_file.close()