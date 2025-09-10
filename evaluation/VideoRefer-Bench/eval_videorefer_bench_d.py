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
# from sam2.build_sam import build_sam2_video_predictor
# from mm_utils import load_video

def extract_frames_from_video(video_path):
    """Extract frames from a video file and save them to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_path)
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
    
    return frame_paths, temp_dir

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

class VideoRefer_Bench_D(Dataset):
    def __init__(self, video_folder, data_list, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        # self.processor = processor
        self.mode = mode
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, line['video'])
        
        question = "Please describe in detail the marked object <region> in the video.\n<video>"
        
        video_name = line['video']
        annotations = line['annotation']
        
        image_files = []
        masks = []
        
        raw_frame_files, temp_dir = extract_frames_from_video(video_path)
      
        for ann in annotations:
            for key, value in ann.items():
                assert key.isdigit(), f"Key {key} in annotations is not a digit."
                image_files.append(raw_frame_files[int(key)-1])
                mask = annToMask(value['segmentation'])
                
                # 验证 mask 是否有效
                if mask.sum() == 0:
                    print(f"Warning: Empty mask found for key {key} in video {video_name}")
                    # 创建一个最小的有效 mask
                    m = np.zeros_like(mask)
                    m[m.shape[0]//2, m.shape[1]//2] = 1  # 在中心放置一个像素
                    masks.append(m)
                    continue

                masks.append(mask)

        indices = np.linspace(0, len(image_files)-1, 8, dtype=int)
        selected_images = [image_files[i] for i in indices]
        selected_masks = [masks[i] for i in indices]

        # Convert frames to PIL images
        processed_images = [Image.open(f).convert('RGB') for f in selected_images]
        processed_masks = [Image.fromarray((m.squeeze() * 255).astype(np.uint8)) for m in selected_masks]
        # processed_masks = []
        # for m in selected_masks:
        #     # 再次验证处理后的 mask
        #     if m.sum() == 0:
        #         # 创建一个最小的有效 mask
        #         m = np.zeros_like(m)
        #         m[m.shape[0]//2, m.shape[1]//2] = 1  # 在中心放置一个像素
        #     processed_masks.append(Image.fromarray((m.squeeze() * 255).astype(np.uint8)))

        shutil.rmtree(temp_dir)

        return {
            'video_name': video_name,
            'video': processed_images,
            'masks': processed_masks,
            'question': question,
            # 'mask_ids': mask_ids,
            'answer': line['caption'],
        }

def collate_fn(batch):
    vin = [x['video_name'] for x in batch]
    vid = [x['video'] for x in batch]
    msk = [x['masks'] for x in batch]
    qs = [x['question'] for x in batch]
    # mid = [x['mask_ids'] for x in batch]
    ans = [x['answer'] for x in batch]
    return vin, vid, msk, qs, ans

def build_VideoRefer_Bench_D_eval(args):
    # convert parquet to json
    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    dataset = VideoRefer_Bench_D(args.video_folder, questions, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --points '[[1824, 397]]' --output_image_dir videos/1_visualization
    # Example: python examples/dam_video_with_sam2.py --video_file videos/1.mp4 --points '[[1824, 397]]' --output_image_dir videos/1_visualization

    # Example: python examples/dam_video_with_sam2.py --video_dir videos/1 --box '[1612, 364, 1920, 430]' --output_image_dir videos/1_visualization
    
    parser = argparse.ArgumentParser(description="Describe Anything script")

    parser.add_argument('--model_path', type=str, default='nvidia/DAM-3B-Video', help='Path to the model checkpoint')
    parser.add_argument('--video_folder', type=str, default='data/VideoRefer-Bench/Panda-70M-part', help='Directory containing video files.')
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', default='data/VideoRefer-Bench/VideoRefer-Bench-D.json')
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', default='evaluation/model_outputs_cache/DAM-3B-VideoRefer-Bench-D.json')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    parser.add_argument('--query', type=str, default='Video: <image><image><image><image><image><image><image><image>\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.', help='Prompt for the model')
    
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

    answer_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    
    val_loader = build_VideoRefer_Bench_D_eval(args)
 
    for i, (video_names, video, masks_, questions, answers) in enumerate(tqdm(val_loader)):
        # 检查是否有有效数据
        if video is None or masks_ is None:
            continue
        video_name = video_names[0]
        video_tensor = video[0]
        masks = masks_[0]
        question = questions[0]
        # mask_ids = mask_ids[0]
        answer = answers[0]

        # 验证 masks 是否有效
        valid_masks = []
        for mask_pil in masks:
            mask_array = np.array(mask_pil)
            if mask_array.sum() == 0:
                # 创建最小有效 mask
                mask_array = np.zeros_like(mask_array)
                mask_array[mask_array.shape[0]//2, mask_array.shape[1]//2] = 255
                mask_pil = Image.fromarray(mask_array)
            valid_masks.append(mask_pil)

        try:
            outputs = dam.get_description(video_tensor, valid_masks, args.query, 
                                        temperature=args.temperature, top_p=args.top_p, 
                                        num_beams=1, max_new_tokens=512)
            print(f"video_index: {i}\n")
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            outputs = "Error in processing"
        
        # print("Description:")
        # outputs = dam.get_description(video_tensor, masks, args.query, temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=512)
        # print(f"Description:\n{outputs}")
        # if not args.no_stream:
        #     for token in dam.get_description(video_tensor, masks, args.query, streaming=True, temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=512):
        #         print_streaming(token)
        #     print()
        # else:
        #     outputs = dam.get_description(video_tensor, masks, args.query, temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=512)
        #     print(f"Description:\n{outputs}")

        record = {
            'video': video_name,
            'Answer': answer,
            'pred': outputs,
        }
        ans_file.write(json.dumps(record) + "\n")

    ans_file.close()