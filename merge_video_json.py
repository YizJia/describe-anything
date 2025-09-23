#!/usr/bin/env python3
"""
视频标注信息合并脚本

从两个JSON文件中读取视频标注信息，将file2中的timestamps信息合并到file1的标注中。
file1包含详细的帧信息和边界框信息，file2包含时间戳信息。
"""

import json
import sys
import argparse
from typing import Dict, List, Any
from pathlib import Path
import shutil


def load_json_file1(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON文件 {file_path} 失败: {e}")
        sys.exit(1)

def load_json_file2(file_path: str) -> Dict[str, Any]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON文件 {file_path} 失败: {e}")
        sys.exit(1)


def transfer_video_name(image_root: str) -> str:
    """
    从image_root字段提取视频名称
    例如: "V_djpr7UMlnSw" -> "djpr7UMlnSw"
    """
    if image_root.startswith("V_"):
        return "v_" + image_root[2:]  # "V_" 前缀修改为 "v_"
    return image_root


def find_matching_video_in_file2(video_name: str, file2_data: Dict[str, Any], video_entry: Dict[str, Any]) -> List[str]:
    """
    在file2中查找匹配的视频
    除了名称匹配，还需要比较事件描述来确保是同一个条目
    返回匹配的key列表
    """
    matches = []
    
    # 从file1中提取所有事件的gt描述
    file1_gts = set()
    if "annotations" in video_entry:
        for annotation in video_entry["annotations"]:
            if "gt" in annotation:
                file1_gts.add(annotation["gt"].strip().lower())
    
    # file2_entry = file2_data.get(video_name, {})
    for key in file2_data.keys():
        # 首先检查名称匹配
        name_match = False
        if key == video_name:
            name_match = True
        elif key.startswith("v_") and key[2:] == video_name:
            name_match = True
        elif video_name.startswith("v_") and video_name[2:] == key:
            name_match = True
        elif video_name in key or key in video_name:
            name_match = True
        
        if not name_match:
            continue
        
        # 名称匹配后，检查描述文本匹配
        file2_entry = file2_data[key]
        if "sentences" not in file2_entry:
            continue
        
        # 提取file2中的所有句子描述
        file2_sentences = set()
        for sentence in file2_entry["sentences"]:
            file2_sentences.add(sentence.strip().lower())
        
        # 计算文本匹配度
        if file1_gts and file2_sentences:
            # 检查是否有相同的描述
            common_descriptions = file1_gts.intersection(file2_sentences)
            if common_descriptions:
                matches.append(key)
                print(f"    文本匹配成功，共同描述数: {len(common_descriptions)}")
            else:
                print(f"    名称匹配但文本不匹配: {key}")
        elif not file1_gts and not file2_sentences:
            # 如果两边都没有描述，仅依靠名称匹配
            matches.append(key)
        elif not file1_gts:
            print(f"    file1无gt描述，仅依靠名称匹配: {key}")
            matches.append(key)
        else:
            print(f"    file2无句子描述，跳过: {key}")
    
    return matches


def merge_annotations(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    合并两个标注文件
    
    Args:
        file1_path: 包含详细帧信息的文件路径
        file2_path: 包含时间戳信息的文件路径
        output_path: 输出文件路径
    """
    print(f"正在加载文件: {file1_path}")
    file1_data = load_json_file1(file1_path)
    
    print(f"正在加载文件: {file2_path}")
    file2_data = load_json_file2(file2_path)
    
    print(f"file1中包含 {len(file1_data)} 个视频条目")
    print(f"file2中包含 {len(file2_data)} 个视频条目")
    
    merged_data = []
    matched_count = 0
    unmatched_videos = []
    
    # 如果file1_data是单个视频对象，转换为列表
    if isinstance(file1_data, dict) and "type" in file1_data:
        file1_data = [file1_data]
    
    for video_entry in file1_data:
        if not isinstance(video_entry, dict) or "image_root" not in video_entry:
            print(f"警告: 跳过无效的视频条目: {video_entry}")
            continue
            
        video_root = video_entry["image_root"]
        video_name = transfer_video_name(video_root)

        print(f"\n处理视频: {video_name}")

        # 在file2中查找匹配的视频
        matches = find_matching_video_in_file2(video_name, file2_data, video_entry)
        
        if len(matches) == 0:
            print(f"  警告: 在file2中未找到匹配的视频")
            unmatched_videos.append(video_name)
            # 仍然保留原始数据，但不添加timestamps
            merged_data.append(video_entry.copy())
            continue
        elif len(matches) > 1:
            print(f"  错误: 找到多个匹配项: {matches}")
            print(f"  对应的file2数据:")
            for match in matches:
                print(f"    {match}: {file2_data[match]}")
            sys.exit(1)
        
        # 找到唯一匹配项
        match_key = matches[0]
        file2_entry = file2_data[match_key]
        
        print(f"  成功匹配: {match_key}")
        print(f"  file2条目包含 {len(file2_entry.get('sentences', []))} 个句子")
        print(f"  file1条目包含 {len(video_entry.get('annotations', []))} 个事件")
        
        # 创建合并后的条目
        merged_entry = video_entry.copy()
        merged_entry['image_root'] = video_name  # 确保image_root格式一致
        
        # 添加timestamps信息
        if "timestamps" in file2_entry:
            merged_entry["timestamps"] = file2_entry["timestamps"]
            print(f"  添加了 {len(file2_entry['timestamps'])} 个时间戳")
        
        # 添加duration信息（如果存在）
        if "duration" in file2_entry:
            merged_entry["duration"] = file2_entry["duration"]
            print(f"  添加了duration: {file2_entry['duration']}")
        
        # 可选：添加file2中的sentences信息作为参考
        # if "sentences" in file2_entry:
        #     merged_entry["file2_sentences"] = file2_entry["sentences"]
        #     print(f"  添加了file2的句子作为参考")
        
        merged_data.append(merged_entry)
        matched_count += 1

        # 如果成功匹配并添加了timestamps，复制视频文件
        video_dest_dir = Path("data/videos")
        video_dest_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持多种视频格式
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
        video_copied = False
        
        for ext in video_extensions:
            video_source_path = f"data/ActivityNet_Captions/Activity_Videos/{match_key}{ext}"
            if Path(video_source_path).exists():
                video_dest_path = video_dest_dir / f"{match_key}{ext}"
            try:
                shutil.copy2(video_source_path, video_dest_path)
                print(f"  复制视频文件: {video_source_path} -> {video_dest_path}")
                video_copied = True
                break
            except Exception as e:
                print(f"  错误: 复制视频文件失败: {e}")
        
        if not video_copied:
            print(f"  警告: 未找到视频文件 {match_key} (支持格式: {', '.join(video_extensions)})")
    
    # 保存合并后的数据
    print(f"\n保存合并结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"\n合并完成!")
    print(f"总处理视频数: {len(file1_data)}")
    print(f"成功匹配数: {matched_count}")
    print(f"未匹配数: {len(unmatched_videos)}")
    
    if unmatched_videos:
        print(f"\n未匹配的视频:")
        for video in unmatched_videos:
            print(f"  - {video}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='合并视频标注信息')
    parser.add_argument('--file1', help='包含详细帧信息的JSON文件路径', default='data/PAM_ActivityNetCap_Subset/input_full_md.jsonl')
    parser.add_argument('--file2', help='包含时间戳信息的JSON文件路径', default='data/ActivityNet_Captions/raw_data/val_1.json')
    parser.add_argument('--output', help='输出文件路径', default='data/merged_output.json')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.file1).exists():
        print(f"错误: 文件 {args.file1} 不存在")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"错误: 文件 {args.file2} 不存在")
        sys.exit(1)
    
    # 执行合并
    try:
        merge_annotations(args.file1, args.file2, args.output)
    except Exception as e:
        print(f"错误: 合并过程中发生异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
