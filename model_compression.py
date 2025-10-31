#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL模型压缩脚本

该脚本用于对Qwen2.5-VL系列模型进行压缩，支持多种压缩方法，并提供压缩前后的对比分析。
支持的模型：qwen2.5-vl-3b, qwen2.5-vl-7b等
支持的压缩方法：量化、剪枝等
"""

import os
import sys
import time
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import numpy as np
import json


def get_model_size(model_path):
    """
    获取模型的大小（以MB或GB为单位）
    """
    total_size = 0
    if os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    elif os.path.isfile(model_path):
        total_size = os.path.getsize(model_path)
    else:
        return "未知大小"
    
    # 转换为合适的单位
    if total_size < 1024 * 1024:
        return f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 * 1024 * 1024:
        return f"{total_size / (1024 * 1024):.2f} MB"
    else:
        return f"{total_size / (1024 * 1024 * 1024):.2f} GB"


def measure_inference_time(model, inputs, num_runs=2):
    """
    测量模型推理时间
    """
    print("  正在进行推理速度测试...", end="", flush=True)
    # 预热运行（减少token数量）
    model.generate(**inputs, max_new_tokens=5, do_sample=False)
    
    # 测量多次运行的平均时间（减少运行次数和token数量）
    total_time = 0
    for i in range(num_runs):
        start_time = time.time()
        # 减少生成的token数量以加快测试
        model.generate(**inputs, max_new_tokens=20, do_sample=False)
        end_time = time.time()
        total_time += (end_time - start_time)
        print(f" {i+1}/{num_runs}", end="" if i < num_runs-1 else "\n", flush=True)
    
    avg_time = total_time / num_runs
    print(f"  平均推理时间: {avg_time:.4f}秒")
    return avg_time


def calculate_perplexity(model, inputs):
    """
    计算模型的困惑度（作为简单的性能评估指标）
    """
    try:
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()
    except Exception as e:
        print(f"计算困惑度时出错: {e}")
        return None


def test_model_functionality(model, processor, test_image="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"):
    """
    测试模型功能是否正常工作
    """
    try:
        print("  正在测试模型功能...", end="", flush=True)
        
        # 创建简化的测试消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": "这是什么？"},
                ],
            }
        ]
        
        # 准备推理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 添加超时机制，使用较短的max_new_tokens
        start_time = time.time()
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=20,  # 减少生成的token数量
            do_sample=False,    # 禁用采样以加快速度
            temperature=0.0     # 使用确定性输出
        )
        generation_time = time.time() - start_time
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(" 完成")
        return {
            "success": True,
            "generation_time": generation_time,
            "output_text": output_text[0],
            "output_length": len(output_text[0])
        }
    except Exception as e:
        print(f" 失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def quantize_model(model, quantization_type="int8", model_name=None):
    """
    对模型进行量化
    
    Args:
        model: 要量化的模型
        quantization_type: 量化类型 ("int8", "int4", "nf4", "fp8")
        model_name: 模型名称（用于重新加载）
    
    Returns:
        量化后的模型
    """
    print(f"开始对模型进行{quantization_type}量化...")
    
    if quantization_type == "int8":
        # 对于大型模型，int8量化可能非常耗时，提供进度反馈
        print("  注意：对大型模型的INT8量化可能需要一些时间，请耐心等待...")
        # 使用PyTorch的int8量化（更高效的方式）
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    elif quantization_type in ["int4", "nf4"]:
        # 使用bitsandbytes进行4位量化
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4" if quantization_type == "nf4" else "fp4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            # 对于4位量化，我们需要重新加载模型
            if model_name is None:
                model_name = model.config._name_or_path
            
            print("  使用bitsandbytes加载量化模型...")
            quantized_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True  # 优化内存使用
            )
        except ImportError:
            print("错误：需要安装bitsandbytes库以使用4位量化。")
            print("请运行: pip install bitsandbytes")
            sys.exit(1)
    elif quantization_type == "fp8":
        # FP8量化（如果支持）
        if hasattr(model, 'to_fp8'):
            quantized_model = model.to_fp8()
        else:
            print("警告：当前模型不支持FP8量化，返回原始模型。")
            quantized_model = model
    else:
        print(f"警告：未知的量化类型 '{quantization_type}'，返回原始模型。")
        quantized_model = model
    
    print(f"模型{quantization_type}量化完成。")
    return quantized_model


def prune_model(model, pruning_ratio=0.1):
    """
    对模型进行剪枝（基于权重绝对值）
    
    Args:
        model: 要剪枝的模型
        pruning_ratio: 剪枝比例（要移除的权重比例）
    
    Returns:
        剪枝后的模型
    """
    print(f"开始对模型进行剪枝，剪枝比例: {pruning_ratio:.1%}")
    
    # 对每个线性层进行剪枝
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 获取权重绝对值
            weights_abs = torch.abs(module.weight.data)
            # 计算阈值
            threshold = torch.quantile(weights_abs, pruning_ratio)
            # 将低于阈值的权重置零
            mask = weights_abs > threshold
            module.weight.data *= mask
    
    print("模型剪枝完成。")
    return model


def save_compressed_model(model, processor, save_dir, compression_type):
    """
    保存压缩后的模型
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(save_dir)
    # 保存处理器
    processor.save_pretrained(save_dir)
    
    print(f"压缩后的模型已保存到: {save_dir}")
    return save_dir


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Qwen2.5-VL模型压缩脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                      help="要压缩的模型名称或路径")
    parser.add_argument("--compression_type", type=str, default="int8", 
                      choices=["int8", "int4", "nf4", "fp8", "prune"],
                      help="压缩类型")
    parser.add_argument("--pruning_ratio", type=float, default=0.1, 
                      help="剪枝比例（仅在compression_type=prune时有效）")
    parser.add_argument("--output_dir", type=str, default="./compressed_model", 
                      help="压缩后模型的保存目录")
    parser.add_argument("--device", type=str, default="cpu", 
                      help="使用的设备（cpu或cuda）")
    parser.add_argument("--test_image", type=str, default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                      help="用于测试模型功能的图像URL")
    
    args = parser.parse_args()
    
    print(f"开始处理模型: {args.model_name}")
    print(f"压缩类型: {args.compression_type}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载原始模型和处理器
    print("正在加载原始模型和处理器...")
    original_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # 准备测试输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.test_image},
                {"type": "text", "text": "描述这张图片。"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    test_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    test_inputs = test_inputs.to(args.device)
    
    # 评估原始模型
    print("\n评估原始模型性能...")
    print("注意：为了加速测试，推理和功能测试都已优化，使用较少的token数量。")
    original_inference_time = measure_inference_time(original_model, test_inputs)
    original_functionality = test_model_functionality(original_model, processor, args.test_image)
    
    # 释放内存
    if hasattr(original_model, 'to'):
        original_model = original_model.to('cpu')
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 给用户提供进度反馈
    print("\n准备进行模型压缩...")
    
    # 确定原始模型大小（估计值，基于模型类型）
    if "3B" in args.model_name:
        original_model_size = "约12GB"
    elif "7B" in args.model_name:
        original_model_size = "约28GB"
    elif "14B" in args.model_name:
        original_model_size = "约56GB"
    else:
        original_model_size = "未知"
    
    # 压缩模型
    print("\n开始压缩模型...")
    if args.compression_type == "prune":
        compressed_model = prune_model(original_model, args.pruning_ratio)
    else:
        # 传递模型名称以优化量化过程
        compressed_model = quantize_model(original_model, args.compression_type, args.model_name)
    
    # 评估压缩后模型
    print("\n评估压缩后模型性能...")
    compressed_inference_time = measure_inference_time(compressed_model, test_inputs)
    compressed_functionality = test_model_functionality(compressed_model, processor, args.test_image)
    
    # 保存压缩后的模型
    model_save_dir = os.path.join(args.output_dir, f"{os.path.basename(args.model_name)}-{args.compression_type}")
    save_compressed_model(compressed_model, processor, model_save_dir, args.compression_type)
    
    # 获取压缩后模型的实际大小
    compressed_model_size = get_model_size(model_save_dir)
    
    # 计算压缩比
    # 注意：这里使用的是估计值，实际压缩比可能有所不同
    if original_model_size != "未知" and "GB" in original_model_size:
        original_size_gb = float(original_model_size.split()[0])
        if "GB" in compressed_model_size:
            compressed_size_gb = float(compressed_model_size.split()[0])
        elif "MB" in compressed_model_size:
            compressed_size_gb = float(compressed_model_size.split()[0]) / 1024
        else:
            compressed_size_gb = float(compressed_model_size.split()[0]) / (1024 * 1024)
        
        compression_ratio = original_size_gb / compressed_size_gb
        size_reduction = (1 - 1/compression_ratio) * 100
    else:
        compression_ratio = "N/A"
        size_reduction = "N/A"
    
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time
    
    # 生成压缩报告
    report = {
        "model_name": args.model_name,
        "compression_type": args.compression_type,
        "pruning_ratio": args.pruning_ratio if args.compression_type == "prune" else "N/A",
        "total_processing_time": f"{total_time:.2f}秒",
        "original_model": {
            "estimated_size": original_model_size,
            "inference_time": f"{original_inference_time:.4f}秒",
            "functionality_test": original_functionality
        },
        "compressed_model": {
            "actual_size": compressed_model_size,
            "inference_time": f"{compressed_inference_time:.4f}秒",
            "functionality_test": compressed_functionality,
            "save_path": model_save_dir
        },
        "compression_metrics": {
            "compression_ratio": compression_ratio if compression_ratio == "N/A" else f"{compression_ratio:.2f}x",
            "size_reduction": size_reduction if size_reduction == "N/A" else f"{size_reduction:.1f}%",
            "speed_change": f"{((compressed_inference_time - original_inference_time) / original_inference_time) * 100:.1f}%"
        }
    }
    
    # 打印报告
    print("\n===== 模型压缩报告 =====")
    print(f"模型名称: {report['model_name']}")
    print(f"压缩类型: {report['compression_type']}")
    if report['pruning_ratio'] != "N/A":
        print(f"剪枝比例: {report['pruning_ratio']:.1%}")
    print(f"总处理时间: {report['total_processing_time']}")
    
    print("\n原始模型:")
    print(f"  估计大小: {report['original_model']['estimated_size']}")
    print(f"  推理时间: {report['original_model']['inference_time']}")
    print(f"  功能测试: {'成功' if report['original_model']['functionality_test']['success'] else '失败'}")
    if report['original_model']['functionality_test']['success']:
        print(f"  输出示例: {report['original_model']['functionality_test']['output_text'][:100]}...")
    
    print("\n压缩后模型:")
    print(f"  实际大小: {report['compressed_model']['actual_size']}")
    print(f"  推理时间: {report['compressed_model']['inference_time']}")
    print(f"  功能测试: {'成功' if report['compressed_model']['functionality_test']['success'] else '失败'}")
    if report['compressed_model']['functionality_test']['success']:
        print(f"  输出示例: {report['compressed_model']['functionality_test']['output_text'][:100]}...")
    print(f"  保存路径: {report['compressed_model']['save_path']}")
    
    print("\n压缩指标:")
    print(f"  压缩比: {report['compression_metrics']['compression_ratio']}")
    print(f"  大小减少: {report['compression_metrics']['size_reduction']}")
    print(f"  速度变化: {report['compression_metrics']['speed_change']}")
    
    # 保存报告到JSON文件
    report_file = os.path.join(args.output_dir, "compression_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存到: {report_file}")
    print("压缩任务完成！")


if __name__ == "__main__":
    main()