# Qwen2.5-VL 模型压缩工具

这是一个用于压缩 Qwen2.5-VL 系列模型的 Python 脚本，具有良好的扩展性，可以支持不同大小的模型（如 3B、7B 等）和多种压缩方法。

## 功能特性

- 支持多种压缩方法：
  - INT8 量化
  - INT4 量化
  - NF4 量化（4位NormalFloat）
  - FP8 量化
  - GPTQ 量化（需要预训练的GPTQ模型）
  - AWQ 量化（需要预训练的AWQ模型）
  - GGUF 量化（需要预训练的GGUF模型）
  - 模型剪枝（支持多种剪枝策略）
  - 知识蒸馏（Knowledge Distillation）
- 剪枝策略：
  - Magnitude-based（基于权重幅度）
  - Random（随机剪枝）
  - Structured（结构化剪枝）
  - Layer-adaptive（层自适应剪枝）
- 提供压缩前后的详细对比报告，包括：
  - 模型大小对比
  - 推理速度对比
  - 功能测试结果对比
  - 压缩比和大小减少百分比
- 自动保存压缩后的模型和报告文件
- 支持自定义测试图像
- 支持在 CPU 或 GPU 上运行

## 安装依赖

在使用本脚本之前，请确保安装以下依赖：

```bash
pip install torch transformers qwen-vl-utils

# 对于 4 位量化，还需要安装：
pip install bitsandbytes
```

## 使用方法

### 基本用法

```bash
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "int8" --output_dir "./compressed_model"
```

### 参数说明

- `--model_name`：要压缩的模型名称或本地路径（默认为 "Qwen/Qwen2.5-VL-3B-Instruct"）
- `--compression_type`：压缩类型，可选值："int8", "int4", "nf4", "fp8", "gptq", "awq", "gguf", "prune", "distill"（默认为 "int8"）
- `--pruning_ratio`：剪枝比例，仅在 `compression_type=prune` 时有效（默认为 0.1，即 10%）
- `--pruning_method`：剪枝方法，仅在 `compression_type=prune` 时有效，可选值："magnitude", "random", "structured", "layer_adaptive"（默认为 "magnitude"）
- `--student_model_name`：学生模型名称，仅在 `compression_type=distill` 时有效（默认为 None，表示使用与教师模型相同的模型）
- `--distill_epochs`：蒸馏训练轮数，仅在 `compression_type=distill` 时有效（默认为 3）
- `--output_dir`：压缩后模型的保存目录（默认为 "./compressed_model"）
- `--device`：使用的设备，可选值："cpu" 或 "cuda"（默认为 "cpu"）
- `--test_image`：用于测试模型功能的图像 URL（默认为示例图像）

### 示例命令

#### 快速测试命令

如果您只是想快速测试脚本是否正常工作，可以先运行一个简化版本（使用更小的测试图像和更少的评估）：

```bash
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "int8" --device "cpu"
```

#### 2. 使用 4 位量化（NF4）压缩 Qwen2.5-VL-7B 模型

```bash
python model_compression.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct" --compression_type "nf4" --device "cuda"
```

#### 3. 使用剪枝（20%）压缩模型

```bash
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "prune" --pruning_ratio 0.2
```

#### 4. 指定自定义输出目录和测试图像

```bash
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "int8" --output_dir "./my_compressed_models" --test_image "https://example.com/test.jpg"
```

### 快速测试命令

```bash
# INT8量化
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "int8"

# 4位NF4量化
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "nf4"

# 剪枝（10%剪枝率，magnitude方法）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "prune" --pruning_ratio 0.1
```

### 高级使用示例

```bash
# 使用GPTQ量化（需要预训练的GPTQ模型）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "gptq"

# 使用AWQ量化（需要预训练的AWQ模型）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "awq"

# 使用GGUF量化（需要预训练的GGUF模型）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "gguf"

# 使用不同的剪枝方法
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "prune" \
  --pruning_ratio 0.2 --pruning_method "random"

# 结构化剪枝
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "prune" \
  --pruning_ratio 0.15 --pruning_method "structured"

# 层自适应剪枝
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "prune" \
  --pruning_ratio 0.2 --pruning_method "layer_adaptive"

# 使用知识蒸馏（使用相同模型作为学生模型）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-3B-Instruct" --compression_type "distill" \
  --distill_epochs 5

# 使用知识蒸馏（指定不同的学生模型）
python model_compression.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct" --compression_type "distill" \
  --student_model_name "Qwen/Qwen2.5-VL-3B-Instruct" --distill_epochs 3
```

## 输出结果

脚本执行完成后，会生成以下输出：

1. **压缩后的模型文件**：保存在指定的输出目录中
2. **压缩报告**：
   - 控制台输出的摘要报告
   - JSON 格式的详细报告文件（`compression_report.json`）

## 报告内容说明

压缩报告包含以下主要信息：

- **模型基本信息**：模型名称、压缩类型、处理时间等
- **原始模型信息**：估计大小、推理时间、功能测试结果
- **压缩后模型信息**：实际大小、推理时间、功能测试结果、保存路径
- **压缩指标**：压缩比、大小减少百分比、速度变化百分比

## 性能优化说明

脚本已经过优化，可以在保持功能完整的同时提高运行速度：

1. **减少测试token数量**：评估过程中生成更少的token以加快测试速度
2. **减少重复运行次数**：推理时间测量使用更少的运行次数
3. **禁用采样**：使用确定性输出以加快生成速度
4. **内存管理**：在不同阶段释放不必要的内存
5. **进度反馈**：添加详细的进度指示，让用户了解当前处理阶段

## 注意事项

1. **内存要求**：压缩大模型时可能需要较大的内存，请确保系统有足够的可用内存
2. **4位量化**：使用 INT4 或 NF4 量化时，需要安装 bitsandbytes 库
3. **功能测试**：脚本会自动测试压缩前后模型的功能，确保压缩后模型仍能正常工作
4. **性能权衡**：压缩可以减小模型大小和提高推理速度，但可能会略微影响模型性能
5. **GPU使用**：对于较大的模型，建议使用 `--device "cuda"` 参数在 GPU 上运行以提高速度
6. **CPU处理**：在CPU上处理大型模型可能需要较长时间，请耐心等待
7. **进度反馈**：如果长时间没有进度更新，模型可能仍在处理中，请不要过早中断脚本

## 支持的模型

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- 其他 Qwen2.5-VL 系列模型

## 常见问题

### 1. 为什么 4 位量化失败？

请确保已安装 bitsandbytes 库：
```bash
pip install bitsandbytes
```

### 2. 如何选择合适的压缩方法？

- **INT8**：较好的平衡点，通常可以将模型大小减少约 75%，对性能影响较小
- **INT4/NF4**：更大的压缩比，但可能对性能有一定影响
- **剪枝**：可以在保持精度的同时减小模型大小，但需要谨慎调整剪枝比例

### 3. 压缩后的模型如何使用？

压缩后的模型可以通过标准的 Transformers 库加载和使用，与原始模型使用方式相同。