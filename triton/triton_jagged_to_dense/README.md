# triton_jagged_to_dense ATK 测试用例

## 算子功能

`triton_jagged_to_dense` 将 jagged tensor（不规则张量）转换为 dense tensor（稠密张量）。

**输入参数：**
- `jagged_values`: 2D tensor，shape 为 `(total_elements, embedding_dim)`，存储所有jagged元素
- `jagged_offsets`: 1D tensor，shape 为 `(batch_size + 1,)`，存储每个batch的起始偏移量
- `max_seq_len`: 标量，输出dense tensor的最大序列长度
- `padding_value`: 标量，用于填充的值（默认0.0）

**输出：**
- `output_dense`: 3D tensor，shape 为 `(batch_size, max_seq_len, embedding_dim)`

**示例：**
```python
# jagged_values: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]  # (5, 2)
# jagged_offsets: [0, 2, 5]  # batch_size = 2
# max_seq_len: 3
# 输出 dense:
# [[[1, 2], [3, 4], [0, 0]],   # batch 0: 2个元素，第3个padding
#  [[5, 6], [7, 8], [9, 10]]]  # batch 1: 3个元素
```

## 文件结构

```
triton_jagged_to_dense/
├── triton_jagged_to_dense.yaml              # YAML配置文件
├── generate_triton_jagged_to_dense.py       # Generate约束生成器
├── triton_jagged_to_dense_api.py            # API执行接口
└── README.md                                 # 本说明文档
```

## 关键设计说明

### 1. 确定性参数生成

**jagged_offsets 的生成策略：**
- ⚠️ **禁止使用随机数**：ATK需要在GPU和NPU上用相同输入对比精度
- 使用**确定性算法**：基于batch索引的模运算生成变化的segment长度
- 确保 `offsets[-1] == total_elements`
- 确保每个segment长度 `<= max_seq_len`

### 2. Shape约束

**Generate文件中的约束：**
- `embedding_dim` 调整为2的幂次方（Triton性能优化）
- `batch_size >= 1`
- `total_elements` 与 `batch_size` 匹配
- `max_seq_len >= avg_elements_per_batch`（避免截断）

**API文件中的约束：**
- 确定性生成 `jagged_offsets`，保证总和等于 `total_elements`
- 转换参数到正确设备（NPU/GPU）
- 确保tensor连续性

### 3. 数据类型支持

根据源代码分析：
- `jagged_values`: 支持 `fp32`, `fp16`, `bf16`
- `jagged_offsets`: 使用 `int64`
- `max_seq_len`: `int64`
- `padding_value`: `fp32`

## 使用方法

### 1. 生成测试用例

```bash
cd E:\ai_infra\triton\triton_jagged_to_dense

# 生成测试用例JSON
atk case -f triton_jagged_to_dense.yaml -p generate_triton_jagged_to_dense.py
```

生成结果：`all_triton_jagged_to_dense.json`

### 2. 创建node.yaml配置

创建 `node.yaml` 文件，配置测试节点：

```yaml
nodes:
    - backend: npu
      task: ['accuracy']
      devices: [0]
    - backend: gpu
      host: 90.90.80.24
      port: 12132
      devices: [0]
      task: ['accuracy']
```

### 3. 运行测试

```bash
# 精度测试
atk task -c all_triton_jagged_to_dense.json -n node.yaml -p triton_jagged_to_dense_api.py

# 性能测试
# 修改 node.yaml 中的 task 为 ['performance_device']
atk task -c all_triton_jagged_to_dense.json -n node.yaml -p triton_jagged_to_dense_api.py
```

## 约束说明

### Generate文件约束
- ✅ Shape调整：embedding_dim为2的幂次方
- ✅ Shape调整：batch_size >= 1
- ✅ Shape调整：total_elements与batch_size匹配
- ✅ 间接约束：max_seq_len范围调整

### API文件约束
- ✅ 确定性生成jagged_offsets
- ✅ 设备转换和同步
- ✅ Tensor连续性保证

## 常见问题

### Q1: 为什么要在API中重新生成jagged_offsets？
**A:** ATK工具生成的JSON用例只包含shape、dtype和ranges，不包含具体数值。而jagged_offsets有特殊约束：
- `offsets[-1]` 必须等于 `total_elements`
- 必须单调递增
- 使用确定性算法保证GPU/NPU输入一致

### Q2: 为什么embedding_dim要调整为2的幂次方？
**A:** Triton在处理2的幂次方维度时，内存访问模式更优化，性能更好。这不是硬性要求，但是最佳实践。

### Q3: 如何验证生成的测试用例？
**A:** 可以运行简单的单元测试：
```python
import torch
from triton_jagged_to_dense_api import TritonJaggedToDenseApi
from atk.api.dataset import InputDataset

# 创建测试数据
api = TritonJaggedToDenseApi()
input_data = InputDataset(kwargs={
    "jagged_values": torch.randn(100, 64),
    "jagged_offsets": torch.tensor([0, 10]),
    "max_seq_len": 64,
    "padding_value": 0.0
})

api.init_by_input_data(input_data)
output = api(input_data)
print(f"Output shape: {output.shape}")  # 应该是 (batch_size, max_seq_len, embedding_dim)
```

## 参考资料

- 源代码：`E:\ai_infra\FBGEMM-1.5.0\fbgemm_gpu\fbgemm_gpu\triton\jagged\triton_jagged_tensor_ops.py`
- ATK测试指南：`E:\ai_infra\triton\skill\ATKtest\skill.md`
- Triton文档：https://triton-lang.org/

## 更新日志

- 2026-06-29: 初始版本，支持2D jagged tensor转换
