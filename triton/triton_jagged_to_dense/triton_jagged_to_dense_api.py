#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API file for triton_jagged_to_dense ATK test cases.
Implements the algorithm execution interface with deterministic parameter generation.
"""

import torch
from atk.api import BaseApi, register
from atk.api.dataset import InputDataset


def get_device():
    """获取可用设备，优先使用NPU"""
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


@register("triton_jagged_to_dense")
class TritonJaggedToDenseApi(BaseApi):
    """
    triton_jagged_to_dense 算子的 ATK API 实现

    算子功能：将 jagged tensor 转换为 dense tensor
    - 输入：jagged_values, jagged_offsets, max_seq_len, padding_value
    - 输出：dense tensor with shape (batch_size, max_seq_len, embedding_dim)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_device()

    def _init_device(self):
        """初始化设备，优先使用 NPU"""
        if hasattr(torch, 'npu') and torch.npu.is_available():
            self._device = 'npu'
        elif torch.cuda.is_available():
            self._device = 'cuda'
        else:
            self._device = 'cpu'

    def _generate_jagged_offsets_deterministic(self, batch_size, total_elements, max_seq_len):
        """
        确定性生成 jagged_offsets

        ⚠️ 重要：不能使用随机数！必须保证GPU和NPU上的输入完全一致

        算法：
        1. 计算平均每个batch的元素数
        2. 使用确定性规则（如交替变化）生成每个segment的长度
        3. 确保总和等于 total_elements
        4. 确保每个segment长度不超过 max_seq_len
        """
        # 计算平均步长
        avg_step = total_elements // batch_size

        # 生成确定性的segment长度
        segment_lengths = []
        remaining = total_elements

        for i in range(batch_size):
            # 确定性规则：使用索引的模运算来产生变化
            # 这样可以产生不同长度的segment，但完全确定
            variation = (i % 3) - 1  # 产生 -1, 0, 1 的循环
            seg_len = avg_step + variation

            # 确保segment长度合理
            seg_len = max(1, seg_len)
            seg_len = min(seg_len, max_seq_len)
            seg_len = min(seg_len, remaining - (batch_size - i - 1))  # 确保剩余够分配

            segment_lengths.append(seg_len)
            remaining -= seg_len

        # 如果有剩余元素，分配给最后几个batch
        if remaining > 0:
            for i in range(batch_size - 1, -1, -1):
                if remaining == 0:
                    break
                add_amount = min(remaining, max_seq_len - segment_lengths[i])
                segment_lengths[i] += add_amount
                remaining -= add_amount

        # 生成累积偏移量
        cumulative_offsets = [0]
        for seg_len in segment_lengths:
            cumulative_offsets.append(cumulative_offsets[-1] + seg_len)

        return cumulative_offsets

    def init_by_input_data(self, input_data: InputDataset):
        """
        初始化操作，不计入性能统计
        ⚠️ 所有预处理都在这里完成，禁止使用随机数！

        参数获取：
        - jagged_values: tensor
        - jagged_offsets: tensor (需要重新生成以满足约束)
        - max_seq_len: scalar
        - padding_value: scalar
        """
        device = self._device

        # 同步操作
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ========== 获取输入参数 ==========
        jagged_values_input = input_data.kwargs.get("jagged_values", None)
        jagged_offsets_input = input_data.kwargs.get("jagged_offsets", None)
        max_seq_len = input_data.kwargs.get("max_seq_len", 128)
        padding_value = input_data.kwargs.get("padding_value", 0.0)

        # 参数校验
        if jagged_values_input is None:
            raise ValueError("jagged_values not found in input_data")
        if jagged_offsets_input is None:
            raise ValueError("jagged_offsets not found in input_data")

        # ========== 转换 jagged_values 到正确设备 ==========
        if not isinstance(jagged_values_input, torch.Tensor):
            jagged_values = torch.tensor(jagged_values_input, dtype=torch.float32, device=device)
        else:
            jagged_values = jagged_values_input.to(device)

        # 确保连续性
        if not jagged_values.is_contiguous():
            jagged_values = jagged_values.contiguous()

        # ========== 关键：使用确定性算法重新生成 jagged_offsets ==========
        # 从 jagged_offsets_input 获取 shape 信息
        if isinstance(jagged_offsets_input, torch.Tensor):
            batch_size = jagged_offsets_input.shape[0] - 1
        else:
            batch_size = len(jagged_offsets_input) - 1

        total_elements = jagged_values.shape[0]

        # 确定性生成 jagged_offsets
        offsets_list = self._generate_jagged_offsets_deterministic(
            batch_size, total_elements, max_seq_len
        )

        # 转换为 tensor
        jagged_offsets = torch.tensor(offsets_list, dtype=torch.int64, device=device)

        # ========== 转换标量参数 ==========
        if isinstance(max_seq_len, torch.Tensor):
            max_seq_len = int(max_seq_len.item())
        else:
            max_seq_len = int(max_seq_len)

        if isinstance(padding_value, torch.Tensor):
            padding_value = float(padding_value.item())
        else:
            padding_value = float(padding_value)

        # ========== 保存实例变量供 __call__ 使用 ==========
        self._jagged_values = jagged_values
        self._jagged_offsets = jagged_offsets
        self._max_seq_len = max_seq_len
        self._padding_value = padding_value
        self._batch_size = batch_size

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        执行算子，只负责调用 kernel，不做参数处理

        输出：dense tensor with shape (batch_size, max_seq_len, embedding_dim)
        """
        device = self._device

        # 同步
        if device == 'npu':
            torch.npu.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

        # ========== 导入算子函数 ==========
        # 需要从 FBGEMM 中导入 jagged_to_dense 函数
        try:
            import sys
            import os
            # 添加 FBGEMM 路径到 sys.path
            fbgemm_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'FBGEMM-1.5.0', 'fbgemm_gpu')
            if os.path.exists(fbgemm_path):
                sys.path.insert(0, fbgemm_path)

            from fbgemm_gpu.triton.jagged.triton_jagged_tensor_ops import jagged_to_dense
        except ImportError as e:
            raise ImportError(f"Failed to import jagged_to_dense: {e}")

        # ========== 准备参数 ==========
        # jagged_offsets 需要是 list[torch.Tensor] 格式
        # 对于 2D jagged tensor，只需要一个 offset tensor
        jagged_offsets_list = [self._jagged_offsets]

        # jagged_max_lengths 是输出 dense 的 shape 中间维度
        # 对于 2D jagged tensor，只需要一个 max_length
        jagged_max_lengths = [self._max_seq_len]

        # ========== 调用 kernel ==========
        output_dense = jagged_to_dense(
            jagged_values=self._jagged_values,
            jagged_offsets=jagged_offsets_list,
            jagged_max_lengths=jagged_max_lengths,
            padding_value=self._padding_value,
            operation_function=None,  # 不使用融合操作
            operation_dense=None
        )

        return output_dense
