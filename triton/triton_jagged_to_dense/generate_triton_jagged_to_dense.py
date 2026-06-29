#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate file for triton_jagged_to_dense ATK test cases.
Ensures shape consistency between jagged_values, jagged_offsets, and max_seq_len.
"""

from atk.generator import GENERATOR_REGISTRY, CaseGenerator
from atk.generator.config import CaseConfig


def is_power_of_2(n):
    """判断 n 是否是 2 的幂次方"""
    return n > 0 and (n & (n - 1)) == 0


def get_next_power_of_2(n):
    """获取大于等于 n 的最小 2 的幂次方"""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@GENERATOR_REGISTRY.register("generate_triton_jagged_to_dense")
class TritonJaggedToDenseGenerator(CaseGenerator):
    """
    生成 triton_jagged_to_dense 算子的测试用例约束

    约束规则：
    1. jagged_values.shape = (total_elements, embedding_dim)
    2. jagged_offsets.shape = (batch_size + 1,)
    3. jagged_offsets[-1] 应该等于 total_elements（在API中确定性生成）
    4. max_seq_len >= max(offsets[i+1] - offsets[i])，确保不会截断
    5. embedding_dim 建议是 2 的幂次方（triton性能优化）
    """

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """
        在用例配置生成后，调整参数约束以满足算子要求

        参数顺序：
        0: jagged_values
        1: jagged_offsets
        2: max_seq_len
        3: padding_value
        """
        # 获取输入参数
        jagged_values = case_config.inputs[0]
        jagged_offsets = case_config.inputs[1]
        max_seq_len = case_config.inputs[2]

        # 获取 jagged_values 的 shape
        jagged_values_shape = list(jagged_values.shape) if hasattr(jagged_values, 'shape') else [128, 64]
        total_elements = jagged_values_shape[0]
        embedding_dim = jagged_values_shape[1]

        # ========== 约束1: embedding_dim 调整为 2 的幂次方 ==========
        # Triton 在处理内存访问时，2的幂次方维度性能更好
        if not is_power_of_2(embedding_dim):
            embedding_dim = get_next_power_of_2(embedding_dim)
            # 确保不会太大
            if embedding_dim > 1024:
                embedding_dim = 512
            elif embedding_dim < 16:
                embedding_dim = 16

        # ========== 约束2: 确保 batch_size 合理 ==========
        # jagged_offsets 的长度是 batch_size + 1
        jagged_offsets_shape = list(jagged_offsets.shape) if hasattr(jagged_offsets, 'shape') else [9]
        batch_size = jagged_offsets_shape[0] - 1

        # 确保 batch_size 至少为 1
        if batch_size < 1:
            batch_size = 8
            jagged_offsets_shape[0] = batch_size + 1

        # ========== 约束3: 调整 total_elements 以匹配 batch_size ==========
        # 确保平均每个 batch 有合理数量的元素
        avg_elements_per_batch = total_elements // batch_size
        if avg_elements_per_batch < 4:
            avg_elements_per_batch = 16

        # 重新计算 total_elements，确保能被 batch_size 合理分配
        total_elements = batch_size * avg_elements_per_batch

        # ========== 约束4: max_seq_len 必须 >= avg_elements_per_batch ==========
        # 这样可以避免序列被截断（实际的 offsets 会在 API 中确定性生成）
        # 保守估计，让 max_seq_len 稍大一些
        min_max_seq_len = avg_elements_per_batch + (avg_elements_per_batch // 4)

        # 调整 max_seq_len 的范围
        if hasattr(max_seq_len, 'range_values'):
            current_min = max_seq_len.range_values[0]
            current_max = max_seq_len.range_values[1]

            # 确保 max_seq_len 至少是 min_max_seq_len
            new_min = max(current_min, min_max_seq_len)
            new_max = max(current_max, min_max_seq_len + 64)

            max_seq_len.range_values = [new_min, new_max]

        # ========== 应用修改后的 shape ==========
        jagged_values.shape = [total_elements, embedding_dim]
        jagged_offsets.shape = [batch_size + 1]

        return case_config
