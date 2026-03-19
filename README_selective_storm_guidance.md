# Selective Storm Guidance - 开发简报

## 项目信息
- **开发日期**: 2024-12-30
- **功能名称**: Selective Storm Guidance（选择性风暴引导）
- **新主脚本**: `10_12-30_gencast_readout_selective_storm_guidance.py`
- **基于脚本**: `10_12-15_gencast_readout_general_guidance_woRollOut.py`

## 问题背景

原脚本当目标日期有多个风暴时，guidance 只指定一个目标位置，其他风暴可能被"消除"（因为 loss 在整个空间计算）。本功能实现选择性引导：只对指定风暴计算 guidance loss，其他风暴保持自然发展。

## 文件改动清单

### 新增文件

1. **`10_12-30_gencast_readout_selective_storm_guidance.py`**
   - 新主脚本
   - 核心函数：
     - `extract_storms_from_end_date()`: 从 end_date 提取所有风暴信息
     - `get_storm_id_for_date_pair()`: Storm ID 选择（配置优先 → 缓存 → IO输入）
     - `compute_bounding_box_mask()`: 计算包含选定原始风暴和目标风暴的矩形 mask
   - Storm ID 缓存机制：每个 `(start_date, end_date)` 组合只询问一次，超参数 loop 自动复用

2. **`README_selective_storm_guidance.md`** (本文件)

### 修改文件

1. **`graphcast/dpm_solver_plus_plus_2s.py`**
   - 新增函数：`two_channel_crossentropy_optax_with_guidance_mask()`
     - 支持 `guidance_mask` 参数，只在 mask 区域计算 loss
   - 修改方法：`guided()` 添加可选 `guidance_mask` 参数
   - 原函数保持不变（向后兼容）

2. **`graphcast/gencast.py`**
   - 新增配置类：`ReadOutGuidanceConfigWithMask`
     - 继承自 `ReadOutGuidanceConfig`，添加 `guidance_mask` 字段
   - 修改方法：`readout_guided_inference_vis()` 支持新配置类
   - 原配置类保持不变（向后兼容）

## 核心功能

1. **风暴信息提取**: 从 `end_date` 自动提取所有风暴（storm_id, lat, lon, rsize）
2. **Storm ID 选择**: 配置优先 → 缓存检查 → IO输入（每个日期组合只输入一次）
3. **Bounding Box Mask**: 计算包含选定原始风暴和目标风暴的矩形区域
4. **选择性 Loss**: 只在 mask 区域内计算 guidance loss

## 使用方式

### 配置示例

```python
SWEEP_WANT_TIMES = ["2017-09-07 00:00:00"]  # start_date
END_DATES = ["2017-09-08 00:00:00"]         # end_date

# 情况A：已指定 storm_id（不需要 IO）
SELECTED_STORM_IDS = [0]

# 情况B：未指定（需要 IO，但只输入一次）
SELECTED_STORM_IDS = [None]

MANUAL_TARGETS = [{"lat": 22.0, "lon": 292.0, "radius": 4}]
GUIDANCE_MASK_PADDING = 5
```

### 执行流程

```
外层循环：(start_date, end_date)
  → 提取风暴信息
  → 获取 storm_id（配置/缓存/IO）
  → 计算 bounding box mask
内层循环：超参数 sweep
  → 复用已选择的 storm_id
  → 只在 mask 区域计算 loss
```

## 技术细节

- 使用 `scipy.ndimage.label` 提取连通区域识别风暴
- Bounding box 计算：包含选定原始风暴 + 目标风暴 + padding
- Loss 归一化：按 mask 区域大小归一化，而非整个空间
- 向后兼容：原函数/类保持不变

## 测试建议

- 单风暴/多风暴场景
- 配置指定 vs IO 输入
- 验证缓存机制和超参数复用
- 验证其他风暴是否保持自然发展

