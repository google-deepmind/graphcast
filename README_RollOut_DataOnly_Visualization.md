# Rollout 数据生成与可视化分离方案

## 概述

为了加速 rollout 实验，将原始脚本分离为两个独立脚本：
1. **数据生成脚本**：执行推理并保存数据
2. **可视化脚本**：读取保存的数据并生成图片/视频

## 文件说明

### 1. `10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly.py`
- **功能**：执行 rollout 推理（baseline 和 guided），保存数据为 NetCDF 格式
- **输出**：在每个输出目录下创建 `rollout_data/` 子目录，包含：
  - `predictions_step*.nc`: 每一步的预测结果
  - `readouts_step*_denoising*.nc`: 每一步的 readout 结果
  - `gt_step*.nc`: Ground truth 数据
  - `one_hot.npy`: One-hot mask
  - `metadata.json`: 元数据（epoch, visual_vars, num_steps 等）
- **优势**：快速完成推理，无需等待画图

### 2. `10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py`
- **功能**：扫描目录，找到所有 `rollout_data/`，加载数据并生成可视化
- **输出**：在原输出目录下创建 `visualization/` 子目录，包含：
  - PNG 图片（按变量和步骤组织）
  - GIF 动画
  - 对比视频（baseline vs guided）
- **优势**：
  - 可以多次运行，调整可视化参数
  - 支持批量处理多个实验结果
  - 可以在不同机器上运行（只要有数据文件）

## 使用流程

### Step 1: 生成数据（推理）

```bash
# 运行 rollout 推理，保存数据
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly.py
```

配置文件中的参数（与原脚本相同）：
- `SWEEP_WANT_TIMES`: 输入时间点
- `END_DATES`: 目标时间点
- `SELECTED_STORM_IDS`: 要引导的风暴 ID
- `MANUAL_TARGETS`: 目标位置和半径
- `NUM_ROLLOUT_STEPS`: rollout 步数

输出示例：
```
output_dir/
  by_dates/
    20221015_00/
      steer_one_storm3/
        step_vis10_15_inner0-18_str0.7_seed42/
          A_no_guidance/
            rollout_data/           # <- 数据目录
              predictions_step00.nc
              predictions_step01.nc
              ...
              gt_step00.nc
              one_hot.npy
              metadata.json
          B_steer_one_storm3/
            rollout_data/
              (同上)
```

### Step 2: 生成可视化

```bash
# 扫描并可视化所有保存的数据
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /path/to/output_dir \
    --vis_n_procs 4 \
    --vis_dpi 150
```

常用参数：
- `--root_dir`: 输出根目录（会递归搜索所有 `rollout_data/`）
- `--skip_existing`: 跳过已存在的可视化
- `--only_video`: 只生成视频，跳过 PNG 生成
- `--vis_n_procs`: 并行进程数（推荐 4-8）
- `--vis_dpi`: PNG 分辨率（默认 150，原脚本用 240）
- `--vis_coastlines_resolution`: 海岸线分辨率（110m/50m/10m，默认 110m 更快）

输出示例：
```
output_dir/
  by_dates/
    20221015_00/
      steer_one_storm3/
        step_vis10_15_inner0-18_str0.7_seed42/
          A_no_guidance/
            rollout_data/           # <- 数据目录（不变）
            visualization/          # <- 可视化目录（新增）
              2m_temperature/
                step00_2m_temperature_frame0_epoch0.png
                step01_2m_temperature_frame1_epoch0.png
                ...
              geopotential/
                ...
          B_steer_one_storm3/
            rollout_data/
            visualization/
            comparison_videos/      # <- 对比视频（新增）
              2m_temperature_comparison.mp4
              geopotential_comparison.mp4
              ...
```

## 数据格式说明

### NetCDF 格式 (.nc)
- **优势**：
  - 保留完整的 xarray 元数据（坐标、维度、属性）
  - 压缩效率高（通常比 pickle 小 3-5 倍）
  - 跨语言兼容（Python, R, MATLAB, NCL 等）
  - 支持分步读取，内存友好
- **文件大小**：
  - 12 步 × 10 变量 × 181×360 × 4 bytes ≈ 30-50MB（未压缩）
  - 压缩后通常 10-20MB（使用 zlib level 5）

### 元数据 (metadata.json)
```json
{
  "epoch": 0,
  "visual_vars": ["2m_temperature", "geopotential", ...],
  "num_steps": 12,
  "type": "baseline" 或 "guided",
  "want_time": "2022-10-15T00:00:00",
  "end_date": "2022-10-17T00:00:00",
  "guidance_mode": "steer_one",
  "guidance_strength": 0.7,
  "loss_history": {...}  # 仅 guided
}
```

## 高级用法

### 批量处理多个实验
```bash
# 一次性可视化所有实验
python visualization.py --root_dir /path/to/all/experiments --vis_n_procs 8

# 只生成缺失的可视化
python visualization.py --root_dir /path/to/all/experiments --skip_existing

# 只生成视频（PNG 已存在）
python visualization.py --root_dir /path/to/all/experiments --only_video
```

### 调整可视化参数
```bash
# 高分辨率可视化
python visualization.py --root_dir /path/to/output \
    --vis_dpi 300 \
    --vis_coastlines_resolution 50m \
    --vis_draw_gridlabels \
    --vis_add_borders

# 快速预览（低分辨率）
python visualization.py --root_dir /path/to/output \
    --vis_dpi 100 \
    --vis_coastlines_resolution 110m \
    --vis_contourf_levels 12
```

### 在不同机器上运行
```bash
# 机器 A：运行推理（GPU）
python DataOnly.py

# 将 output_dir/ 复制到机器 B

# 机器 B：生成可视化（CPU，多核）
python visualization.py --root_dir /path/to/copied/output --vis_n_procs 16
```

## 性能对比

### 原始脚本（推理 + 可视化）
- 12 步 rollout：~10 分钟（推理） + ~30 分钟（可视化） = **40 分钟**
- 必须顺序执行
- 可视化参数调整需要重新运行整个流程

### 分离方案
- **推理**：~10 分钟
- **可视化**：~30 分钟（独立运行）
- 总时间：10 分钟（如果只需要推理结果）
- 可以多次调整可视化参数，无需重新推理

## 注意事项

1. **数据文件大小**：
   - 每个 rollout（12 步）约 10-20MB（压缩后）
   - 多个实验会累积，注意磁盘空间

2. **兼容性**：
   - 可视化脚本需要 `graphcast.vis` 模块
   - 确保 PYTHONPATH 包含 graphcast 路径

3. **元数据完整性**：
   - metadata.json 包含所有可视化所需的参数
   - 如果手动修改数据，记得更新 metadata.json

4. **并行可视化**：
   - `--vis_n_procs` 设置为 CPU 核心数的 50-75%
   - 太高可能导致内存不足

## 故障排除

### 问题：可视化脚本找不到数据
```bash
# 检查数据目录结构
find /path/to/output -name "rollout_data" -type d

# 检查 metadata.json 是否存在
find /path/to/output -name "metadata.json"
```

### 问题：内存不足
```bash
# 减少并行进程数
python visualization.py --root_dir /path/to/output --vis_n_procs 2

# 或者逐个目录处理
for dir in /path/to/output/by_dates/*/; do
    python visualization.py --root_dir "$dir" --vis_n_procs 4
done
```

### 问题：可视化函数导入失败
```bash
# 确保 graphcast 在 PYTHONPATH 中
export PYTHONPATH=/path/to/WeatherControl:$PYTHONPATH
python visualization.py --root_dir /path/to/output
```

## 未来改进

- [ ] 支持更多数据格式（Zarr, HDF5）
- [ ] 增量可视化（只处理新数据）
- [ ] Web 界面查看结果
- [ ] 自动检测损坏的数据文件
- [ ] 支持远程数据加载（S3, GCS）
