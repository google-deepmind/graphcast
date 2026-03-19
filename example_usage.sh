#!/bin/bash
# 使用示例：分离的 rollout 推理和可视化流程

# ============================================================================
# 示例 1：基本用法
# ============================================================================

cd /fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/graphcast_Cleaned/
conda activate gencast

# Step 1: 运行 rollout 推理（生成数据）
echo "Step 1: Running rollout inference..."
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_DataOnly.py

# Step 2: 生成可视化
echo "Step 2: Generating visualizations..."
cd /fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/graphcast_Cleaned/
python 10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/NatureJournal_0226-01—LocalAffine_RollOut/seed789 \
    --vis_n_procs 4 \
    --parallel_folders 3 \
    --vis_dpi 150 \
    --skip_existing

# /fs/nexus-projects/DeepOptic_Ev/16_WeatherControl/NatureJournal_0124—WeakStrong_RollOut/General_Guidance_01

# ============================================================================
# 示例 2：只生成视频（PNG 已存在）
# ============================================================================

echo "Generating videos only..."
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /path/to/output \
    --only_video \
    --skip_existing

# ============================================================================
# 示例 3：高分辨率可视化
# ============================================================================

echo "High-resolution visualization..."
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /path/to/output \
    --vis_dpi 300 \
    --vis_coastlines_resolution 50m \
    --vis_draw_gridlabels \
    --vis_add_borders \
    --vis_n_procs 8

# ============================================================================
# 示例 4：快速预览（低分辨率）
# ============================================================================

echo "Quick preview..."
python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
    --root_dir /path/to/output \
    --vis_dpi 100 \
    --vis_coastlines_resolution 110m \
    --vis_contourf_levels 12 \
    --vis_n_procs 4

# ============================================================================
# 示例 5：批量处理多个实验
# ============================================================================

echo "Batch processing..."
for experiment in exp1 exp2 exp3; do
    python graphcast_Cleaned/10_12-30_gencast_readout_selective_storm_guidance_RollOut_visualization.py \
        --root_dir /path/to/experiments/$experiment \
        --skip_existing \
        --vis_n_procs 4
done

# ============================================================================
# 示例 6：跨机器运行（GPU 推理 + CPU 可视化）
# ============================================================================

# 在 GPU 机器上：
echo "On GPU machine: running inference..."
# python DataOnly.py  # 生成数据
# rsync -av output_dir/ remote_cpu_machine:/path/to/output_dir/

# 在 CPU 机器上（多核）：
echo "On CPU machine: generating visualizations..."
# python visualization.py --root_dir /path/to/output_dir --vis_n_procs 16

# ============================================================================
# 示例 7：检查目录结构
# ============================================================================

echo "Checking directory structure..."
find /path/to/output -name "rollout_data" -type d | head -5
find /path/to/output -name "metadata.json" | head -5

# ============================================================================
# 示例 8：查看数据文件大小
# ============================================================================

echo "Checking data sizes..."
du -sh /path/to/output/*/*/rollout_data 2>/dev/null | head -5

echo "Done!"
