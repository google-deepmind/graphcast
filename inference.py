import dataclasses
import datetime
import math
import argparse
from google.cloud import storage
from typing import Optional
import haiku as hk
from IPython.display import HTML
from IPython import display
import ipywidgets as widgets
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

from graphcast import (
    rollout, xarray_jax, normalization, checkpoint, 
    data_utils, xarray_tree, gencast, denoiser, nan_cleaning
)

# 데이터 처리 함수들
def select(
        data: xarray.Dataset, 
        variable: str, 
        level: Optional[int] = None, 
        max_steps: Optional[int] = None
        ) -> xarray.Dataset:
    """데이터셋에서 특정 변수 선택"""
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(
        data: xarray.Dataset, 
        center: Optional[float] = None,
        robust: bool = False
        ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    """데이터 스케일링"""
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax),
            ("RdBu_r" if center is not None else "viridis"))

# 모델 관련 함수들
def load_model(params_file: str, source: str = "Checkpoint") -> dict:
    """모델 파라미터 로드"""
    DEFAULT_CONFIG = {
        "latent_size": 512,
        "attention_type": "splash_mha",
        "mesh_size": 4,
        "num_heads": 4,
        "attention_k_hop": 16,
        "resolution": "1p0"
    }
    
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    
    if source == "Random":
        # Random 초기화 로직
        params = None
        state = {}
        task_config = gencast.TASK
        sampler_config = gencast.SamplerConfig()
        noise_config = gencast.NoiseConfig()
        noise_encoder_config = denoiser.NoiseEncoderConfig()
        denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
            sparse_transformer_config=denoiser.SparseTransformerConfig(**DEFAULT_CONFIG),
            mesh_size=DEFAULT_CONFIG["mesh_size"],
            latent_size=DEFAULT_CONFIG["latent_size"],
        )
    else:
        # Checkpoint 로드 로직
        with gcs_bucket.blob(f"params/{params_file}").open("rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)
            params = ckpt.params
            state = {}
            task_config = ckpt.task_config
            sampler_config = ckpt.sampler_config
            noise_config = ckpt.noise_config
            noise_encoder_config = ckpt.noise_encoder_config
            denoiser_architecture_config = ckpt.denoiser_architecture_config
    
    return {
        "params": params,
        "state": state,
        "task_config": task_config,
        "sampler_config": sampler_config,
        "noise_config": noise_config,
        "noise_encoder_config": noise_encoder_config,
        "denoiser_architecture_config": denoiser_architecture_config
    }

def load_example_data(dataset_file: str) -> xarray.Dataset:
    """예제 데이터 로드"""
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    
    with gcs_bucket.blob(f"dataset/{dataset_file}").open("rb") as f:
        return xarray.load_dataset(f).compute()

# 모델 추론 함수들
def construct_wrapped_gencast(model_dict):
    """GenCast Predictor 구성"""
    predictor = gencast.GenCast(**model_dict)
    predictor = normalization.InputsAndResiduals(predictor)
    predictor = nan_cleaning.NaNCleaner(predictor)
    return predictor

@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
    predictor = construct_wrapped_gencast()
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

# 추론 관련 함수들
def run_inference(model_dict: dict, example_data: xarray.Dataset, 
                 variable: str, num_ensemble_members: int = 8) -> xarray.Dataset:
    """모델 추론 실행"""
    # 데이터 추출
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_data, 
        target_lead_times=slice("12h", f"{(example_data.dims['time']-2)*12}h"),
        **dataclasses.asdict(model_dict["task_config"])
    )
    
    params = model_dict["params"]
    state = model_dict["state"]
    
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )
    run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")
    
    print(f"Number of local devices {len(jax.local_devices())}")    
    
    rng = jax.random.key(0)     #PRNGKey(0)
    rngs = np.stack([jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)
    
    chunks = []
    for chunk in rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=num_ensemble_members,
        pmap_devices=jax.local_devices()
    ):
        chunks.append(chunk)
    return xarray.combine_by_coords(chunks)

# 시각화 함수들
def plot_results(example_data: xarray.Dataset, predictions: xarray.Dataset,
                variable: str, level: int = 500, output_path: Optional[str] = None):
    """결과 시각화"""
    fig = plt.figure(figsize=(20, 10))
    time_indices = range(0, example_data.dims["time"], 6)
    
    for i, t in enumerate(time_indices):
        # 입력 데이터 플롯
        ax = fig.add_subplot(2, len(time_indices), i+1)
        plot_data = select(example_data, variable, level).isel(time=t)
        scaled_data = scale(plot_data, robust=True)
        im = ax.imshow(scaled_data, origin="lower")
        ax.set_title(f"Input t={t}h")
        plt.colorbar(im, ax=ax)
        
        # 예측 결과 플롯
        ax = fig.add_subplot(2, len(time_indices), len(time_indices)+i+1)
        plot_data = select(predictions.mean(dim="sample"), variable, level).isel(time=t)
        scaled_data = scale(plot_data, robust=True)
        im = ax.imshow(scaled_data, origin="lower")
        ax.set_title(f"Prediction t={t}h")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def main(args):
    """메인 실행 함수"""
    # 모델 로드
    model_dict = load_model(args.p, args.s)
    
    # 데이터 로드
    example_data = load_example_data(args.d)
    
    # 추론 실행
    predictions = run_inference(model_dict, example_data, args.v, args.n)
    
    # 결과 시각화
    plot_results(example_data, predictions, args.v, args.l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCast Inference")
    parser.add_argument("--p", type=str, required=True, help="Path to model parameters file")
    parser.add_argument("--s", type=str, default="Checkpoint", choices=["Checkpoint", "Random"], 
                       help="Source of model parameters")
    parser.add_argument("--d", type=str, required=True, help="Path to example dataset file")
    parser.add_argument("--v", type=str, default="2m_temperature", help="Variable to predict and plot")
    parser.add_argument("--l", type=int, default=500, help="Pressure level for 3D variables")
    parser.add_argument("--n", type=int, default=8, help="Number of ensemble members")
    
    args = parser.parse_args()
    main(args)

"""
python inference.py --p graphcast_1p0deg_mini_2019.npz --d era5-2019-1p0deg.nc --v 2m_temperature --l 500 --n 8
# 모든 옵션을 사용한 실행
python inference.py \
    --p graphcast_1p0deg_mini_2019.npz \  # 모델 파라미터 파일
    --s Checkpoint \                       # 소스 타입 (Checkpoint 또는 Random)
    --d era5-2019-1p0deg.nc \             # 데이터셋 파일
    --v 2m_temperature \                   # 예측할 변수
    --l 500 \                             # 기압 레벨 (hPa)
    --n 8                                 # 앙상블 멤버 수
"""