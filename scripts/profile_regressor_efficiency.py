import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import yaml
from thop import profile as thop_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classifier import CNN1DRegressor, LSTMRegressor, MambaRULRegressor, TransformerRegressor
from src.utils.checkpoint_utils import load_lightning_module_checkpoint


TARGET_MODELS = {"LSTMRegressor", "CNN1DRegressor", "TransformerRegressor", "MambaRegressor"}


def _results_root(repo_root: Path) -> Path:
    return repo_root / "results" / "bearing_rul"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_checkpoint_path(run_dir: Path, manifest: Dict) -> Optional[Path]:
    checkpoint_path = manifest.get("best_model_path") or manifest.get("classifier_checkpoint")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path

    checkpoint_dir = run_dir / "best_model_classifier"
    if checkpoint_dir.exists():
        ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
        if ckpts:
            return ckpts[-1]

    return None


def _classifier_name_from_manifest(manifest: Dict, run_dir: Path) -> Optional[str]:
    classifier_model = manifest.get("classifier_model")
    if classifier_model in TARGET_MODELS:
        return classifier_model

    model_name = manifest.get("model_name", "")
    if model_name in TARGET_MODELS:
        return model_name

    parent_name = run_dir.parent.name
    if parent_name in TARGET_MODELS:
        return parent_name

    return None


def _build_model(model_name: str, checkpoint_path: Path):
    if model_name == "LSTMRegressor":
        model = load_lightning_module_checkpoint(LSTMRegressor, checkpoint_path, map_location="cpu")
    elif model_name == "CNN1DRegressor":
        model = load_lightning_module_checkpoint(CNN1DRegressor, checkpoint_path, map_location="cpu")
    elif model_name == "TransformerRegressor":
        model = load_lightning_module_checkpoint(TransformerRegressor, checkpoint_path, map_location="cpu")
    elif model_name == "MambaRegressor":
        model = load_lightning_module_checkpoint(MambaRULRegressor, checkpoint_path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported regressor model: {model_name}")
    model.eval()
    return model


def _parameter_bytes(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))


def _buffer_bytes(model: torch.nn.Module) -> int:
    return int(sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()))


def _format_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for efficiency profiling, but no GPU is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _measure_throughput(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, float]:
    sample = sample.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(warmup_steps, 0)):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(max(measure_steps, 1)):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    total_samples = sample.shape[0] * max(measure_steps, 1)
    latency_ms_per_batch = (elapsed / max(measure_steps, 1)) * 1000.0
    latency_ms_per_sample = (elapsed / total_samples) * 1000.0
    throughput = total_samples / max(elapsed, 1e-12)
    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_per_batch": float(latency_ms_per_batch),
        "latency_ms_per_sample": float(latency_ms_per_sample),
    }


def _measure_peak_memory(model: torch.nn.Module, sample: torch.Tensor, device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "peak_forward_allocated_mb": None,
            "peak_forward_reserved_mb": None,
        }

    sample = sample.to(device)
    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(sample)
        torch.cuda.synchronize(device)

    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "peak_forward_allocated_mb": float(allocated),
        "peak_forward_reserved_mb": float(reserved),
    }


def _profile_run(
    run_dir: Path,
    checkpoint_path: Path,
    model_name: str,
    input_dim: int,
    window_size: int,
    device: torch.device,
    throughput_batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> Dict:
    model = _build_model(model_name, checkpoint_path)

    flops_sample = torch.randn(1, window_size, input_dim, dtype=torch.float32, device=device)
    flops_model = _build_model(model_name, checkpoint_path).to(device).eval()
    macs, _ = thop_profile(flops_model, inputs=(flops_sample,), verbose=False)
    flops = float(macs) * 2.0
    flops_model = flops_model.cpu()
    del flops_model

    throughput_sample = torch.randn(throughput_batch_size, window_size, input_dim, dtype=torch.float32)

    params = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_params = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    parameter_bytes = _parameter_bytes(model)
    buffer_bytes = _buffer_bytes(model)
    total_size_bytes = parameter_bytes + buffer_bytes

    throughput_metrics = _measure_throughput(
        model=model,
        sample=throughput_sample,
        device=device,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )
    peak_memory_metrics = _measure_peak_memory(
        model=model,
        sample=throughput_sample,
        device=device,
    )

    profile = {
        "model_name": model_name,
        "device": str(device),
        "input_dim": int(input_dim),
        "window_size": int(window_size),
        "throughput_batch_size": int(throughput_batch_size),
        "warmup_steps": int(warmup_steps),
        "measure_steps": int(measure_steps),
        "parameters": params,
        "trainable_parameters": trainable_params,
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "total_model_bytes": int(total_size_bytes),
        "parameter_mb": float(parameter_bytes / (1024 ** 2)),
        "buffer_mb": float(buffer_bytes / (1024 ** 2)),
        "total_model_mb": float(total_size_bytes / (1024 ** 2)),
        "macs": float(macs),
        "gmacs": float(macs / 1e9),
        "flops": float(flops),
        "gflops": float(flops / 1e9),
    }
    profile.update(throughput_metrics)
    profile.update(peak_memory_metrics)

    target_path = run_dir / "model_profile.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
    return profile


def _iter_classifier_runs(results_root: Path, datasets: Iterable[str]) -> Iterable[Path]:
    for dataset in datasets:
        dataset_root = results_root / dataset
        if not dataset_root.exists():
            continue
        for manifest_path in sorted(dataset_root.rglob("run_manifest.json")):
            yield manifest_path.parent


def main():
    parser = argparse.ArgumentParser(description="Profile regressor efficiency and update model_profile.json.")
    parser.add_argument("--datasets", nargs="+", default=["FEMTO", "XJTU-SY"])
    parser.add_argument("--models", nargs="+", default=sorted(TARGET_MODELS))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used for throughput and memory profiling.")
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--measure_steps", type=int, default=50)
    parser.add_argument("--run_id", type=str, default=None, help="Optional single run_id filter.")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    results_root = _results_root(repo_root)
    selected_models = set(args.models)
    device = _format_device(args.device)

    processed = 0
    skipped = 0

    for run_dir in _iter_classifier_runs(results_root, args.datasets):
        if args.run_id and run_dir.name != args.run_id and run_dir.name != f"run_{args.run_id}":
            continue

        manifest_path = run_dir / "run_manifest.json"
        config_path = run_dir / "run_configs.yaml"
        if not manifest_path.exists() or not config_path.exists():
            skipped += 1
            continue

        manifest = _load_json(manifest_path)
        model_name = _classifier_name_from_manifest(manifest, run_dir)
        if model_name is None or model_name not in selected_models:
            skipped += 1
            continue

        checkpoint_path = _resolve_checkpoint_path(run_dir, manifest)
        if checkpoint_path is None:
            skipped += 1
            continue

        config = _load_yaml(config_path)
        dataset_name = manifest["dataset"]
        dataset_cfg = config.get("datasets", {}).get(dataset_name, {})
        window_size = int(manifest.get("window_size", dataset_cfg.get("window_size")))
        input_dim = int(manifest.get("input_dim", 1))

        print(f"[Profile] {dataset_name} | {model_name} | {run_dir.name}")
        try:
            profile = _profile_run(
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                input_dim=input_dim,
                window_size=window_size,
                device=device,
                throughput_batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
            )
            print(
                f"[Profile] Saved {run_dir / 'model_profile.json'} | "
                f"gflops={profile['gflops']:.6f}, "
                f"throughput={profile['throughput_samples_per_sec']:.2f} samples/s, "
                f"model_mb={profile['total_model_mb']:.4f}"
            )
            processed += 1
        except Exception as exc:
            error_payload = {
                "model_name": model_name,
                "device": str(device),
                "profile_error": str(exc),
            }
            with (run_dir / "model_profile.json").open("w", encoding="utf-8") as handle:
                json.dump(error_payload, handle, indent=2, sort_keys=True)
            print(f"[Profile] Failed {run_dir.name}: {exc}")

    print(f"[Profile] Processed={processed} Skipped={skipped}")


if __name__ == "__main__":
    main()
