import os
import argparse
import yaml
import torch
import numpy as np

from src.utils.data_helper import get_data_module
from src.evaluation import TimeSeriesEvaluator
from src.baselines import TimeGAN, TimeVAE, DiffusionTS, TimeFlow
from flowmatchPdM.flowmatch_model import FlowMatchPdM

def load_generator(model_name, checkpoint_path, input_dim, window_size, config):
    """Dynamically loads the trained generator weights."""
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    if model_name == "TimeVAE":
        return TimeVAE.load_from_checkpoint(checkpoint_path, input_dim=input_dim, window_size=window_size)
    elif model_name == "TimeGAN":
        return TimeGAN.load_from_checkpoint(checkpoint_path, input_dim=input_dim, window_size=window_size)
    elif model_name == "DiffusionTS":
        return DiffusionTS.load_from_checkpoint(checkpoint_path, input_dim=input_dim, window_size=window_size)
    elif model_name == "TimeFlow":
        return TimeFlow.load_from_checkpoint(checkpoint_path, input_dim=input_dim, window_size=window_size)
    elif model_name == "FlowMatch":
        return FlowMatchPdM.load_from_checkpoint(checkpoint_path, input_dim=input_dim, window_size=window_size, config=config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Generate Data and Run Thorough Evaluation")
    parser.add_argument("--track", type=str, required=True, help="engine_rul, bearing_rul, bearing_fault")
    parser.add_argument("--dataset", type=str, required=True, help="CMAPSS, CWRU, etc.")
    parser.add_argument("--model", type=str, required=True, help="TimeGAN, TimeVAE, DiffusionTS, TimeFlow, FlowMatch")
    parser.add_argument("--run_id", type=str, required=True, help="Specific run folder, e.g., run1_20260308_1427")
    args = parser.parse_args()

    # 1. Locate the Run Directory
    run_dir = os.path.join("results", args.track, args.dataset, args.model, args.run_id)
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load the exact config used for this run
    config_path = os.path.join(run_dir, "run_configs.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output paths
    gen_data_dir = os.path.join(run_dir, "generator_datas")
    eval_dir = os.path.join(run_dir, "evaluation_results")
    ckpt_dir = os.path.join(run_dir, "best_models_generator")

    # Find the .ckpt file
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    checkpoint_path = os.path.join(ckpt_dir, ckpt_files[0])

    # 2. Extract Real Minority Data
    window_size = config['datasets']['window_size_engine'] if 'rul' in args.track else config['datasets']['window_size_bearing']
    input_dim = 14 if args.track == "engine_rul" else (2 if args.track == "bearing_rul" else 1)
    
    dm = get_data_module(track=args.track, dataset_name=args.dataset, window_size=window_size, batch_size=256)
    dm.prepare_data()
    dm.setup(stage='fit')
    
    minority_ds = dm.get_minority_dataset(rul_threshold_ratio=config['datasets']['rul_threshold'])
    
    # Collate real data into a single numpy array for evaluation
    print(f"Extracting real minority data for evaluation...")
    real_data_list = []
    conditions_list = []
    for x, y in minority_ds:
        real_data_list.append(x.numpy())
        conditions_list.append(y.numpy())
    
    real_data = np.stack(real_data_list)
    conditions = torch.tensor(np.stack(conditions_list)).float()
    
    num_samples = len(real_data)
    print(f"Real data shape: {real_data.shape}")

    # 3. Load Model & Generate Synthetic Data
    model = load_generator(args.model, checkpoint_path, input_dim, window_size, config['generative'].get('ss_flowmatch', {}))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n⚙️ Generating {num_samples} synthetic samples using {args.model}...")
    with torch.no_grad():
        if args.model == "FlowMatch":
            # FlowMatch specifically requires the physics conditions (e.g., RUL) to drive the Harmonic Prior
            cond_input = conditions.unsqueeze(-1).to(device) if conditions.dim() == 1 else conditions.to(device)
            synthetic_tensor = model.generate(conditions=cond_input, num_samples=num_samples)
        else:
            # Standard baselines just take num_samples
            synthetic_tensor = model.generate(num_samples=num_samples)
            
    synthetic_data = synthetic_tensor.cpu().numpy()
    
    # Save the synthetic arrays so you don't have to re-generate them later
    syn_save_path = os.path.join(gen_data_dir, "synthetic_data.npy")
    real_save_path = os.path.join(gen_data_dir, "real_minority_data.npy")
    np.save(syn_save_path, synthetic_data)
    np.save(real_save_path, real_data)
    print(f"💾 Saved raw synthetic and real data arrays to {gen_data_dir}")

    # 4. Execute the Thorough Evaluation Suite
    evaluator = TimeSeriesEvaluator(
        real_data=real_data, 
        synthetic_data=synthetic_data, 
        save_dir=eval_dir
    )
    
    evaluator.run_full_suite()

if __name__ == "__main__":
    main()