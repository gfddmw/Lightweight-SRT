#!/usr/bin/env python
import argparse
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import optuna
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/student/st_gcn/wlasl2000/train_multistream_kd_hint_optuna.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src/student_model/distillation/recognition_kd_multistream_hint.py"
TMP_DIR = PROJECT_ROOT / "work_dir/optuna_multistream_tmp_configs"


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def parse_top1(stdout_text: str) -> float:
    matches = re.findall(r"Top1:\s*([0-9]+(?:\.[0-9]+)?)%", stdout_text)
    if not matches:
        return 0.0
    return float(matches[-1])


def run_trial(base_config_path: Path, trial: optuna.Trial, fast_epochs: int, device: str):
    base_cfg = load_yaml(base_config_path)
    cfg = deepcopy(base_cfg)

    # --- Hyperparameter Search Space (Refined by Model-Hyperparameter-Tuning Skill) ---
    
    # 1. Distillation Weights - Categorical & Float Mix for better coverage
    cfg["kd_alpha"] = trial.suggest_float("kd_alpha", 0.4, 0.9)
    cfg["kd_temperature"] = trial.suggest_categorical("kd_temperature", [2.0, 3.0, 4.0, 6.0, 8.0])
    
    # MSE Hint Loss: Sensitive, use log-uniform
    cfg["hint_weight"] = trial.suggest_float("hint_weight", 1e-3, 1.0, log=True)
    
    # SP Loss: Global topology, needs larger range
    cfg["sp_weight"] = trial.suggest_float("sp_weight", 1.0, 100.0, log=True)
    
    # 2. Multi-Stream Specific
    cfg["aux_loss_weight"] = trial.suggest_float("aux_loss_weight", 0.0, 0.5)
    
    # 3. Optimization - LR is the most critical
    cfg["base_lr"] = trial.suggest_float("base_lr", 1e-4, 0.2, log=True)
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Fixed for fast trial
    cfg["num_epoch"] = fast_epochs
    cfg["eval_interval"] = 1  # Need evaluation every epoch for pruning
    
    trial_work_dir = PROJECT_ROOT / f"work_dir/recognition/wlasl2000/optuna_multistream_kd/trial_{trial.number}"
    cfg["work_dir"] = str(trial_work_dir).replace("\\", "/")

    trial_cfg_path = TMP_DIR / f"trial_{trial.number}.yaml"
    dump_yaml(trial_cfg_path, cfg)

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(trial_cfg_path)]
    if torch.cuda.is_available():
        command.extend(["--device", device])
    
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: {trial.params}")
    
    # Start subprocess and capture output in real-time if possible, 
    # but for simplicity we'll parse at the end. 
    # To support pruning, we'd need to parse logs epoch by epoch.
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
    )

    last_top1 = 0.0
    epoch_count = 0
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end="") # Forward output to console
        
        # Parse Top1 from line
        if "Top1:" in line:
            match = re.search(r"Top1:\s*([0-9]+(?:\.[0-9]+)?)%", line)
            if match:
                last_top1 = float(match.group(1))
                epoch_count += 1
                trial.report(last_top1, epoch_count)
                
                # Check for pruning
                if trial.should_prune():
                    process.terminate()
                    print(f"\n[Optuna] Trial {trial.number} pruned at epoch {epoch_count}")
                    raise optuna.exceptions.TrialPruned()

    process.wait()

    if process.returncode != 0 and not trial.should_prune():
        print(f"Trial {trial.number} failed with return code {process.returncode}!")
        return 0.0

    return last_top1


def main():
    parser = argparse.ArgumentParser(description="Optuna search for Multi-Stream KD hyperparameters")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--fast_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--study_name", type=str, default="wlasl2000_multistream_kd")
    parser.add_argument("--storage", type=str, default="sqlite:///work_dir/optuna_multistream.db")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    
    # Ensure work dir for DB
    Path("work_dir").mkdir(exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler() # Use TPE for better search efficiency
    )
    
    try:
        study.optimize(
            lambda trial: run_trial(config_path, trial, fast_epochs=args.fast_epochs, device=args.device),
            n_trials=args.trials,
        )
    except KeyboardInterrupt:
        print("Search interrupted by user.")

    print("\n" + "="*30)
    print("===== Optuna Best Trial =====")
    print(f"Best Top1: {study.best_value:.4f}%")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*30)


if __name__ == "__main__":
    main()
