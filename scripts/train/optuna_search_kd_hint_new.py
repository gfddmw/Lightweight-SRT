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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/student/st_gcn/wlasl2000/train_kd_hint_optuna_new.yaml"
TRAIN_SCRIPT = PROJECT_ROOT / "src/student_model/distillation/recognition_kd_hint_new.py"
TMP_DIR = PROJECT_ROOT / "work_dir/optuna_tmp_configs"


def new_load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def new_dump_yaml(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def new_parse_top1(stdout_text: str) -> float:
    matches = re.findall(r"Top1:\s*([0-9]+(?:\.[0-9]+)?)%", stdout_text)
    if not matches:
        return 0.0
    return float(matches[-1])


def new_run_trial(base_config_path: Path, trial: optuna.Trial, fast_epochs: int, device: str):
    base_cfg = new_load_yaml(base_config_path)
    cfg = deepcopy(base_cfg)

    kd_temperature = trial.suggest_float("kd_temperature", 2.0, 10.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    cfg["kd_temperature"] = kd_temperature
    cfg["weight_decay"] = weight_decay
    cfg["num_epoch"] = fast_epochs
    cfg["eval_interval"] = 1

    trial_work_dir = PROJECT_ROOT / f"work_dir/recognition/wlasl2000/optuna_kd_hint_new/trial_{trial.number}"
    cfg["work_dir"] = str(trial_work_dir).replace("\\", "/")

    trial_cfg_path = TMP_DIR / f"trial_{trial.number}.yaml"
    new_dump_yaml(trial_cfg_path, cfg)

    command = [sys.executable, str(TRAIN_SCRIPT), "--config", str(trial_cfg_path)]
    if torch.cuda.is_available():
        command.extend(["--device", device, "--use_gpu", "True"])
    else:
        command.extend(["--use_gpu", "False"])
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Trial {trial.number} failed with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    top1 = new_parse_top1(result.stdout)
    trial.set_user_attr("stdout_tail", result.stdout[-4000:])
    return top1


def main():
    parser = argparse.ArgumentParser(description="Optuna search for KD-Hint hyperparameters")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--fast_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="0", help="pass-through for --device of training script")
    parser.add_argument("--study_name", type=str, default="wlasl2000_kd_hint_new")
    parser.add_argument("--storage", type=str, default=None, help="e.g. sqlite:///work_dir/optuna.db")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: new_run_trial(config_path, trial, fast_epochs=args.fast_epochs, device=args.device),
        n_trials=args.trials,
    )

    print("\n===== Optuna Best Trial =====")
    print(f"Best Top1: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
