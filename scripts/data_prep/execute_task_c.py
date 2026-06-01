"""
execute_task_c.py
=================
Task C 执行脚本

按顺序执行：
1. build_csl_daily_vocab.py
2. generate_csl_daily_splits.py

Usage:
python scripts/data_prep/execute_task_c.py
"""

import os
import sys
from pathlib import Path

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"执行命令: {cmd}")
    print('='*60)
    result = os.system(cmd)
    if result != 0:
        print(f"命令执行失败，退出码: {result}")
        sys.exit(result)

def main():
    print("开始执行 Task C：语言系统与全局索引构建")
    print("="*60)

    # 设置基础路径
    csl_root = "D:/nju/2/SLT/dataset/CSL-Daily/"
    csl_config_dir = "configs/csl_daily/"

    # 确保目录存在
    Path(csl_config_dir).mkdir(parents=True, exist_ok=True)

    # 步骤 1: 构建词表
    print("\n步骤 1: 构建词表...")
    run_command(f"python scripts/data_prep/build_csl_daily_vocab.py "
                f"--csl_root {csl_root} "
                f"--gloss_output {csl_config_dir}gloss_vocab.json "
                f"--text_output {csl_config_dir}text_vocab.json "
                f"--splits_output {csl_config_dir}dataset_splits.json")

    # 步骤 2: 构建全局索引
    print("\n步骤 2: 构建全局索引...")
    run_command(f"python scripts/data_prep/generate_csl_daily_splits.py "
                f"--csl_root {csl_root} "
                f"--vocab_root {csl_config_dir} "
                f"--splits_file {csl_config_dir}dataset_splits.json "
                f"--output {csl_config_dir}global_index.json "
                f"--skeleton_dir ../{csl_root}skeletons/ "
                f"--teacher_feat_dir ../{csl_root}processed-csl_daily/teacher_features/ "
                f"--teacher_logits_dir ../{csl_root}processed-csl_daily/teacher_logits/")

    print("\n" + "="*60)
    print("Task C 执行完成！")
    print("="*60)
    print("生成的文件:")
    print(f"  - {csl_config_dir}gloss_vocab.json")
    print(f"  - {csl_config_dir}text_vocab.json")
    print(f"  - {csl_config_dir}dataset_splits.json")
    print(f"  - {csl_config_dir}global_index.json")
    print(f"  - {csl_config_dir}dataset_summary.json")

if __name__ == "__main__":
    main()