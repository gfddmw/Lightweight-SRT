import json
import re
import random

# 1. 读取全量数据
with open("Lightweight-SRT-main/processed/clean_indices.json", "r") as f:
    all_samples = json.load(f)

# 2. 提取唯一视频 ID (去掉括号后缀)
def get_base_id(sample_id):
    match = re.match(r"^(\d+)", sample_id)
    return match.group(1) if match else sample_id

base_ids = list(set(get_base_id(s) for s in all_samples))
random.seed(42) # 固定随机种子，保证每次划分结果一致
random.shuffle(base_ids)

# 3. 按 90% / 10% 划分 base ID
split_idx = int(len(base_ids) * 0.9)
train_base_ids = set(base_ids[:split_idx])
test_base_ids = set(base_ids[split_idx:])

# 4. 根据 base ID 分配全量样本
train_samples = [s for s in all_samples if get_base_id(s) in train_base_ids]
test_samples = [s for s in all_samples if get_base_id(s) in test_base_ids]

print(f"总样本数: {len(all_samples)}")
print(f"训练集样本数: {len(train_samples)}")
print(f"测试集样本数: {len(test_samples)}")

# 5. 保存
with open("Lightweight-SRT-main/processed/train_indices.json", "w") as f:
    json.dump(train_samples, f, indent=2)
with open("Lightweight-SRT-main/processed/test_indices.json", "w") as f:
    json.dump(test_samples, f, indent=2)

print("✅ 已成功生成 train_indices.json 和 test_indices.json")