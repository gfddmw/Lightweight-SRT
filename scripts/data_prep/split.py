import json
import re
import os

# 获取脚本所在目录，确保路径相对于项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def get_path(rel_path):
    return os.path.join(PROJECT_ROOT, rel_path)

# 1. 加载官方划分映射表 (nslt_2000.json)
with open(get_path("data/nslt_2000.json"), "r") as f:
    split_map = json.load(f)


# 2. 加载清洗后的索引列表
with open(get_path("processed/clean_indices.json"), "r") as f:
    all_samples = json.load(f)

train_samples = []
test_samples = []
missing_count = 0

# 3. 辅助函数：提取基础 ID (去掉末尾的括号编号，如 "00295(0)" -> "00295")
def get_base_id(sample_id):
    match = re.match(r"^(\d+)", sample_id)
    return match.group(1) if match else sample_id

# 4. 根据官方 subset 字段严格划分
for sample_id in all_samples:
    base_id = get_base_id(sample_id)
    if base_id in split_map:
        subset = split_map[base_id].get("subset", "train").strip().lower()
        if subset == "train":
            train_samples.append(sample_id)
        elif subset == "test":
            test_samples.append(sample_id)
        else:
            # 非 train/test 的样本（如 val）默认归入训练集
            train_samples.append(sample_id)
    else:
        # 若视频 ID 不在映射表中，默认归入训练集并记录
        train_samples.append(sample_id)
        missing_count += 1

print(f"📊 总样本数: {len(all_samples)}")
print(f"📈 训练集样本数: {len(train_samples)}")
print(f"📉 测试集样本数: {len(test_samples)}")
if missing_count > 0:
    print(f"⚠️ 未在 nslt_2000.json 中匹配的样本数: {missing_count} (已默认归入训练集)")

# 5. 保存划分结果
with open(get_path("processed/train_indices.json"), "w") as f:
    json.dump(train_samples, f, indent=2)
with open(get_path("processed/test_indices.json"), "w") as f:
    json.dump(test_samples, f, indent=2)

print("✅ 已成功按官方 subset 划分生成 train_indices.json 和 test_indices.json")
