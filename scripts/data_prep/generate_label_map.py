import json
import re

# 1. 读取原始标签文件
with open("Lightweight-SRT-main/data/WLASL_v0.3.json", "r", encoding="utf-8") as f:
    wlasl_data = json.load(f)

# 2. 读取你的干净索引
with open("Lightweight-SRT-main/processed/clean_indices.json", "r") as f:
    clean_indices = json.load(f)

# 3. 清洗索引，去掉 "(1)" 这样的后缀，只保留纯数字部分
# 比如 "00592(1)" -> 592, "00295" -> 295
clean_ids_int = set()
for idx in clean_indices:
    # 用正则提取开头的数字部分
    match = re.match(r"^(\d+)", idx)
    if match:
        clean_ids_int.add(int(match.group(1)))

print(f"清洗后的唯一视频 ID 数量: {len(clean_ids_int)}")

# 4. 提取所有的 gloss，并建立全局的 "单词" -> "数字ID" 的映射
all_glosses = sorted(set(item["gloss"] for item in wlasl_data))
gloss_to_int = {gloss: idx for idx, gloss in enumerate(all_glosses)}
real_num_class = len(all_glosses)

print(f"📊 WLASL 数据集总共包含 {real_num_class} 个类别")

# 5. 根据你的 clean_indices 生成映射文件
label_map = {}
found_count = 0

for item in wlasl_data:
    gloss = item["gloss"]
    int_label = gloss_to_int[gloss]
    
    for instance in item["instances"]:
        video_id = instance["video_id"]
        
        # 检查这个样本是否在你的 clean_indices 里
        if int(video_id) in clean_ids_int:
            # 保持前导零的格式作为 key (对于有括号的，也保留括号)
            # 例如：匹配到 592，就把 "00592" 和 "00592(1)" 都加上
            base_str_id = str(int(video_id)).zfill(5)
            
            # 找出所有以这个 base_str_id 开头的原始索引
            for original_idx in clean_indices:
                if original_idx.startswith(base_str_id):
                    label_map[original_idx] = int_label
                    found_count += 1

print(f"🔍 在你的 {len(clean_indices)} 个样本中，成功匹配到了 {found_count} 个样本的真实标签")

if found_count == 0:
    print("❌ 匹配失败！可能 video_id 和索引的对应关系不是简单的补零，请检查")
else:
    # 统计这批样本实际覆盖了多少个类别
    used_classes = sorted(set(label_map.values()))
    print(f"🎯 你的样本实际覆盖了 {len(used_classes)} 个不同的手语类别")
    print(f"   类别 ID 范围: {min(used_classes)} ~ {max(used_classes)}")
    
    # 保存
    out_path = "Lightweight-SRT-main/processed/label_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"✅ 已保存真实的标签映射到 {out_path}")