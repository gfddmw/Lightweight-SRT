import json
import os

def create_wlasl_subsets(input_file='WLASL_v0.3.json'):
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 '{input_file}'。请确保该 JSON 文件在当前目录下。")
        return

    print(f"正在读取 {input_file} ...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 失败: {e}")
        return

    total_words = len(data)
    print(f"原始数据集包含 {total_words} 个单词 (Glosses)。")

    # 2. 核心逻辑：根据 'instances' 列表的长度（即视频样本数）进行降序排序
    # x['instances'] 是包含该单词所有视频信息的列表
    sorted_data = sorted(data, key=lambda x: len(x.get('instances', [])), reverse=True)

    # 3. 定义切分配置
    splits = {
        'WLASL100.json': 100,
        'WLASL300.json': 300
    }

    # 4. 执行切分并保存
    for filename, count in splits.items():
        if total_words < count:
            print(f"警告: 原始数据只有 {total_words} 个词，无法生成 {count} 个词的子集。")
            continue
            
        subset_data = sorted_data[:count]
        
        # 统计一下子集里的具体信息（可选，方便确认）
        min_samples = len(subset_data[-1]['instances'])
        max_samples = len(subset_data[0]['instances'])
        
        print(f"\n正在生成 {filename} ...")
        print(f"  - 包含前 {count} 个高频词")
        print(f"  - 样本数范围: 最多 {max_samples} 个, 最少 {min_samples} 个")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, indent=4)
            
        print(f"  - 已保存至 {filename}")

    print("\n所有处理完成！")

if __name__ == "__main__":
    create_wlasl_subsets()
