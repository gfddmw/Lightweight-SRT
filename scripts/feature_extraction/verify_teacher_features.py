import os
import json
import argparse
import numpy as np

def verify_extracted_data():
    parser = argparse.ArgumentParser(description="验证 CSL-Daily 教师特征与 Logits 的正确性与完整性")
    parser.add_argument("--csl_json", type=str, default="../../data/CSL/csl-daily.json", help="CSL-Daily JSON 标注文件路径")
    parser.add_argument("--feat_dir", type=str, default="../../processed/csl_daily/teacher_features", help="特征目录")
    parser.add_argument("--logits_dir", type=str, default="../../processed/csl_daily/teacher_logits", help="Logits目录")
    parser.add_argument("--expected_feat_dim", type=int, default=1024, help="期望的特征维度")
    parser.add_argument("--expected_class_dim", type=int, default=2001, help="期望的分类 Logits 维度 (教师模型)")
    parser.add_argument("--fix", action="store_true", help="是否自动修复维度异常 (Squeeze 掉多余的维度 1，并删除损坏文件以便重提)")
    args = parser.parse_args()

    # 1. 确保路径为绝对路径
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csl_json_path = os.path.abspath(os.path.join(current_dir, args.csl_json))
    feat_dir = os.path.abspath(os.path.join(current_dir, args.feat_dir))
    logits_dir = os.path.abspath(os.path.join(current_dir, args.logits_dir))

    print("=" * 60)
    print(" CSL-Daily 教师特征与 Logits 数据验证工具")
    print("=" * 60)
    print(f"标注 JSON 路径: {csl_json_path}")
    print(f"特征保存目录: {feat_dir}")
    print(f"Logits保存目录: {logits_dir}")
    print("-" * 60)

    # 2. 检查基本文件与文件夹是否存在
    if not os.path.exists(csl_json_path):
        print(f"[Error] 标注 JSON 文件不存在: {csl_json_path}")
        return
    if not os.path.exists(feat_dir):
        print(f"[Error] 特征目录不存在: {feat_dir}")
        return
    if not os.path.exists(logits_dir):
        print(f"[Error] Logits 目录不存在: {logits_dir}")
        return

    # 3. 读取 JSON 获取全部视频列表
    with open(csl_json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    video_names = sorted(list(data_dict.keys()))
    total_videos = len(video_names)
    print(f"JSON 中声明的视频总数: {total_videos}")

    # 4. 统计变量
    missing_feats = []
    missing_logits = []
    corrupted_feats = []
    corrupted_logits = []
    mismatched_lengths = []
    dimension_errors = []
    nan_inf_errors = []
    
    passed_count = 0

    print("\n开始扫描并验证每个视频的特征和 Logits (这可能需要 1~2 分钟)...")
    
    # 5. 循环验证每个视频
    for idx, video_name in enumerate(video_names):
        feat_path = os.path.join(feat_dir, f"{video_name}.npy")
        logits_path = os.path.join(logits_dir, f"{video_name}.npy")
        
        has_feat = os.path.exists(feat_path)
        has_logits = os.path.exists(logits_path)

        # 检查文件缺失
        if not has_feat:
            missing_feats.append(video_name)
        if not has_logits:
            missing_logits.append(video_name)
        if not has_feat or not has_logits:
            continue

        # 尝试加载文件并验证内容
        error_flag = False
        
        # 5.1 验证特征文件
        try:
            feat_arr = np.load(feat_path)
        except Exception as e:
            corrupted_feats.append((video_name, f"加载异常: {e}"))
            if args.fix:
                for path in [feat_path, logits_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
            continue
            
        # 5.2 验证 Logits 文件
        try:
            logits_arr = np.load(logits_path)
        except Exception as e:
            corrupted_logits.append((video_name, f"加载异常: {e}"))
            if args.fix:
                for path in [feat_path, logits_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
            continue

        # 5.3 验证维度 (Shape) 与自动修复
        feat_shape_error = len(feat_arr.shape) != 2 or feat_arr.shape[1] != args.expected_feat_dim
        logits_shape_error = len(logits_arr.shape) != 2 or logits_arr.shape[1] != args.expected_class_dim
        
        if args.fix and (feat_shape_error or logits_shape_error):
            fixed_any = False
            # 自动修复特征维度: (T, 1, 1024) -> (T, 1024)
            if len(feat_arr.shape) == 3 and feat_arr.shape[1] == 1 and feat_arr.shape[2] == args.expected_feat_dim:
                feat_arr = feat_arr.squeeze(1)
                np.save(feat_path, feat_arr)
                feat_shape_error = False
                fixed_any = True
            # 自动修复 Logits 维度: (T, 1, 1296) -> (T, 1296)
            if len(logits_arr.shape) == 3 and logits_arr.shape[1] == 1 and logits_arr.shape[2] == args.expected_class_dim:
                logits_arr = logits_arr.squeeze(1)
                np.save(logits_path, logits_arr)
                logits_shape_error = False
                fixed_any = True
            
            if fixed_any:
                pass

        if feat_shape_error:
            dimension_errors.append((video_name, f"特征 shape 异常: {feat_arr.shape}，期望为 (T, {args.expected_feat_dim})"))
            error_flag = True
        if logits_shape_error:
            dimension_errors.append((video_name, f"Logits shape 异常: {logits_arr.shape}，期望为 (T, {args.expected_class_dim})"))
            error_flag = True

        if error_flag:
            continue

        # 5.4 验证时序长度一致性
        if feat_arr.shape[0] != logits_arr.shape[0]:
            mismatched_lengths.append((video_name, f"时序长度不匹配: 特征长 {feat_arr.shape[0]} != Logits长 {logits_arr.shape[0]}"))
            continue

        # 5.5 验证是否有 NaN 或 Inf 值
        if np.isnan(feat_arr).any() or np.isinf(feat_arr).any():
            nan_inf_errors.append((video_name, "特征包含 NaN 或 Inf"))
            continue
        if np.isnan(logits_arr).any() or np.isinf(logits_arr).any():
            nan_inf_errors.append((video_name, "Logits 包含 NaN 或 Inf"))
            continue

        # 全部通过
        passed_count += 1

        # 每 2000 个视频打印一次心跳，证明脚本在正常扫描
        if (idx + 1) % 2000 == 0:
            print(f"已校验进度: [{(idx + 1)}/{total_videos}] ...")

    # 6. 输出结果报告
    print("\n" + "=" * 60)
    print(" 验证结果报告")
    print("=" * 60)
    print(f"数据完好且通过校验样本数: {passed_count} / {total_videos} ({passed_count / total_videos * 100:.2f}%)")
    
    # 报告异常详情
    total_errors = len(missing_feats) + len(missing_logits) + len(corrupted_feats) + \
                   len(corrupted_logits) + len(dimension_errors) + len(mismatched_lengths) + len(nan_inf_errors)
                   
    if total_errors == 0:
        print("\n🎉 恭喜！未检测到任何损坏或缺失文件，全部提取正确且完整！")
    else:
        print(f"\n⚠️ 警告：检测到共计 {total_errors} 处异常！")
        
        if missing_feats:
            print(f"\n[缺失特征文件] 数量: {len(missing_feats)} (展示前5个):")
            for name in missing_feats[:5]:
                print(f"  - {name}.npy")
                
        if missing_logits:
            print(f"\n[缺失 Logits 文件] 数量: {len(missing_logits)} (展示前5个):")
            for name in missing_logits[:5]:
                print(f"  - {name}.npy")
                
        if corrupted_feats:
            print(f"\n[损坏特征文件] 数量: {len(corrupted_feats)} (展示前5个):")
            for name, reason in corrupted_feats[:5]:
                print(f"  - {name}.npy: {reason}")
                
        if corrupted_logits:
            print(f"\n[损坏 Logits 文件] 数量: {len(corrupted_logits)} (展示前5个):")
            for name, reason in corrupted_logits[:5]:
                print(f"  - {name}.npy: {reason}")
                
        if dimension_errors:
            print(f"\n[维度不正确样本] 数量: {len(dimension_errors)} (展示前5个):")
            for name, reason in dimension_errors[:5]:
                print(f"  - {name}.npy: {reason}")
                
        if mismatched_lengths:
            print(f"\n[特征与 Logits 长度不一致] 数量: {len(mismatched_lengths)} (展示前5个):")
            for name, reason in mismatched_lengths[:5]:
                print(f"  - {name}.npy: {reason}")
                
        if nan_inf_errors:
            print(f"\n[包含 NaN/Inf 非法数值] 数量: {len(nan_inf_errors)} (展示前5个):")
            for name, reason in nan_inf_errors[:5]:
                print(f"  - {name}.npy: {reason}")
                
    print("=" * 60)

if __name__ == "__main__":
    verify_extracted_data()
