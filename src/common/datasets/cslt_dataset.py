import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data_utl

def _find_project_root():
    """自动查找项目根目录"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "processed").exists():
            return parent
    return Path(__file__).resolve().parents[3]

PROJECT_ROOT = _find_project_root()

class CSLTDataset(data_utl.Dataset):
    """
    用于连续手语识别与翻译 (CSLT) 的 Dataset。
    
    返回每个视频的变长多流特征 [Joint, Bone, Motion] 及其对应的教师模型监督信号
    与文本标签。该 Dataset 必须配合 `cslt_collate_fn` 一起使用以实现变长 Batch 级的 Padding。
    """
    def __init__(self, 
                 global_index_path="data/csl_daily/vocabularies/global_index.json", 
                 subset="train", 
                 in_channels=3,
                 **kwargs):
        super().__init__()
        
        # 路径解析
        if not os.path.isabs(global_index_path):
            self.global_index_path = PROJECT_ROOT / global_index_path
        else:
            self.global_index_path = Path(global_index_path)
            
        if not self.global_index_path.exists():
            raise FileNotFoundError(f"未找到全局数据索引文件: {self.global_index_path}")
            
        with open(self.global_index_path, 'r', encoding='utf-8') as f:
            self.global_index = json.load(f)
            
        # 根据子集（train / dev / test）筛选样本
        self.sample_ids = [vid for vid, meta in self.global_index.items() if meta.get('subset') == subset]
        self.subset = subset
        self.in_channels = in_channels
        
    def __len__(self):
        return len(self.sample_ids)
        
    def _load_npy(self, path):
        """加载 numpy 特征文件"""
        abs_path = PROJECT_ROOT / path if not os.path.isabs(path) else Path(path)
        if not abs_path.exists():
            raise FileNotFoundError(f"未找到特征数据文件: {abs_path}")
        return np.load(abs_path, allow_pickle=True).astype(np.float32)
        
    def __getitem__(self, idx):
        vid_id = self.sample_ids[idx]
        meta = self.global_index[vid_id]
        
        # 1. 加载骨架多流数据（形状为 [T, 42, 9]）
        skeleton_path = meta['skeleton']
        skeleton_data = self._load_npy(skeleton_path) # [T, 42, 9]
        
        # 分离多流数据并转置为 [C, T, V] = [3, T, 42]
        joints = skeleton_data[:, :, 0:3].transpose(2, 0, 1) # [3, T, 42]
        bones = skeleton_data[:, :, 3:6].transpose(2, 0, 1)  # [3, T, 42]
        motion = skeleton_data[:, :, 6:9].transpose(2, 0, 1) # [3, T, 42]
        
        # 如果只要 2D 坐标（例如只提取 x 和 y，剔除 z），做截断
        if self.in_channels == 2:
            joints = joints[:2]
            bones = bones[:2]
            motion = motion[:2]
            
        # 2. 加载强教师特征与 logits 监督信号
        teacher_feat = self._load_npy(meta['teacher_feat']) # [T_t, 1024]
        teacher_logits = self._load_npy(meta['teacher_logits']) # [T_t, 2001]
        
        # 3. 获取词与字符 IDs 标注
        gloss_ids = meta['gloss_ids']
        sentence_ids_word = meta['sentence_ids_word']
        sentence_ids_char = meta['sentence_ids_char']
        
        return {
            "vid": vid_id,
            "joints": torch.from_numpy(joints),
            "bones": torch.from_numpy(bones),
            "motion": torch.from_numpy(motion),
            "teacher_feat": torch.from_numpy(teacher_feat),
            "teacher_logits": torch.from_numpy(teacher_logits),
            "gloss_ids": torch.tensor(gloss_ids, dtype=torch.long),
            "sentence_ids_word": torch.tensor(sentence_ids_word, dtype=torch.long),
            "sentence_ids_char": torch.tensor(sentence_ids_char, dtype=torch.long),
        }

def cslt_collate_fn(batch):
    """
    变长手语序列的 Batch padding 整理函数。
    将骨骼、教师特征、中文及 Gloss 用 0 (或 <PAD>) padding 至 Batch 内最大长度。
    """
    input_lengths = []
    teacher_lengths = []
    gloss_lengths = []
    word_lengths = []
    char_lengths = []
    
    for item in batch:
        input_lengths.append(item['joints'].shape[1]) # T
        teacher_lengths.append(item['teacher_feat'].shape[0]) # T_t
        gloss_lengths.append(len(item['gloss_ids']))
        word_lengths.append(len(item['sentence_ids_word']))
        char_lengths.append(len(item['sentence_ids_char']))
        
    max_T = max(input_lengths)
    max_T_t = max(teacher_lengths)
    max_G = max(gloss_lengths)
    max_W = max(word_lengths)
    max_C = max(char_lengths)
    
    batch_size = len(batch)
    V = batch[0]['joints'].shape[2] # 42 关节数
    C = batch[0]['joints'].shape[0] # 通道数
    
    # 1. 零填充骨架流与教师监督信号
    padded_joints = torch.zeros(batch_size, C, max_T, V)
    padded_bones = torch.zeros(batch_size, C, max_T, V)
    padded_motion = torch.zeros(batch_size, C, max_T, V)
    
    padded_teacher_feats = torch.zeros(batch_size, max_T_t, 1024)
    padded_teacher_logits = torch.zeros(batch_size, max_T_t, 2001)
    
    # 2. 用 ID 0 (<PAD>) 填充文本和 Gloss 标记序列
    padded_gloss_ids = torch.zeros(batch_size, max_G, dtype=torch.long)
    padded_word_ids = torch.zeros(batch_size, max_W, dtype=torch.long)
    padded_char_ids = torch.zeros(batch_size, max_C, dtype=torch.long)
    
    vids = []
    for i, item in enumerate(batch):
        vids.append(item['vid'])
        
        # 填充多流骨架点
        t = input_lengths[i]
        padded_joints[i, :, :t, :] = item['joints']
        padded_bones[i, :, :t, :] = item['bones']
        padded_motion[i, :, :t, :] = item['motion']
        
        # 填充教师层
        t_t = teacher_lengths[i]
        padded_teacher_feats[i, :t_t, :] = item['teacher_feat']
        padded_teacher_logits[i, :t_t, :] = item['teacher_logits']
        
        # 填充文本映射
        padded_gloss_ids[i, :gloss_lengths[i]] = item['gloss_ids']
        padded_word_ids[i, :word_lengths[i]] = item['sentence_ids_word']
        padded_char_ids[i, :char_lengths[i]] = item['sentence_ids_char']
        
    return {
        "vids": vids,
        "joints": padded_joints.unsqueeze(-1), # [B, C, T, V, 1] 兼容标准 ST-GCN 多流输入
        "bones": padded_bones.unsqueeze(-1),   # [B, C, T, V, 1]
        "motion": padded_motion.unsqueeze(-1), # [B, C, T, V, 1]
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "teacher_feats": padded_teacher_feats,
        "teacher_logits": padded_teacher_logits,
        "teacher_lengths": torch.tensor(teacher_lengths, dtype=torch.long),
        "gloss_ids": padded_gloss_ids,
        "gloss_lengths": torch.tensor(gloss_lengths, dtype=torch.long),
        "sentence_ids_word": padded_word_ids,
        "sentence_ids_char": padded_char_ids,
        "word_lengths": torch.tensor(word_lengths, dtype=torch.long),
        "char_lengths": torch.tensor(char_lengths, dtype=torch.long),
    }

if __name__ == '__main__':
    import sys
    
    # Windows 编码修复，防止在 GBK 控制台打印 UTF-8 或是 Unicode 报错，并开启行缓冲确保实时输出
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    print("=" * 60)
    print("开始对 CSLTDataset 和 cslt_collate_fn 进行读取测试...")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    try:
        # 1. 实例化 Dataset
        dataset = CSLTDataset(
            global_index_path="data/csl_daily/vocabularies/global_index.json",
            subset="train",
            in_channels=3
        )
        print(f"成功加载 CSLTDataset，总计 {len(dataset)} 个训练样本。")
        
        # 2. 实例化 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=cslt_collate_fn
        )
        
        # 3. 提取第一个批次
        batch = next(iter(dataloader))
        
        print("\n第一个 Batch 数据维度校验:")
        print(f"  - 样本 VIDs: {batch['vids']}")
        print(f"  - joints 形状 (B, C, T, V, 1): {batch['joints'].shape}")
        print(f"  - bones 形状  (B, C, T, V, 1): {batch['bones'].shape}")
        print(f"  - motion 形状 (B, C, T, V, 1): {batch['motion'].shape}")
        print(f"  - input_lengths  (B): {batch['input_lengths'].tolist()}")
        print(f"  - teacher_feats  (B, T_t, 1024): {batch['teacher_feats'].shape}")
        print(f"  - teacher_logits (B, T_t, 2001): {batch['teacher_logits'].shape}")
        print(f"  - teacher_lengths(B): {batch['teacher_lengths'].tolist()}")
        print(f"  - gloss_ids      (B, G): {batch['gloss_ids'].shape}")
        print(f"  - gloss_lengths  (B): {batch['gloss_lengths'].tolist()}")
        print(f"  - word_lengths   (B): {batch['word_lengths'].tolist()}")
        print(f"  - char_lengths   (B): {batch['char_lengths'].tolist()}")
        
        print("\n所有通道与时序 Padding 校验成功！数据管线完全就绪。")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试遇到异常: {str(e)}")
        import traceback
        traceback.print_exc()
