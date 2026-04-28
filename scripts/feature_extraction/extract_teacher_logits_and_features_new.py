import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.teacher_model.architecture.pytorch_i3d import InceptionI3d
from src.common.transforms import videotransforms
from src.common.datasets.nslt_dataset_all import NSLT as Dataset


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../data", "WLASL2000"))
JSON_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../../data", "nslt_2000.json"))
TEACHER_WEIGHTS = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../weights/teacher", "nslt_2000_018216_0.448072.pt")
)
LOGITS_SAVE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../processed/logits"))
FEATURES_SAVE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../processed/teacher_features"))


def new_ensure_output_dirs():
    os.makedirs(LOGITS_SAVE_DIR, exist_ok=True)
    os.makedirs(FEATURES_SAVE_DIR, exist_ok=True)


def new_load_teacher_model(device: torch.device) -> InceptionI3d:
    model = InceptionI3d(2000, in_channels=3)

    if os.path.exists(TEACHER_WEIGHTS):
        state_dict = torch.load(TEACHER_WEIGHTS, map_location=device)
        if "module." in list(state_dict.keys())[0]:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_state_dict[key.replace("module.", "")] = value
            state_dict = new_state_dict
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: teacher weights not found at {TEACHER_WEIGHTS}. Using random init for debugging.")

    model.to(device)
    model.eval()
    return model


def new_extract_teacher_logits_and_feature_vectors(model: InceptionI3d, inputs: torch.Tensor):
    with torch.no_grad():
        per_frame_logits = model(inputs)
        video_logits = torch.mean(per_frame_logits, dim=2)

        pooled_feature = model.extract_features(inputs)
        # 增加时间维度的平均池化，确保输出始终为 (N, 1024)
        feature_vector = torch.mean(pooled_feature, dim=2).flatten(start_dim=1)

    return video_logits, feature_vector


def new_save_teacher_outputs(video_id: str, video_logits: torch.Tensor, feature_vector: torch.Tensor):
    logits_path = os.path.join(LOGITS_SAVE_DIR, f"{video_id}.npy")
    features_path = os.path.join(FEATURES_SAVE_DIR, f"{video_id}.npy")

    np.save(logits_path, video_logits.cpu().numpy().flatten())
    np.save(features_path, feature_vector.cpu().numpy().flatten())


def new_build_dataloader(subset: str, batch_size: int):
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(JSON_FILE, subset, DATA_ROOT, "rgb", test_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataset, dataloader


def new_run_extraction(subset="train", batch_size=1, device_name="cuda"):
    new_ensure_output_dirs()

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Running extraction on {device} | subset={subset}")

    dataset, dataloader = new_build_dataloader(subset=subset, batch_size=batch_size)
    print(f"Total samples to process: {len(dataset)}")

    model = new_load_teacher_model(device)

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            inputs, _labels, video_id = batch
            inputs = inputs.to(device)

            video_logits, feature_vector = new_extract_teacher_logits_and_feature_vectors(model, inputs)

            for inner_index, vid in enumerate(video_id):
                new_save_teacher_outputs(
                    video_id=str(vid),
                    video_logits=video_logits[inner_index : inner_index + 1],
                    feature_vector=feature_vector[inner_index : inner_index + 1],
                )

            if index % 100 == 0:
                print(f"Progress: {index}/{len(dataloader)} | latest={video_id[0]}")

    print(f"Finished teacher logits + feature extraction for subset={subset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract teacher logits and pooled teacher features")
    parser.add_argument("--subset", type=str, default="train", choices=["train", "test", "val"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    new_run_extraction(subset=args.subset, batch_size=args.batch_size, device_name=args.device)
