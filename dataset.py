# dataset.py
# BUVFolderDataset with:
#   - VFlip ✅
#   - Discrete small Rotation (±5–15°) ✅
#   - NO scale jitter
#   - NO color jitter
#   - NO HFlip
#   - NO random crop
#
# Folder structure:
#   root_dir/
#     benign/<video_id>/*.png|jpg|jpeg
#     malignant/<video_id>/*.png|jpg|jpeg
#
# Returns:
#   train: (clip, label) where clip is [S, F, C, H, W]
#   test : (clip, label) where clip is [S, F, C, H, W] with center sampling

import json
import os
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BUVFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        video_ids: Optional[List[str]] = None,
        segment_count: int = 6,
        frames_per_segment: int = 3,
        img_size: int = 224,
        label_map: Optional[dict] = None,
        augment: bool = False,
        test_mode: bool = False,
        # Allowed augments:
        enable_vflip: bool = True,
        enable_rotation: bool = True,
        rotation_angles: Tuple[int, ...] = (-15, -10, -5, 5, 10, 15),
        # Debug
        verbose_errors: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.video_ids = set(video_ids) if video_ids is not None else None

        self.segment_count = int(segment_count)
        self.frames_per_segment = int(frames_per_segment)
        self.img_size = int(img_size)

        self.augment = bool(augment)
        self.test_mode = bool(test_mode)

        self.enable_vflip = bool(enable_vflip)
        self.enable_rotation = bool(enable_rotation)
        self.rotation_angles = tuple(int(a) for a in rotation_angles)
        self.verbose_errors = bool(verbose_errors)

        if label_map is None:
            label_map = {"benign": 0, "malignant": 1}
        self.label_map = {k.lower(): int(v) for k, v in label_map.items()}

        # ImageNet norm (keep consistent with pretrained ConvNeXt)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # samples: (video_folder, frame_files, label, video_id)
        self.samples: List[Tuple[str, List[str], int, str]] = []
        for class_name in ["benign", "malignant"]:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: missing class dir: {class_dir}")
                continue

            label = self.label_map[class_name.lower()]
            for vid in sorted(os.listdir(class_dir)):
                if self.video_ids is not None and vid not in self.video_ids:
                    continue

                vdir = os.path.join(class_dir, vid)
                if not os.path.isdir(vdir):
                    continue

                frame_files = sorted(
                    f for f in os.listdir(vdir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                )
                if len(frame_files) == 0:
                    continue

                self.samples.append((vdir, frame_files, label, vid))

        print(f"Loaded {len(self.samples)} videos for {'test' if test_mode else 'train'} mode")

    def __len__(self):
        return len(self.samples)

    def get_video_ids(self):
        return [vid for (_, _, _, vid) in self.samples]

    def get_labels(self):
        return [lab for (_, _, lab, _) in self.samples]

    # -------------------------
    # IO
    # -------------------------
    def _safe_open(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            if self.verbose_errors:
                print(f"Error loading {path}: {e}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    # -------------------------
    # Sampling
    # -------------------------
    def _sample_indices_train(self, total_frames: int) -> List[int]:
        """
        Paper-style segment sampling:
        Divide video into S segments; in each segment pick an anchor uniformly
        from valid anchors, then take F consecutive frames.
        """
        S = self.segment_count
        F = self.frames_per_segment
        T = S * F
        if total_frames <= 0:
            return [0] * T

        indices: List[int] = []
        for s in range(S):
            start = int(s * total_frames / S)
            end = int((s + 1) * total_frames / S)  # exclusive
            seg_len = max(1, end - start)

            max_anchor = max(0, seg_len - F)
            anchor = start + random.randint(0, max_anchor)

            for k in range(F):
                indices.append(min(anchor + k, total_frames - 1))

        return indices if len(indices) == T else (indices + [indices[-1]])[:T]

    def _sample_indices_test_center(self, total_frames: int) -> List[int]:
        """
        Deterministic center anchor per segment for reproducible validation/testing.
        """
        S = self.segment_count
        F = self.frames_per_segment
        T = S * F
        if total_frames <= 0:
            return [0] * T

        indices: List[int] = []
        for s in range(S):
            start = int(s * total_frames / S)
            end = int((s + 1) * total_frames / S)  # exclusive
            seg_len = max(1, end - start)

            max_anchor = max(0, seg_len - F)
            anchor = start + (max_anchor // 2)

            for k in range(F):
                indices.append(min(anchor + k, total_frames - 1))

        return indices if len(indices) == T else (indices + [indices[-1]])[:T]

    # -------------------------
    # Transform (temporal-consistent)
    # -------------------------
    def _transform_frames(self, pil_frames: List[Image.Image]) -> torch.Tensor:
        """
        pil_frames: length T = S*F
        Return: [S, F, C, H, W]
        """
        S = self.segment_count
        F = self.frames_per_segment
        T = S * F
        assert len(pil_frames) == T, f"Expected {T} frames, got {len(pil_frames)}"

        do_aug = self.augment and (not self.test_mode)

        # Decide ONCE per clip (apply identically across all frames)
        do_vflip = do_aug and self.enable_vflip and (random.random() < 0.5)
        if do_aug and self.enable_rotation and len(self.rotation_angles) > 0:
            angle = random.choice(self.rotation_angles)
        else:
            angle = 0

        frames_t = []
        for im in pil_frames:
            # Resize to model input size (no random crop, no scale jitter)
            im = TF.resize(im, (self.img_size, self.img_size))

            # Rotation (small discrete angles)
            if angle != 0:
                im = TF.rotate(
                    im,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0,
                )

            # Vertical flip
            if do_vflip:
                im = TF.vflip(im)

            t = TF.to_tensor(im)
            t = TF.normalize(t, self.mean, self.std)
            frames_t.append(t)

        clip = torch.stack(frames_t, dim=0)  # [T, C, H, W]
        clip = clip.view(S, F, 3, self.img_size, self.img_size)  # [S, F, C, H, W]
        return clip

    def _load_clip(self, folder: str, files: List[str], indices: List[int]) -> torch.Tensor:
        pil_frames = []
        n = len(files)
        for idx in indices:
            idx = max(0, min(int(idx), n - 1))
            pil_frames.append(self._safe_open(os.path.join(folder, files[idx])))
        return self._transform_frames(pil_frames)

    def __getitem__(self, idx: int):
        folder, files, label, _ = self.samples[idx]
        total_frames = len(files)

        if self.test_mode:
            indices = self._sample_indices_test_center(total_frames)
            clip = self._load_clip(folder, files, indices)
            return clip, torch.tensor(label, dtype=torch.long)

        indices = self._sample_indices_train(total_frames)
        clip = self._load_clip(folder, files, indices)
        return clip, torch.tensor(label, dtype=torch.long)


class BUVImageDataset(Dataset):
    """
    Static ultrasound image dataset for frame-branch pretraining only.
    Folder structure:
      root_dir/
        benign/*.png|jpg|jpeg
        malignant/*.png|jpg|jpeg
    Supports optional split_json to restrict images to specific videos.
    """
    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        train_mode: bool = True,
        eval_resize: int = 256,
        rotation_angles: Tuple[int, ...] = (-15, -10, -5, 5, 10, 15),
        label_map: Optional[dict] = None,
        split_json: Optional[str] = None,
        samples: Optional[List[Tuple[str, int]]] = None,
        verbose_errors: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = int(img_size)
        self.train_mode = bool(train_mode)
        self.eval_resize = int(eval_resize)
        self.rotation_angles = tuple(int(a) for a in rotation_angles)
        self.verbose_errors = bool(verbose_errors)

        if label_map is None:
            label_map = {"benign": 0, "malignant": 1}
        self.label_map = {k.lower(): int(v) for k, v in label_map.items()}

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # Resolve allowed (class_name, video_id) pairs from split_json
        _allowed_pairs: Optional[set] = None
        if split_json is not None and os.path.isfile(split_json):
            _allowed_pairs = self._parse_split_json(split_json)
        elif split_json is not None:
            raise FileNotFoundError(f"split_json not found: {split_json}")
        self._allowed_pairs = _allowed_pairs

        if samples is not None:
            self.samples = list(samples)
        else:
            self.samples: List[Tuple[str, int]] = []
            for class_name in ["benign", "malignant"]:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    print(f"Warning: missing class dir: {class_dir}")
                    continue

                label = self.label_map[class_name.lower()]
                # Recursively scan to support both flat and nested folder layouts.
                for cur_dir, _, files in os.walk(class_dir):
                    # Derive video_id: directory name relative to class_dir
                    rel = os.path.relpath(cur_dir, class_dir)
                    if rel == ".":
                        video_id = "__flat__"
                    else:
                        video_id = rel.split(os.sep)[0]

                    if self._allowed_pairs is not None:
                        if (class_name, video_id) not in self._allowed_pairs:
                            continue

                    for fn in sorted(files):
                        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            continue
                        self.samples.append((os.path.join(cur_dir, fn), label))

        mode_name = "train" if self.train_mode else "eval"
        print(f"Loaded {len(self.samples)} static images for {mode_name} mode")

    @staticmethod
    def _parse_split_json(json_path: str) -> set:
        """Parse {videos: [{name: "benign/vid_id"}, ...]} into a set of (class, video_id)."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "videos" not in data or not isinstance(data["videos"], list):
            raise ValueError(f"Invalid split JSON schema in {json_path}: missing 'videos' list")

        allowed = set()
        for vid in data["videos"]:
            full_name = str(vid.get("name", "")).strip()
            if "/" not in full_name:
                raise ValueError(f"Invalid video name '{full_name}' in {json_path}")
            class_name, video_id = full_name.split("/", 1)
            class_name = class_name.strip().lower()
            video_id = video_id.strip()
            allowed.add((class_name, video_id))
        return allowed

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [lab for _, lab in self.samples]

    def _safe_open(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            if self.verbose_errors:
                print(f"Error loading {path}: {e}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    def _transform(self, im: Image.Image) -> torch.Tensor:
        if self.train_mode:
            im = TF.resize(im, (self.img_size, self.img_size))
            if len(self.rotation_angles) > 0:
                angle = random.choice(self.rotation_angles)
                im = TF.rotate(
                    im,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0,
                )
            if random.random() < 0.5:
                im = TF.vflip(im)
        else:
            im = TF.resize(im, self.eval_resize, interpolation=TF.InterpolationMode.BILINEAR)
            im = TF.center_crop(im, [self.img_size, self.img_size])

        t = TF.to_tensor(im)
        t = TF.normalize(t, self.mean, self.std)
        return t

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        im = self._safe_open(img_path)
        img = self._transform(im)
        return img, torch.tensor(label, dtype=torch.long)
