"""
generate_au_prior.py
用途：在 RAF-DB aligned 图上用 MediaPipe 离线生成 AU 先验掩码
输出：每张图对应一个 14×14 的 float32 数组，统一保存为两个 .npy 字典文件
      au_prior_train.npy  → { "train_00001": array(14,14), ... }
      au_prior_test.npy   → { "test_0001":   array(14,14), ... }

运行方式：
    pip install mediapipe opencv-python-headless scipy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
    python generate_au_prior.py \
        --aligned_dir /Data/hjt/NLA/datasets/RAF-DB/basic/Image/aligned \
        --output_dir  /Data/hjt/NLA/datasets/RAF-DB/basic/Annotation/au_prior

首次运行会自动下载 face_landmarker.task 模型文件（约 8MB）到当前目录。
耗时估计：15339 张图约 10~20 分钟（CPU）
"""

import argparse
import os
import urllib.request
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned_dir', required=True,
                        help='RAF-DB aligned 图像目录')
    parser.add_argument('--output_dir',  required=True,
                        help='先验掩码输出目录')
    parser.add_argument('--model_path',  default='face_landmarker.task',
                        help='FaceLandmarker 模型文件路径（不存在时自动下载）')
    parser.add_argument('--patch_size',  default=16, type=int)
    parser.add_argument('--input_size',  default=224, type=int)
    parser.add_argument('--sigma',       default=8.0, type=float,
                        help='高斯扩散半径')
    parser.add_argument('--bg_weight',   default=0.3, type=float,
                        help='背景 patch 基础权重')
    return parser.parse_args()


# ── MediaPipe 468点 → AU 区域关键点索引 ──────────────────────────────────
AU_LANDMARK_INDICES = {
    'left_brow':   [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_brow':  [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'left_eye':    [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155],
    'right_eye':   [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382],
    'nose':        [1, 2, 98, 327, 168, 197, 195, 5, 4, 19, 94],
    'mouth':       [61, 291, 39, 269, 0, 17, 84, 314,
                    13, 312, 82, 311, 87, 317, 14, 178,
                    88, 95, 78, 191, 80, 81],
    'nasolabial':  [92, 322, 206, 426, 205, 425],
}


def download_model(model_path):
    """首次运行时自动下载 FaceLandmarker 模型（约 8MB）"""
    if os.path.exists(model_path):
        print(f"模型文件已存在: {model_path}")
        return
    url = ('https://storage.googleapis.com/mediapipe-models/'
           'face_landmarker/face_landmarker/float16/1/face_landmarker.task')
    print(f"正在下载 FaceLandmarker 模型（约8MB）: {url}")
    urllib.request.urlretrieve(url, model_path)
    print(f"下载完成: {model_path}")


def build_detector(model_path):
    """用新版 mediapipe Tasks API 初始化 FaceLandmarker"""
    import mediapipe as mp
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def build_au_mask(landmarks, img_size=224, patch_size=16,
                  sigma=8.0, bg_weight=0.3):
    """
    输入：mediapipe NormalizedLandmark 列表
    输出：[n_patches, n_patches] float32 先验掩码，值域 [bg_weight, 1.0]
    """
    n_patches = img_size // patch_size  # 14
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)

    for group_indices in AU_LANDMARK_INDICES.values():
        for idx in group_indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            px = int(np.clip(lm.x * img_size, 0, img_size - 1))
            py = int(np.clip(lm.y * img_size, 0, img_size - 1))
            heatmap[py, px] += 1.0

    # 高斯扩散
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # 下采样到 14×14
    heatmap_patches = heatmap.reshape(
        n_patches, patch_size,
        n_patches, patch_size
    ).mean(axis=(1, 3))  # [14, 14]

    # 归一化到 [bg_weight, 1.0]
    h_min, h_max = heatmap_patches.min(), heatmap_patches.max()
    if h_max - h_min > 1e-6:
        heatmap_patches = (heatmap_patches - h_min) / (h_max - h_min)
        heatmap_patches = bg_weight + (1.0 - bg_weight) * heatmap_patches
    else:
        heatmap_patches = np.ones((n_patches, n_patches), dtype=np.float32)

    return heatmap_patches.astype(np.float32)


def fallback_mask(n_patches=14):
    """检测失败时返回全1掩码，退化为原始行为"""
    return np.ones((n_patches, n_patches), dtype=np.float32)


def process_split(image_paths, detector, args):
    """处理一批图像，返回 { stem: mask_14x14 } 字典"""
    import mediapipe as mp
    n_patches = args.input_size // args.patch_size
    results = {}
    failed = []

    for img_path in tqdm(image_paths, desc=f"处理 {img_path.parent.name if False else 'images'}"):
        stem = img_path.stem.replace('_aligned', '')

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            failed.append(stem)
            results[stem] = fallback_mask(n_patches)
            continue

        img_rgb = cv2.cvtColor(
            cv2.resize(img_bgr, (args.input_size, args.input_size)),
            cv2.COLOR_BGR2RGB
        )

        # 新版 API：用 mp.Image 包装
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection = detector.detect(mp_image)

        if detection.face_landmarks:
            landmarks = detection.face_landmarks[0]
            mask = build_au_mask(
                landmarks,
                img_size=args.input_size,
                patch_size=args.patch_size,
                sigma=args.sigma,
                bg_weight=args.bg_weight
            )
        else:
            failed.append(stem)
            mask = fallback_mask(n_patches)

        results[stem] = mask

    return results, failed


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    aligned_dir = Path(args.aligned_dir)
    all_images = sorted(aligned_dir.glob('*_aligned.jpg'))
    if not all_images:
        all_images = sorted(aligned_dir.glob('*_aligned.png'))
    print(f"共找到 {len(all_images)} 张 aligned 图像")

    train_images = [p for p in all_images if p.stem.startswith('train')]
    test_images  = [p for p in all_images if p.stem.startswith('test')]
    print(f"  训练集: {len(train_images)}  测试集: {len(test_images)}")

    # ── 下载模型（首次）────────────────────────────────────────────────────
    download_model(args.model_path)

    # ── 初始化检测器 ───────────────────────────────────────────────────────
    detector = build_detector(args.model_path)
    print("FaceLandmarker 初始化完成")

    # ── 处理训练集 ─────────────────────────────────────────────────────────
    print("\n[1/2] 处理训练集...")
    train_masks, train_failed = process_split(train_images, detector, args)
    train_out = Path(args.output_dir) / 'au_prior_train.npy'
    np.save(str(train_out), train_masks)
    print(f"训练集完成: {len(train_masks)} 张，检测失败 {len(train_failed)} 张 "
          f"({100*len(train_failed)/max(len(train_masks),1):.1f}%)")
    if train_failed:
        print(f"  失败样本（前10）: {train_failed[:10]}")

    # ── 处理测试集 ─────────────────────────────────────────────────────────
    print("\n[2/2] 处理测试集...")
    test_masks, test_failed = process_split(test_images, detector, args)
    test_out = Path(args.output_dir) / 'au_prior_test.npy'
    np.save(str(test_out), test_masks)
    print(f"测试集完成: {len(test_masks)} 张，检测失败 {len(test_failed)} 张 "
          f"({100*len(test_failed)/max(len(test_masks),1):.1f}%)")
    if test_failed:
        print(f"  失败样本（前10）: {test_failed[:10]}")

    detector.close()

    # ── 可视化样例 ─────────────────────────────────────────────────────────
    print("\n正在生成可视化样例...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sample_keys = list(train_masks.keys())[:8]
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    for i, key in enumerate(sample_keys):
        img_path = aligned_dir / f"{key}_aligned.jpg"
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        axes[0, i].imshow(img)
        axes[0, i].set_title(key[-5:], fontsize=8)
        axes[0, i].axis('off')
        axes[1, i].imshow(train_masks[key], cmap='jet', vmin=0, vmax=1,
                          interpolation='bilinear')
        axes[1, i].axis('off')

    plt.suptitle('AU Prior Masks (top: image, bottom: 14×14 mask)', fontsize=12)
    plt.tight_layout()
    vis_path = Path(args.output_dir) / 'au_prior_preview.png'
    plt.savefig(str(vis_path), dpi=150, bbox_inches='tight')
    plt.close()

    # ── 统计摘要 ───────────────────────────────────────────────────────────
    all_arr = np.stack(list(train_masks.values()) + list(test_masks.values()))
    print(f"\n=== 掩码统计 ===")
    print(f"均值: {all_arr.mean():.4f}  最小: {all_arr.min():.4f}  最大: {all_arr.max():.4f}")
    print(f"\n输出文件:")
    print(f"  {train_out}")
    print(f"  {test_out}")
    print(f"  {vis_path}")


if __name__ == '__main__':
    main()