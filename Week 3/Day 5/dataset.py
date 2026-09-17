"""
Week 3 - Day 5 Weekly Project
Dataset Module: Synthetic Geometric & Pattern Dataset
Generates 5-class visual patterns (Circle, Square, Triangle, Star, Hexagon)
with realistic variation, noise, transformations, and PyTorch DataLoader integration.
"""

import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

CLASSES = ["Circle", "Square", "Triangle", "Star", "Hexagon"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}


def draw_star(img, center, size, color):
    cx, cy = center
    points = []
    for i in range(10):
        r = size if i % 2 == 0 else size // 2
        angle = i * (math.pi / 5) - math.pi / 2
        px = int(cx + r * math.cos(angle))
        py = int(cy + r * math.sin(angle))
        points.append([px, py])
    pts = np.array(points, np.int32)
    cv2.fillPoly(img, [pts], color)


def draw_hexagon(img, center, size, color):
    cx, cy = center
    points = []
    for i in range(6):
        angle = i * (math.pi / 3)
        px = int(cx + size * math.cos(angle))
        py = int(cy + size * math.sin(angle))
        points.append([px, py])
    pts = np.array(points, np.int32)
    cv2.fillPoly(img, [pts], color)


def generate_single_sample(class_name, img_size=48):
    """Draw a single shape with random parameters and noise."""
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cx = np.random.randint(img_size // 3, 2 * img_size // 3)
    cy = np.random.randint(img_size // 3, 2 * img_size // 3)
    size = np.random.randint(img_size // 6, img_size // 3)
    color = 255

    if class_name == "Circle":
        cv2.circle(img, (cx, cy), size, color, -1)
    elif class_name == "Square":
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), color, -1)
    elif class_name == "Triangle":
        pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif class_name == "Star":
        draw_star(img, (cx, cy), size, color)
    elif class_name == "Hexagon":
        draw_hexagon(img, (cx, cy), size, color)

    # Random rotation
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img = cv2.warpAffine(img, M, (img_size, img_size))

    # Add Gaussian noise
    noise = np.random.normal(0, 12, (img_size, img_size)).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Normalize to [0.0, 1.0] and shape (1, H, W)
    tensor_img = noisy_img.astype(np.float32) / 255.0
    return tensor_img[np.newaxis, :, :]


class VisionDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        label = self.labels[idx]

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def build_dataloaders(samples_per_class=300, img_size=48, batch_size=32, seed=42):
    """Generate train, val, and test splits with PyTorch DataLoaders."""
    np.random.seed(seed)
    images = []
    labels = []

    for class_name in CLASSES:
        label = CLASS_TO_IDX[class_name]
        for _ in range(samples_per_class):
            sample = generate_single_sample(class_name, img_size=img_size)
            images.append(sample)
            labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(labels))
    images, labels = images[indices], labels[indices]

    total = len(labels)
    n_train = int(0.70 * total)
    n_val = int(0.15 * total)

    train_imgs, train_lbls = images[:n_train], labels[:n_train]
    val_imgs, val_lbls = images[n_train:n_train + n_val], labels[n_train:n_train + n_val]
    test_imgs, test_lbls = images[n_train + n_val:], labels[n_train + n_val:]

    train_ds = VisionDataset(train_imgs, train_lbls, augment=True)
    val_ds = VisionDataset(val_imgs, val_lbls, augment=False)
    test_ds = VisionDataset(test_imgs, test_lbls, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, CLASSES
