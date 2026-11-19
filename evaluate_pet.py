import csv
import random
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import to_tensor
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

# 依照聊天中用户在Mac运行的版本整理；此文件为复建副本。
torch.set_num_threads(4)
device = torch.device("cpu")
dataset = OxfordIIITPet(root="data", split="test", target_types="segmentation", download=False)
indices = sorted(random.Random(42).sample(range(len(dataset)), 20))
models = {"EfficientSAM-Ti": build_efficient_sam_vitt, "EfficientSAM-S": build_efficient_sam_vits}

OUT = Path("results_pet")
OUT.mkdir(exist_ok=True)
records = []
for model_name, builder in models.items():
    print(f"\n开始测试 {model_name}", flush=True)
    model = builder().to(device).eval()
    for count, idx in enumerate(indices, 1):
        image, trimap = dataset[idx]
        rgb = np.array(image.convert("RGB"))
        gt_raw = np.array(trimap)
        gt = gt_raw == 1
        valid = gt_raw != 3
        distance = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        points = torch.tensor([[[[int(x), int(y)]]]], dtype=torch.float32, device=device)
        labels = torch.tensor([[[1]]], dtype=torch.int64, device=device)
        tensor = to_tensor(image).unsqueeze(0).to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits, scores = model(tensor, points, labels)
        elapsed = time.perf_counter() - start
        best = scores[0, 0].argmax().item()
        pred = (logits[0, 0, best] >= 0).cpu().numpy()
        p, g = pred & valid, gt & valid
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        dice = 2 * intersection / max(p.sum() + g.sum(), 1)
        iou = intersection / max(union, 1)
        records.append({"model": model_name, "image_index": idx, "dice": float(dice), "iou": float(iou), "inference_seconds": elapsed})
        if count <= 3:
            overlay = rgb.copy()
            overlay[pred] = (0.5 * overlay[pred] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.circle(overlay, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.imwrite(str(OUT / f"{model_name}_{idx}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"{count:02d}/20 | Dice={dice:.4f} | IoU={iou:.4f} | {elapsed:.2f}s", flush=True)
    del model
with open(OUT / "evaluation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader(); writer.writerows(records)
with open(OUT / "summary.txt", "w") as f:
    for name in models:
        rows = [r for r in records if r["model"] == name]
        msg = (f"{name}\nMean Dice: {np.mean([r['dice'] for r in rows]):.4f}\n"
               f"Mean IoU: {np.mean([r['iou'] for r in rows]):.4f}\n"
               f"Mean time: {np.mean([r['inference_seconds'] for r in rows]):.3f}s\n")
        print("\n" + msg); f.write(msg + "\n")
print("实验完成，结果保存在 results_pet 文件夹")
