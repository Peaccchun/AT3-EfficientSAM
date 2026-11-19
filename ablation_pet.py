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

out = Path("results_ablation")
out.mkdir(exist_ok=True)
records = []
for model_name, builder in models.items():
    print("\n开始:", model_name, flush=True)
    model = builder().to(device).eval()
    for count, idx in enumerate(indices, 1):
        image, trimap = dataset[idx]
        rgb = np.array(image.convert("RGB"))
        raw = np.array(trimap)
        gt, valid = raw == 1, raw != 3
        fg_dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
        bg_dist = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(fg_dist.argmax(), fg_dist.shape)
        ys, xs = np.where(gt & (fg_dist > 0))
        if len(xs) == 0: raise RuntimeError(f"图像{idx}没有有效前景")
        j = np.argmin(fg_dist[ys, xs]); ey, ex = int(ys[j]), int(xs[j])
        by, bx = np.unravel_index(bg_dist.argmax(), bg_dist.shape)
        settings = {
            "center": ([[int(cx), int(cy)]], [1]),
            "edge": ([[ex, ey]], [1]),
            "positive_negative": ([[int(cx), int(cy)], [int(bx), int(by)]], [1, 0]),
        }
        tensor = to_tensor(image).unsqueeze(0).to(device)
        for condition, (pts, labs) in settings.items():
            points = torch.tensor([[pts]], dtype=torch.float32, device=device)
            labels = torch.tensor([[labs]], dtype=torch.int64, device=device)
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
            records.append({"model": model_name, "condition": condition, "image_index": idx, "dice": float(dice), "iou": float(iou), "seconds": float(elapsed)})
            if count <= 3:
                overlay = rgb.copy()
                overlay[pred] = (0.5 * overlay[pred] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
                for (x, y), label in zip(pts, labs):
                    color = (255, 0, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(overlay, (x, y), 5, color, -1)
                filename = f"{model_name}_{idx}_{condition}.png"
                cv2.imwrite(str(out / filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"{model_name} {count:02d}/20 {condition} Dice={dice:.4f} IoU={iou:.4f}", flush=True)
    del model
with open(out / "ablation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader(); writer.writerows(records)
with open(out / "summary.txt", "w") as f:
    for name in models:
        for condition in ["center", "edge", "positive_negative"]:
            rows = [r for r in records if r["model"] == name and r["condition"] == condition]
            msg = (f"{name} | {condition} | Dice={np.mean([r['dice'] for r in rows]):.4f} "
                   f"IoU={np.mean([r['iou'] for r in rows]):.4f} "
                   f"Time={np.mean([r['seconds'] for r in rows]):.3f}s")
            print(msg); f.write(msg + "\n")
print("\n实验完成！结果保存在 results_ablation")
