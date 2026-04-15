import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
SAVE_DIR = Path("/content/gdrive/MyDrive/Train_Yolov8/Image/Featuremap/KQ_3_3/base")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CH = 16
EPS = 1e-6


# =========================
# SAVE 16 CHANNEL
# =========================
def save_16ch(x, name):
    """
    x: (C,H,W)
    """

    c = x.shape[0]
    n = min(MAX_CH, c)

    fig, ax = plt.subplots(2, 8, figsize=(16, 4))
    ax = np.array(ax).ravel()

    for i in range(n):
        fm = x[i]

        # normalize robust
        fm = fm - fm.mean()
        fm = fm / (fm.std() + EPS)
        fm = np.clip(fm, -2, 2)
        fm = (fm + 2) / 4

        ax[i].imshow(fm, cmap="gray")
        ax[i].set_title(f"ch{i}", fontsize=8)
        ax[i].axis("off")

    for i in range(n, 16):
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{name}.png", dpi=300)
    plt.close()

    print(f"[INFO] Saved {name}")


# =========================
# HOOK ATTENTION (CHÍNH XÁC)
# =========================
def make_hook(i):

    def hook_fn(module, input, output):
        """
        input[0]  = x[i] (before attention)
        output    = self.atte[i](x[i]) (after attention)
        """

        before = input[0][0].detach().cpu().numpy()
        after = output[0].detach().cpu().numpy()

        scale = f"P{3+i}"

        # save BEFORE
        save_16ch(before, f"{scale}_before_atte")

        # save AFTER
        save_16ch(after, f"{scale}_after_atte")

        # save numpy
        np.save(SAVE_DIR / f"{scale}_before.npy", before)
        np.save(SAVE_DIR / f"{scale}_after.npy", after)

        print(f"[INFO] Done {scale}")

    return hook_fn


# =========================
# LOAD MODEL
# =========================
model = YOLO('/content/gdrive/MyDrive/Train_Yolov8/Ketqua_ppe/best_ORG.pt')

# không cần train() vì hook ở giữa pipeline
# nhưng vẫn OK nếu bật
model.model.eval()

detect_layer = model.model.model[-1]

# =========================
# REGISTER HOOK (ATTENTION)
# =========================
for i in range(detect_layer.nl):
    detect_layer.atte[i].register_forward_hook(make_hook(i))

print("[INFO] Hook registered (attention)")

# =========================
# RUN
# =========================
model("/content/gdrive/MyDrive/Train_Yolov8/Image/Featuremap/Construction_3.jpg")

print("[INFO] Done!")
