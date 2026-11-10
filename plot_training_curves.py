#!/usr/bin/env python3
"""Generate training curves plot from training_history.json."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training history
history_path = Path(__file__).parent / "results" / "training_history.json"
with open(history_path) as f:
    history = json.load(f)

epochs = np.arange(1, len(history["loss"]) + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("PointNet++ Training Curves (Toronto-3D + Paris-Lille-3D)", fontsize=14, fontweight='bold')

# Training Loss
ax = axes[0, 0]
ax.plot(epochs, history["loss"], 'b-', linewidth=1.5, label='Train Loss')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)
ax.legend()

# Validation Loss
ax = axes[0, 1]
ax.plot(epochs, history["val_loss"], 'r-', linewidth=1.5, label='Val Loss')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Validation Loss")
ax.grid(True, alpha=0.3)
ax.legend()

# Validation mIoU
ax = axes[1, 0]
ax.plot(epochs, history["val_miou"], 'g-', linewidth=1.5, label='Val mIoU')
ax.set_xlabel("Epoch")
ax.set_ylabel("mIoU")
ax.set_title("Validation mIoU")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, max(history["val_miou"]) * 1.1)

# Validation Overall Accuracy
ax = axes[1, 1]
ax.plot(epochs, history["val_oa"], 'm-', linewidth=1.5, label='Val OA')
ax.set_xlabel("Epoch")
ax.set_ylabel("Overall Accuracy")
ax.set_title("Validation Overall Accuracy")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 1.0)

plt.tight_layout()

output_path = Path(__file__).parent / "assets" / "training_curves.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved training curves to {output_path}")