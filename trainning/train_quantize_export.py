#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a tiny MLP (784->32->10), quantize to int8, and export FPGA-friendly files.

Creates training/artifacts/ with:
  W1.mem  b1.mem  W2.mem  b2.mem        (hex, row-major)
  shift1.txt  shift2.txt                 (int right-shifts for requantization)
  scale_x.txt scale_w1.txt scale_w2.txt  (float info; optional for records)
  sample_input.mem  sample_label.txt     (one test image in int8, hex-per-byte)
  golden_pred_int32.txt                  (predicted digit via quantized math)

Requantization math (hardware-friendly):
  Layer1: int8 x int8 -> int32 acc -> +bias_int32 -> >>> shift1 -> clamp int8 -> ReLU
  Layer2: int8 x int8 -> int32 acc -> +bias_int32 -> (keep int32) -> argmax
"""

import os, math, pathlib, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

ROOT = pathlib.Path(__file__).resolve().parent
ART  = ROOT / "mnist_fpga/hdl/mem_init"
ART.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Data & model definition
# -----------------------------
BATCH = 128
EPOCHS = 3          # small, fast
HIDDEN = 32
INPUT  = 28*28      # 784
OUTPUT = 10
LR = 1e-3

transform = transforms.Compose([transforms.ToTensor()])  # [0,1] float
train_ds = datasets.MNIST(root=str(ROOT/"data"), train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root=str(ROOT/"data"), train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT, HIDDEN, bias=True)
        self.fc2 = nn.Linear(HIDDEN, OUTPUT, bias=True)
        self.act = nn.ReLU()
    def forward(self, x):
        x = x.view(x.size(0), -1)        # (B,784)
        h = self.act(self.fc1(x))
        y = self.fc2(h)
        return y

device = "cpu"
model = MLP().to(device)
opt = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -----------------------------
# 2) Quick training
# -----------------------------
model.train()
for epoch in range(EPOCHS):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
print("[Info] Training done.")

# -----------------------------
# 3) Helper: symmetric int8 scales
# -----------------------------
@torch.no_grad()
def calc_scale_sym(t: torch.Tensor, int8_max=127):
    m = t.abs().max().item()
    return (m / int8_max) if m > 0 else 1.0

def clamp_int8(a: torch.Tensor):
    return torch.clamp(a, -128, 127).to(torch.int8)

def save_mem_int8(path: pathlib.Path, arr2d: np.ndarray):
    with open(path, "w") as f:
        for row in arr2d:
            f.write(" ".join(f"{(int(v)&0xFF):02x}" for v in row.tolist()) + "\n")

def save_mem_int32(path: pathlib.Path, arr1d: np.ndarray):
    with open(path, "w") as f:
        for v in arr1d.tolist():
            f.write(f"{(int(v)&0xFFFFFFFF):08x}\n")

# -----------------------------
# 4) Calibration (collect ranges)
# -----------------------------
model.eval()
with torch.no_grad():
    # sample a small calibration batch
    calib_x = torch.stack([train_ds[i][0] for i in range(256)]).to(device)  # (256,1,28,28)
    x_flat  = calib_x.view(256, -1)                                         # (256,784)

    # pass once to get typical ranges
    h1_f = model.act(model.fc1(x_flat))     # float hidden (pre-ReLU already applied)
    # scales (symmetric)
    scale_x  = calc_scale_sym(x_flat)       # float -> int8 for inputs
    scale_w1 = calc_scale_sym(model.fc1.weight.data)
    scale_w2 = calc_scale_sym(model.fc2.weight.data)
    # choose an activation target scale using observed hidden range
    scale_h1_target = calc_scale_sym(h1_f)  # desired post-ReLU dynamic range

# -----------------------------
# 5) Quantize weights & biases
# -----------------------------
@torch.no_grad()
def quantize_linear(linear: nn.Linear, in_scale: float, w_scale: float):
    Wf = linear.weight.data.clone()
    bf = linear.bias.data.clone()
    # symmetric int8 weights
    Wq = clamp_int8((Wf / w_scale).round())
    # bias is stored in int32 accumulator domain: bias_int = round(b / (in_scale * w_scale))
    bias_scale = in_scale * w_scale
    bq = torch.round(bf / bias_scale).to(torch.int32)
    return Wq, bq

W1_q, b1_q = quantize_linear(model.fc1, in_scale=scale_x,  w_scale=scale_w1)
# For layer2, the "effective" input scale AFTER L1 is (scale_x * scale_w1) / (2^shift1)
# We don't know shift1 yet; compute it next, then quantize L2 bias with the correct in_scale.

# -----------------------------
# 6) Compute right-shift(s) for hardware
# -----------------------------
# L1 accumulator LSB scale before shift = scale_x * scale_w1
acc1_scale = scale_x * scale_w1
# We want output int8 with target ~ scale_h1_target.
# Requantization multiplier M1 = acc1_scale / scale_h1_target
M1 = acc1_scale / (scale_h1_target if scale_h1_target > 0 else 1.0)
# Hardware does arithmetic right shift by shift1 ~ log2(M1). Use nearest integer >= 0.
shift1 = max(0, int(round(math.log2(M1)))) if M1 > 0 else 0
# Effective hidden scale that hardware will realize:
scale_h1_eff = acc1_scale / (2**shift1)

# Now L2's input scale is that effective hidden scale:
in2_scale = scale_h1_eff
W2_q, b2_q = quantize_linear(model.fc2, in_scale=in2_scale, w_scale=scale_w2)

# For L2, we keep logits as int32 and do argmax (no need to shift). If you want, you can also compute a shift2:
# Just for completeness (unused in hardware if you keep int32):
# Estimate typical logit magnitude from a small forward and set a conservative shift2 to avoid overflow if needed.
with torch.no_grad():
    # rough pass to estimate typical acc2 magnitude
    h1_q_sim = torch.clamp((h1_f / scale_h1_eff).round(), -128, 127).to(torch.int8)
    # int32 simulate L2 acc scale = in2_scale * scale_w2
    acc2_scale = in2_scale * scale_w2
    # we won't requantize to int8 in hardware, so set shift2 = 0 for clarity
    shift2 = 0

# -----------------------------
# 7) Save params & metadata
# -----------------------------
(ART/"scale_x.txt").write_text(f"{scale_x}\n")
(ART/"scale_w1.txt").write_text(f"{scale_w1}\n")
(ART/"scale_w2.txt").write_text(f"{scale_w2}\n")
(ART/"shift1.txt").write_text(f"{shift1}\n")
(ART/"shift2.txt").write_text(f"{shift2}\n")

# Row-major save: each row = one output neuron (i.e., weights[j][:])
save_mem_int8(ART/"W1.mem", W1_q.numpy())      # shape (32,784)
save_mem_int32(ART/"b1.mem", b1_q.numpy())     # shape (32,)
save_mem_int8(ART/"W2.mem", W2_q.numpy())      # shape (10,32)
save_mem_int32(ART/"b2.mem", b2_q.numpy())     # shape (10,)
print("[Info] Exported W1/W2/b1/b2 and shift/scales.")

# -----------------------------
# 8) Export one sample input (int8) + golden check
# -----------------------------
# Find first sample with digit 3
test_idx = 0
for i in range(len(test_ds)):
    if test_ds[i][1] == 3:
        test_idx = i
        break
test_img, test_label = test_ds[test_idx]   # Use first sample with digit 3
x0 = test_img.view(-1)                      # (784,)
x0_q = torch.clamp((x0 / scale_x).round(), -128, 127).to(torch.int8)

# Save sample input as hex bytes, 1 per line (easy for $readmemh)
with open(ART/"sample_input.mem", "w") as f:
    for v in x0_q.numpy().tolist():
        f.write(f"{(int(v)&0xFF):02x}\n")
(ART/"sample_label.txt").write_text(f"{int(test_label)}\n")

# -----------------------------
# 9) Quantized forward in Python (mirrors hardware math)
# -----------------------------
def int8_mm_int32_acc(Wq: torch.Tensor, xq: torch.Tensor):
    """(out_dim, in_dim) int8 W times (in_dim,) int8 x -> (out_dim,) int32 acc"""
    # Use int32 accumulation explicitly
    return torch.matmul(Wq.to(torch.int32), xq.to(torch.int32))

@torch.no_grad()
def forward_quantized(xq_int8):
    # L1: acc1 = W1*x + b1  (int32), then >>> shift1, clamp to int8, ReLU
    acc1 = int8_mm_int32_acc(W1_q, xq_int8) + b1_q
    h1_q = torch.clamp((acc1 >> shift1), -128, 127).to(torch.int8)
    h1_q = torch.maximum(h1_q, torch.tensor(0, dtype=torch.int8))  # ReLU

    # L2: acc2 = W2*h1_q + b2  (int32). Keep int32 logits; argmax.
    acc2 = int8_mm_int32_acc(W2_q, h1_q) + b2_q
    pred = int(torch.argmax(acc2).item())
    return pred, acc1, h1_q, acc2

# Run the quantized forward pass and save golden result
pred_q, acc1_q, h1_q, acc2_q = forward_quantized(x0_q)
(ART/"golden_pred_int32.txt").write_text(f"{pred_q}\n")

print(f"[Info] Using test sample index {test_idx}")
print(f"[Info] Sample input quantized forward: predicted digit = {pred_q}")
print(f"[Info] True label = {test_label}")
print(f"[Info] Exported sample + golden prediction.")
