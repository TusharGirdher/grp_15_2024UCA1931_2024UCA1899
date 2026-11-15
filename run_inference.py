"""
Inference Script for Underwater Image Enhancement
==================================================
Supports:
1. Physical Model Mode (SPM-CV, default):
   Executes the paper's Simplified Physical Model (SPM) inversion:
   J = W * I + B, with transmission estimation, cross-space CIELAB balance,
   and frequency detail recovery without requiring pre-trained deep learning weights.

2. Neural Network Mode (SPMFormer):
   Loads SPMFormer-B or SPMFormer-T from models/spmformer.py.
   Can load custom trained weights via --weights <path.pth>.
"""

import os
import argparse
import cv2
import numpy as np
import torch

from models.spmformer import spmformer_b, spmformer_t
from spm_underwater import enhance_underwater_image


def run_spm_enhancement(input_path: str, output_path: str):
    print(f"[+] Reading input image: {input_path}")
    raw_bgr = cv2.imread(input_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")
    
    print("[+] Applying Simplified Physical Model (SPM) Inversion & Cross-Space Enhancement...")
    enhanced_bgr, intermediates = enhance_underwater_image(raw_bgr, return_intermediates=True)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, enhanced_bgr)
    print(f"[✓] Enhanced image saved to: {output_path}")

    # Generate a side-by-side comparison
    comp_path = output_path.replace(".png", "_comparison.png").replace(".jpg", "_comparison.jpg")
    comparison = np.hstack([raw_bgr, enhanced_bgr])
    cv2.imwrite(comp_path, comparison)
    print(f"[✓] Comparison saved to: {comp_path}")
    return output_path, comp_path


def run_nn_spmformer(input_path: str, output_path: str, model_type: str = "spmformer_b", weights_path: str = None):
    print(f"[+] Loading SPMFormer neural network: {model_type}")
    if model_type == "spmformer_t":
        model = spmformer_t()
    else:
        model = spmformer_b()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[+] Using device: {device}")

    if weights_path and os.path.exists(weights_path):
        print(f"[+] Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip potential 'module.' prefix
        cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state)
    else:
        print("[!] Warning: No pre-trained checkpoint provided/found.")
        print("    Running with randomly initialized PyTorch weights.")

    model.to(device)
    model.eval()

    raw_bgr = cv2.imread(input_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    orig_h, orig_w = raw_bgr.shape[:2]
    # Resize to multiple of 256 or fixed patch
    target_h = ((orig_h + 31) // 32) * 32
    target_w = ((orig_w + 31) // 32) * 32

    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(raw_rgb, (target_w, target_h))

    # Scale [0, 255] to [-1, 1]
    tensor = torch.from_numpy(resized_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tensor = tensor.to(device)

    with torch.no_grad():
        out = model(tensor).clamp(-1, 1)

    out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.resize(out_bgr, (orig_w, orig_h))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, out_bgr)
    print(f"[✓] Neural network output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run underwater image enhancement")
    parser.add_argument("--input", default="input_sample.jpg", help="Path to input image")
    parser.add_argument("--output", default="results/enhanced_output.png", help="Path to save output")
    parser.add_argument("--mode", choices=["spm", "nn"], default="spm", help="Mode: 'spm' (physical model pipeline) or 'nn' (SPMFormer transformer)")
    parser.add_argument("--model", choices=["spmformer_b", "spmformer_t"], default="spmformer_b", help="Model variant for nn mode")
    parser.add_argument("--weights", default=None, help="Path to trained .pth weights file")

    args = parser.parse_args()

    if args.mode == "spm":
        run_spm_enhancement(args.input, args.output)
    else:
        run_nn_spmformer(args.input, args.output, model_type=args.model, weights_path=args.weights)
