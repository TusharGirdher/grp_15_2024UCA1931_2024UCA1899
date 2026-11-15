"""
SPM-CV Demo and Benchmark Tool
==============================
Demonstration script to test underwater image enhancement using OpenCV.
Features:
- Full step-by-step breakdown visualization (Raw, Transmission T, Gain W, Bias B, Physical, Cross-space, Final)
- Quantitative quality evaluation (UCIQE, Contrast, Sharpness)
- Synthetic underwater image generator for instant out-of-the-box testing.
"""

import os
import sys
import cv2
import numpy as np
import argparse
from typing import Tuple, Dict, Optional
from spm_underwater import enhance_underwater_image


def generate_synthetic_underwater_scene(width: int = 512, height: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic clear underwater scene and its physically degraded version
    to demonstrate and benchmark the algorithm.
    """
    # Create clear synthetic underwater scene (coral, fish shapes, sand floor)
    clear_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sand gradient (bottom)
    for y in range(int(height * 0.65), height):
        ratio = (y - height * 0.65) / (height * 0.35)
        # Sandy warm tone (BGR: Blue, Green, Red)
        b = int(140 + 30 * ratio)
        g = int(180 + 30 * ratio)
        r = int(210 + 35 * ratio)
        clear_img[y, :] = (b, g, r)
        
    # Open water backdrop (top to middle gradient)
    for y in range(0, int(height * 0.65)):
        ratio = y / (height * 0.65)
        clear_img[y, :] = (int(160 + 20 * ratio), int(120 + 20 * ratio), int(40 + 20 * ratio))
        
    # Draw colorful coral structures (rich red, orange, yellow, magenta)
    # Red coral branch
    cv2.circle(clear_img, (120, 360), 65, (40, 50, 220), -1)
    cv2.circle(clear_img, (160, 330), 50, (50, 60, 240), -1)
    cv2.circle(clear_img, (100, 300), 40, (30, 40, 200), -1)
    
    # Yellow-Orange coral
    cv2.circle(clear_img, (380, 370), 75, (20, 160, 230), -1)
    cv2.circle(clear_img, (420, 320), 55, (30, 180, 250), -1)
    
    # Purple/Cyan sea anemone
    cv2.ellipse(clear_img, (250, 400), (80, 50), 0, 0, 360, (180, 40, 160), -1)
    
    # Swimmer fish (Silver-white / Orange clownfish)
    cv2.ellipse(clear_img, (280, 180), (45, 20), 15, 0, 360, (20, 120, 255), -1)
    cv2.fillPoly(clear_img, [np.array([[240, 185], [215, 165], [215, 205]])], (20, 120, 255))
    cv2.circle(clear_img, (310, 175), 4, (0, 0, 0), -1)
    # White stripes
    cv2.line(clear_img, (275, 162), (275, 198), (250, 250, 250), 5)
    cv2.line(clear_img, (295, 167), (295, 193), (250, 250, 250), 4)

    # Physical underwater degradation simulation
    # Depth map: top is distant (high depth), bottom is close
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    depth_map = 0.3 + 0.7 * (1.0 - (y_coords / height))
    
    # Exponential attenuation: beta_R >> beta_G > beta_B
    beta_r, beta_g, beta_b = 2.4, 0.9, 0.6
    t_r = np.exp(-beta_r * depth_map)
    t_g = np.exp(-beta_g * depth_map)
    t_b = np.exp(-beta_b * depth_map)
    transmission = np.stack([t_b, t_g, t_r], axis=2)
    
    # Ambient water light (deep bluish-green)
    ambient_light = np.array([210, 160, 45], dtype=np.float32) / 255.0
    
    # I(x) = J(x)*T(x) + A*(1-T(x))
    clear_norm = clear_img.astype(np.float32) / 255.0
    degraded = clear_norm * transmission + ambient_light[np.newaxis, np.newaxis, :] * (1.0 - transmission)
    
    # Add scattering blur & light noise
    degraded = cv2.GaussianBlur(degraded, (5, 5), 1.2)
    degraded_uint8 = np.clip(degraded * 255.0, 0, 255).astype(np.uint8)
    
    return clear_img, degraded_uint8


def calculate_uciqe(image_bgr: np.ndarray) -> float:
    """
    Computes the Underwater Color Image Quality Evaluation (UCIQE) metric.
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    where:
    - sigma_c: standard deviation of chroma
    - con_l: contrast of luminance
    - mu_s: average saturation
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # Chroma standard deviation
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)
    sigma_c = np.std(chroma)
    
    # Luminance contrast (top 1% - bottom 1%)
    sorted_l = np.sort(l_ch.flatten())
    idx_top = int(0.99 * len(sorted_l))
    idx_bot = int(0.01 * len(sorted_l))
    con_l = sorted_l[idx_top] - sorted_l[idx_bot]
    
    # Saturation (in HSV space)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    mu_s = np.mean(sat)
    
    # Coefficients as per Yang & Sowmya (2015)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe = c1 * (sigma_c / 100.0) + c2 * (con_l / 255.0) + c3 * mu_s
    return float(uciqe)


def run_demo(input_path: Optional[str] = None, output_dir: str = "results"):
    """
    Runs the enhancement pipeline on an image or creates a synthetic test if none is provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if input_path and os.path.exists(input_path):
        print(f"[+] Loading input image from: {input_path}")
        raw_img = cv2.imread(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
    else:
        print("[+] No input image provided. Generating synthetic underwater test scene...")
        gt_img, raw_img = generate_synthetic_underwater_scene(512, 512)
        base_name = "synthetic_underwater"
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_ground_truth.png"), gt_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_raw_degraded.png"), raw_img)
        print(f"[✓] Saved synthetic ground truth and raw image in '{output_dir}/'")

    # Run SPM enhancement pipeline with intermediate map extraction
    print("[+] Running SPM Physical Inversion & Cross-Space Enhancement...")
    enhanced_img, intermediates = enhance_underwater_image(raw_img, return_intermediates=True)
    
    # Save enhanced output
    enhanced_out_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
    cv2.imwrite(enhanced_out_path, enhanced_img)
    print(f"[✓] Enhanced image saved to: {enhanced_out_path}")
    
    # Metrics
    raw_uciqe = calculate_uciqe(raw_img)
    enh_uciqe = calculate_uciqe(enhanced_img)
    print("\n" + "="*50)
    print(f"📊 Image Quality Assessment (UCIQE):")
    print(f"   - Raw Image UCIQE:      {raw_uciqe:.4f}")
    print(f"   - Enhanced Image UCIQE: {enh_uciqe:.4f}  (+{((enh_uciqe - raw_uciqe)/raw_uciqe)*100:.1f}%)")
    print(f"   - Estimated Ambient A:  [B: {intermediates['ambient_light'][0]:.2f}, G: {intermediates['ambient_light'][1]:.2f}, R: {intermediates['ambient_light'][2]:.2f}]")
    print("="*50 + "\n")
    
    # Create a 2x3 Diagnostic Dashboard
    h, w, _ = raw_img.shape
    dashboard_h, dashboard_w = 256, 256
    
    def prep(img, title):
        resized = cv2.resize(img, (dashboard_w, dashboard_h))
        if len(resized.shape) == 2 or resized.shape[2] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(resized, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return resized
        
    p_raw = prep(raw_img, "1. Raw Degraded")
    p_trans = prep(intermediates["transmission_map"], "2. Trans. Map T(x)")
    p_gain = prep(intermediates["gain_map_w"], "3. Gain Factor W(x)")
    p_phys = prep(intermediates["physical_stage"], "4. J = W*I + B")
    p_cross = prep(intermediates["cross_space_stage"], "5. RGB+LAB Balanced")
    p_final = prep(enhanced_img, "6. Final Detail Enriched")
    
    top_row = np.hstack([p_raw, p_trans, p_gain])
    bot_row = np.hstack([p_phys, p_cross, p_final])
    dashboard = np.vstack([top_row, bot_row])
    
    dashboard_path = os.path.join(output_dir, f"{base_name}_spm_diagnostic_dashboard.png")
    cv2.imwrite(dashboard_path, dashboard)
    print(f"[✓] Step-by-step SPM diagnostic dashboard saved to: {dashboard_path}")
    
    # Also save side-by-side comparison
    side_by_side = np.hstack([raw_img, enhanced_img])
    cv2.putText(side_by_side, "Original Raw", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(side_by_side, "SPM-CV Enhanced", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    cv2.imwrite(comparison_path, side_by_side)
    print(f"[✓] Side-by-side comparison saved to: {comparison_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPM-CV Underwater Image Enhancement Demo")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to degraded underwater image")
    parser.add_argument("--output_dir", "-o", type=str, default="results", help="Directory to save enhanced images")
    args = parser.parse_args()
    
    run_demo(args.input, args.output_dir)
