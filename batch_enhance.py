"""
Batch Underwater Image Enhancement Tool
=======================================
Process entire directories or video streams using SPM-CV.

Usage:
    python batch_enhance.py --input_dir path/to/images --output_dir path/to/results
"""

import os
import glob
import cv2
import time
import argparse
from tqdm import tqdm
from spm_underwater import enhance_underwater_image
from demo_and_benchmark import calculate_uciqe


def process_directory(input_dir: str, output_dir: str, create_comparisons: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tif", "*.tiff")
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"[!] No images found in '{input_dir}'.")
        return

    print(f"[+] Found {len(image_paths)} image(s) to process.")
    
    total_time = 0.0
    uciqe_improvements = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        start_t = time.time()
        enhanced, _ = enhance_underwater_image(img)
        elapsed = time.time() - start_t
        total_time += elapsed
        
        out_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        cv2.imwrite(out_path, enhanced)
        
        raw_uciqe = calculate_uciqe(img)
        enh_uciqe = calculate_uciqe(enhanced)
        uciqe_improvements.append((raw_uciqe, enh_uciqe))
        
        if create_comparisons:
            comp_h, comp_w = img.shape[:2]
            side_by_side = cv2.hconcat([img, enhanced])
            cv2.putText(side_by_side, "Raw Degraded", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(side_by_side, "SPM Enhanced", (comp_w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_comparison.png"), side_by_side)

    avg_time = (total_time / len(image_paths)) * 1000.0
    fps = len(image_paths) / (total_time + 1e-6)
    avg_raw_uciqe = sum(x[0] for x in uciqe_improvements) / len(uciqe_improvements)
    avg_enh_uciqe = sum(x[1] for x in uciqe_improvements) / len(uciqe_improvements)

    print("\n" + "="*55)
    print(f"🎉 Batch Processing Complete:")
    print(f"   - Total Images:      {len(image_paths)}")
    print(f"   - Avg Latency:       {avg_time:.2f} ms per image")
    print(f"   - Processing Speed:  {fps:.1f} FPS")
    print(f"   - Avg UCIQE Before:  {avg_raw_uciqe:.4f}")
    print(f"   - Avg UCIQE After:   {avg_enh_uciqe:.4f} (+{((avg_enh_uciqe - avg_raw_uciqe)/avg_raw_uciqe)*100:.1f}%)")
    print(f"   - Results saved in:  {output_dir}")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch SPM-CV Underwater Image Enhancement")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory containing underwater images")
    parser.add_argument("--output_dir", "-o", type=str, default="batch_results", help="Output directory for enhanced images")
    parser.add_argument("--no_compare", action="store_true", help="Skip generating side-by-side comparison images")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, not args.no_compare)
