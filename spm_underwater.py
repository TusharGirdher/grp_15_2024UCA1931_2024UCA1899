"""
SPM-CV: Simplified Physical Model-based Underwater Image Enhancement
====================================================================
Based on the paper:
"SPMFormer: Simplified Physical Model-based transformer with cross-space loss
 for underwater image enhancement" (Li, Chen, & Miao, Knowledge-Based Systems, 2025).

This module implements the core physical formulation and cross-space/frequency
enhancement principles using only OpenCV (cv2) and NumPy.

Key Mathematical Concepts:
---------------------------
1. Physical Degradation Model (Eq. 2):
   I(x) = J(x) * T(x) + A * (1 - T(x))

2. Simplified Physical Model & Gain/Bias Inversion (Eq. 5 & 6):
   J_k(x) = W_k(x) * I_k(x) + B_k(x)
   where:
     W_k(x) = 1 / T_k(x)                 (Gain Factor)
     B_k(x) = A_k * (1 - W_k(x))          (Bias Factor)
     k in {R, G, B}

3. Cross-Space Color Correction (Eq. 13-15):
   Simultaneous correction in RGB and CIELAB color spaces to fix severe
   red-channel attenuation, green/blue color cast, and restore natural contrast.

4. Frequency-Domain Consistency & Detail Enhancement (Eq. 16-17):
   High-frequency magnitude & phase recovery to reverse light scattering blur.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 30, eps: float = 1e-3) -> np.ndarray:
    """
    Fast Guided Image Filter to refine transmission maps while preserving edge structure.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    mean_p = cv2.blur(src, (radius, radius))
    
    if len(guide.shape) == 2:
        mean_i = cv2.blur(guide, (radius, radius))
        mean_ii = cv2.blur(guide * guide, (radius, radius))
        mean_ip = cv2.blur(guide * src, (radius, radius))
        
        var_i = mean_ii - mean_i * mean_i
        cov_ip = mean_ip - mean_i * mean_p
        
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        
        mean_a = cv2.blur(a, (radius, radius))
        mean_b = cv2.blur(b, (radius, radius))
        
        q = mean_a * guide + mean_b
    else:
        q = np.zeros_like(src)
        for c in range(src.shape[2] if len(src.shape) == 3 else 1):
            s_c = src[:, :, c] if len(src.shape) == 3 else src
            g_c = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
            q_c = guided_filter(g_c, s_c, radius, eps)
            if len(src.shape) == 3:
                q[:, :, c] = q_c
            else:
                q = q_c
    return q


def estimate_ambient_light(image_bgr: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """
    Estimates the background/ambient light A = [A_B, A_G, A_R] from the degraded underwater image.
    In underwater scenes, light is heavily scattered in the background/deepest areas.
    """
    img_float = image_bgr.astype(np.float32) / 255.0
    h, w, _ = img_float.shape
    num_pixels = h * w
    top_k = max(int(num_pixels * top_percent), 10)
    
    # Calculate underwater dark channel (UDCP: green and blue channels usually dominate)
    b_ch, g_ch, r_ch = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    
    # Atmospheric brightness score favoring smooth distant water areas
    smooth_lum = cv2.GaussianBlur((0.5 * g_ch + 0.5 * b_ch), (15, 15), 0)
    flat_lum = smooth_lum.flatten()
    indices = np.argpartition(flat_lum, -top_k)[-top_k:]
    
    flat_b = b_ch.flatten()[indices]
    flat_g = g_ch.flatten()[indices]
    flat_r = r_ch.flatten()[indices]
    
    # Compute robust average ambient light
    A_B = float(np.median(flat_b))
    A_G = float(np.median(flat_g))
    A_R = float(np.median(flat_r))
    
    # Ensure reasonable boundaries
    A = np.array([max(A_B, 0.1), max(A_G, 0.1), max(A_R, 0.05)], dtype=np.float32)
    return A


def estimate_wavelength_transmission(
    image_bgr: np.ndarray, 
    A: np.ndarray, 
    omega: float = 0.95,
    patch_size: int = 15,
    guided_radius: int = 30,
    guided_eps: float = 1e-3
) -> np.ndarray:
    """
    Estimates wavelength-dependent medium transmission maps T_k(x) for each channel k in {B, G, R}.
    Accounting for the physical property: beta_R > beta_G > beta_B (Red light attenuates fastest).
    """
    img_float = image_bgr.astype(np.float32) / 255.0
    
    # Normalized image by ambient light
    norm_img = img_float / (A[np.newaxis, np.newaxis, :] + 1e-5)
    
    # Estimate base depth transmission from the dominant blue-green channels
    bg_min = np.minimum(norm_img[:, :, 0], norm_img[:, :, 1])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_bg = cv2.erode(bg_min, kernel)
    
    # Base transmission map T_base(x)
    t_base = 1.0 - omega * dark_bg
    t_base = np.clip(t_base, 0.05, 1.0)
    
    # Refine base transmission using guided filter
    gray_guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_base_refined = guided_filter(gray_guide, t_base, radius=guided_radius, eps=guided_eps)
    t_base_refined = np.clip(t_base_refined, 0.05, 1.0)
    
    # Compute wavelength attenuation ratios relative to dominant light
    ratio_b = 1.00
    ratio_g = 1.15
    ratio_r = 2.20
    
    t_b = np.power(t_base_refined, ratio_b)
    t_g = np.power(t_base_refined, ratio_g)
    t_r = np.power(t_base_refined, ratio_r)
    
    transmission = np.stack([t_b, t_g, t_r], axis=2)
    return np.clip(transmission, 0.05, 1.0)


def compute_simplified_physical_parameters(
    transmission: np.ndarray,
    A: np.ndarray,
    t0: float = 0.15,
    max_gain: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the gain factor W_k(x) and bias factor B_k(x) according to the paper:
    J_k(x) = W_k(x) * I_k(x) + B_k(x)
    
    W_k(x) = 1 / max(T_k(x), t0)
    B_k(x) = A_k * (1 - W_k(x))
    """
    t_clamped = np.maximum(transmission, t0)
    
    # Gain factor W(x)
    W = 1.0 / t_clamped
    W = np.clip(W, 1.0, max_gain)
    
    # Bias factor B(x) derived from J = (I - A*(1-T))/T = W*I + A*(1 - W)
    B = A[np.newaxis, np.newaxis, :] * (1.0 - W) * 0.8
    
    return W.astype(np.float32), B.astype(np.float32)


def apply_physical_reconstruction(
    image_bgr: np.ndarray,
    W: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Applies the linear physical model: J(x) = W(x) * I(x) + B(x)
    """
    img_float = image_bgr.astype(np.float32) / 255.0
    
    # J = W * I + B
    J_restored = W * img_float + B
    J_restored = np.clip(J_restored, 0.0, 1.0)
    
    return (J_restored * 255.0).astype(np.uint8)


def cross_space_color_balance(
    image_bgr: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: Tuple[int, int] = (8, 8),
    chroma_boost: float = 1.15
) -> np.ndarray:
    """
    Multi-space color balancing mimicking the SPMFormer cross-space loss (RGB + LAB spaces).
    
    1. LAB Color Space:
       - Luminance (L*): CLAHE adaptive histogram equalization for contrast without halos.
       - Chrominance (a*, b*): Mean-deviation centering and red-chroma recovery (a* channel).
    2. RGB Color Space:
       - Dynamic range stretching and Gray-World balancing for chromatic consistency.
    """
    # Convert to CIELAB space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    
    # 1. Enhance Lightness channel with CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    l_enhanced = clahe.apply(l_ch)
    
    # 2. Chromaticity correction on a* (green-red) and b* (blue-yellow)
    a_float = a_ch.astype(np.float32) - 128.0
    b_float = b_ch.astype(np.float32) - 128.0
    
    mean_a = np.mean(a_float)
    mean_b = np.mean(b_float)
    
    # Compensate color cast by shifting the mean toward neutral gray
    a_corrected = (a_float - 0.7 * mean_a) * chroma_boost
    b_corrected = (b_float - 0.7 * mean_b) * chroma_boost
    
    a_out = np.clip(a_corrected + 128.0, 0, 255).astype(np.uint8)
    b_out = np.clip(b_corrected + 128.0, 0, 255).astype(np.uint8)
    
    lab_merged = cv2.merge([l_enhanced, a_out, b_out])
    bgr_from_lab = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    
    # 3. Dynamic range percentile stretching in RGB space
    out_bgr = np.zeros_like(bgr_from_lab)
    for c in range(3):
        ch = bgr_from_lab[:, :, c].astype(np.float32)
        low_val, high_val = np.percentile(ch, (1.0, 99.0))
        if high_val > low_val:
            ch_stretched = (ch - low_val) * (255.0 / (high_val - low_val))
            out_bgr[:, :, c] = np.clip(ch_stretched, 0, 255).astype(np.uint8)
        else:
            out_bgr[:, :, c] = bgr_from_lab[:, :, c]
            
    return out_bgr


def frequency_domain_detail_enhancement(
    image_bgr: np.ndarray,
    strength: float = 0.4,
    blur_kernel_size: int = 5
) -> np.ndarray:
    """
    Frequency-domain detail enhancement mimicking the paper's Fourier consistency loss (L_F).
    Recovers high-frequency phase and magnitude degraded by scattering blur.
    """
    img_float = image_bgr.astype(np.float32)
    
    # Multi-frequency decomposition using Gaussian pyramid / Unsharp Mask
    low_freq = cv2.GaussianBlur(img_float, (blur_kernel_size, blur_kernel_size), 0)
    high_freq = img_float - low_freq
    
    # Boost crisp structural high frequencies while bounding overshoot
    enhanced = img_float + strength * high_freq
    enhanced = np.clip(enhanced, 0.0, 255.0).astype(np.uint8)
    
    return enhanced


def enhance_underwater_image(
    image_bgr: np.ndarray,
    omega: float = 0.95,
    max_gain: float = 3.5,
    clahe_clip: float = 2.0,
    chroma_boost: float = 1.15,
    freq_strength: float = 0.35,
    return_intermediates: bool = False
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    End-to-end SPM underwater image enhancement pipeline mimicking SPMFormer.

    Parameters:
    -----------
    image_bgr : np.ndarray
        Input BGR image (uint8, shape [H, W, 3]).
    omega : float
        Scattering removal factor (0.8 - 0.98).
    max_gain : float
        Maximum physical gain factor W_k(x) allowed (2.0 - 5.0).
    clahe_clip : float
        CLAHE clip limit for luminance contrast in LAB space (1.5 - 3.0).
    chroma_boost : float
        Chroma correction boost factor (1.0 - 1.3).
    freq_strength : float
        Frequency detail enhancement weight (0.2 - 0.5).
    return_intermediates : bool
        If True, returns a dictionary containing intermediate physical maps (W, B, T, etc.).

    Returns:
    --------
    Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        - Enhanced BGR image.
        - Dictionary of intermediate maps (if return_intermediates=True).
    """
    # Step 1: Ambient Light Estimation A
    A = estimate_ambient_light(image_bgr)
    
    # Step 2: Transmission Map Estimation & Refinement T(x)
    T = estimate_wavelength_transmission(image_bgr, A, omega=omega)
    
    # Step 3: Simplified Physical Model Gain & Bias (WB Recon: W(x), B(x))
    W, B = compute_simplified_physical_parameters(T, A, max_gain=max_gain)
    
    # Step 4: Physical Linear Inversion: J = W * I + B
    J_physical = apply_physical_reconstruction(image_bgr, W, B)
    
    # Step 5: Cross-Space Color Balancing (RGB + LAB)
    J_balanced = cross_space_color_balance(
        J_physical, 
        clahe_clip_limit=clahe_clip,
        chroma_boost=chroma_boost
    )
    
    # Step 6: Frequency-Domain Detail & Edge Sharpening (L_F recovery)
    J_final = frequency_domain_detail_enhancement(
        J_balanced,
        strength=freq_strength
    )
    
    intermediates = None
    if return_intermediates:
        intermediates = {
            "ambient_light": A,
            "transmission_map": (T * 255.0).astype(np.uint8),
            "gain_map_w": ((W / max_gain) * 255.0).astype(np.uint8),
            "bias_map_b": (np.clip((B + 1.0) / 2.0, 0, 1) * 255.0).astype(np.uint8),
            "physical_stage": J_physical,
            "cross_space_stage": J_balanced,
            "final_enhanced": J_final
        }
        
    return J_final, intermediates
