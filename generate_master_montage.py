import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "results/tri_panel_comparisons"

# Select 10 images to include in the montage (comparison_01 through comparison_10)
img_names = [
    "comparison_01_coral_reef_sample.png",
    "comparison_02_lsui_0001.png",
    "comparison_03_lsui_0002.png",
    "comparison_04_lsui_0003.png",
    "comparison_05_lsui_0004.png",
    "comparison_06_lsui_0005.png",
    "comparison_07_lsui_0006.png",
    "comparison_08_lsui_0007.png",
    "comparison_09_lsui_0008.png",
    "comparison_10_lsui_0009.png"
]

images = []
for name in img_names:
    p = os.path.join(OUTPUT_DIR, name)
    im = Image.open(p)
    images.append(im)

# Let's create a 5x2 grid (5 rows, 2 columns) of cards resized to 900px wide
thumb_w = 880
col_gap = 24
row_gap = 20
margin_x = 30
margin_y = 90

grid_w = margin_x * 2 + thumb_w * 2 + col_gap
# Calculate row heights
rows = 5
cols = 2

# Scale images keeping aspect ratio
scaled_images = []
for im in images:
    ratio = thumb_w / im.width
    th = int(im.height * ratio)
    scaled_images.append(im.resize((thumb_w, th), Image.LANCZOS))

row_heights = []
for r in range(rows):
    idx1 = r * 2
    idx2 = r * 2 + 1
    h1 = scaled_images[idx1].height
    h2 = scaled_images[idx2].height
    row_heights.append(max(h1, h2))

grid_h = margin_y + sum(row_heights) + row_gap * (rows - 1) + 40

montage = Image.new('RGB', (grid_w, grid_h), (9, 13, 22))
draw = ImageDraw.Draw(montage)

font_path = '/System/Library/Fonts/Helvetica.ttc'
font_title = ImageFont.truetype(font_path, 26)
font_sub = ImageFont.truetype(font_path, 15)

draw.text((margin_x, 24), "Comprehensive Visual Evaluation Gallery: PhysSPM-Former vs Base Paper & Raw Degraded",
          fill=(248, 250, 252), font=font_title)
draw.text((margin_x, 58), "10 Extensive Benchmark & In-The-Wild Test Cases  |  Group 15: Tushar Girdher (2024UCA1931) & Aakash Kumar (2024UCA1899)",
          fill=(148, 163, 184), font=font_sub)

curr_y = margin_y
for r in range(rows):
    for c in range(cols):
        idx = r * 2 + c
        im = scaled_images[idx]
        x = margin_x + c * (thumb_w + col_gap)
        y = curr_y
        montage.paste(im, (x, y))
    curr_y += row_heights[r] + row_gap

montage_out = os.path.join(OUTPUT_DIR, "master_comparisons_montage.png")
montage.save(montage_out)
print(f"Saved master montage to {montage_out} ({montage.size})")
