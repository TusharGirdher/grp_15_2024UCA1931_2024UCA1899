import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "results/tri_panel_comparisons"

img_names = [f"comparison_{i:02d}_diverse_scene.png" for i in range(1, 11)]

images = []
for name in img_names:
    p = os.path.join(OUTPUT_DIR, name)
    im = Image.open(p)
    images.append(im)

thumb_w = 880
col_gap = 24
row_gap = 20
margin_x = 30
margin_y = 90

grid_w = margin_x * 2 + thumb_w * 2 + col_gap
rows = 5
cols = 2

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

draw.text((margin_x, 24), "Comprehensive Visual Evaluation: 10 Diverse Underwater Scenes (Raw vs Base vs PhysSPM-Former)",
          fill=(248, 250, 252), font=font_title)
draw.text((margin_x, 58), "Diverse Test Corpus: Corals, Divers, Murky Algae, Deep Abyss, Sandy Seabeds | Group 15: Tushar Girdher & Aakash Kumar",
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
print(f"Saved updated master montage to {montage_out} ({montage.size})")
