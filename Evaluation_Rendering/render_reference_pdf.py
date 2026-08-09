"""Render the supplied reference presentation PDF for visual inspection."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".pptdeps"))

import fitz
from PIL import Image, ImageDraw


PDF = Path(
    r"C:\Users\Neelash\.codex\attachments\f4c3140a-4566-4956-bdbf-6d765a60e160"
    r"\SUOP_3D_Reconstruction_Two_Routes.pdf"
)
OUT = ROOT.parent / "Documentation_Presentations" / "presentation_reference_pages"
OUT.mkdir(exist_ok=True)

doc = fitz.open(PDF)
thumbs = []
for number, page in enumerate(doc, start=1):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    path = OUT / f"slide_{number:02d}.png"
    pix.save(path)
    image = Image.open(path).convert("RGB")
    image.thumbnail((400, 225))
    tile = Image.new("RGB", (420, 255), "#0b1322")
    tile.paste(image, ((420 - image.width) // 2, 8))
    draw = ImageDraw.Draw(tile)
    draw.text((12, 232), f"{number:02d}", fill="white")
    thumbs.append(tile)

cols = 4
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 420, rows * 255), "#07111f")
for index, image in enumerate(thumbs):
    sheet.paste(image, ((index % cols) * 420, (index // cols) * 255))
sheet.save(OUT / "contact_sheet.png")

# Preserve the chair PCN result panel from slide 16 as a reusable project asset.
chair_slide = Image.open(OUT / "slide_16.png").convert("RGB")
chair_slide.crop((105, 145, 1335, 625)).save(OUT / "chair_pcn_epoch_1_to_15.png")

print(f"Rendered {len(thumbs)} slides to {OUT}")
