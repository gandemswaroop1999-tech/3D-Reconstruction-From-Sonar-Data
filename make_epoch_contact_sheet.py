"""Create a two-column contact sheet from the rendered SUOP dummy epoch images."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = PROJECT_ROOT / "Evaluation_Rendering" / "comparison_outputs"
THUMBNAIL_WIDTH = 900
GAP = 16
HEADER = 66


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(COMPARISON_DIR / "dummy_suop_epochs"))
    parser.add_argument("--out", default=str(COMPARISON_DIR / "dummy_suop_epochs_contact_sheet.png"))
    parser.add_argument("--title", default="SUOP Dummy PCN Completion Progression — Epochs 1 to 10")
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.out)
    images = [Image.open(source / f"epoch_{epoch:02d}.png").convert("RGB") for epoch in range(1, 11)]
    thumbnail_height = round(images[0].height * THUMBNAIL_WIDTH / images[0].width)
    canvas = Image.new("RGB", (THUMBNAIL_WIDTH * 2 + GAP * 3, HEADER + thumbnail_height * 5 + GAP * 6), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    draw.text((GAP, 20), args.title, fill="white")
    for index, image in enumerate(images):
        image.thumbnail((THUMBNAIL_WIDTH, thumbnail_height))
        row, col = divmod(index, 2)
        x = GAP + col * (THUMBNAIL_WIDTH + GAP)
        y = HEADER + GAP + row * (thumbnail_height + GAP)
        canvas.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
