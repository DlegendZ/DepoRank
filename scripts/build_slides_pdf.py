"""Create the PDF viewing copy from full-size rendered slide previews."""

from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    images = sorted((ROOT / ".build/slides").glob("slide-*.png"))
    if len(images) != 16:
        raise ValueError("Expected all 16 rendered slides; run build_slides.mjs first.")
    output = ROOT / "presentation/slides.pdf"
    canvas = Canvas(str(output), pagesize=(960, 540))
    canvas.setTitle("DepoRank Stakeholder Presentation")
    for image in images:
        canvas.drawImage(str(image), 0, 0, width=960, height=540)
        canvas.showPage()
    canvas.save()
    print(output)
