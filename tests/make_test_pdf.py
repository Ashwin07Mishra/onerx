"""Builds a small multi-page PDF fixture for tests."""
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io


def build_test_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    c.drawString(72, 720, "Company Overview")
    c.drawString(72, 700, "OneRx was founded in 2019 and is headquartered in Boston, Massachusetts.")
    c.drawString(72, 680, "The company builds software for pharmacy benefit management.")
    c.showPage()

    c.drawString(72, 720, "Product Details")
    c.drawString(72, 700, "The flagship product is called RxConnect, launched in 2021.")
    c.drawString(72, 680, "RxConnect processes prescription claims in real time.")
    c.showPage()

    c.drawString(72, 720, "Leadership")
    c.drawString(72, 700, "The CEO of OneRx is Jane Smith, who joined the company in 2020.")
    c.drawString(72, 680, "The CTO is Raj Patel, who previously worked at a major hospital system.")
    c.showPage()

    return _finish(c, buf)


def build_resume_test_pdf() -> bytes:
    """A small resume-like fixture used to test broader/semantic questions
    (e.g. 'what AI technologies are mentioned') that share little or no
    literal vocabulary with the answer text — the case that TF-IDF handled
    poorly and sentence embeddings should handle correctly."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    c.drawString(72, 720, "Page 1: Ashwin Kumar Mishra")
    c.drawString(72, 700, "Final-year Computer Science student focused on applied machine learning.")
    c.drawString(72, 680, "Contact: ashwin.mishra@example.com")
    c.showPage()

    c.drawString(72, 720, "Page 2: Experience")
    c.drawString(72, 700, "Research intern at a university lab, working on multilingual text summarization.")
    c.drawString(72, 680, "Built a document intelligence pipeline that extracts structured data from scanned forms.")
    c.drawString(72, 660, "Interned at an advertising technology company on generative content automation.")
    c.showPage()

    c.drawString(72, 720, "Page 3: Skills")
    c.drawString(72, 700, "Built retrieval-augmented generation systems using large language models and vector search.")
    c.drawString(72, 680, "Comfortable with Python, PyTorch, and transformer-based embedding models.")
    c.drawString(72, 660, "Deployed OCR pipelines feeding downstream language model based question answering.")
    c.showPage()

    return _finish(c, buf)


def _finish(c, buf: io.BytesIO) -> bytes:
    c.save()
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    with open("/home/claude/onerx/tests/fixture.pdf", "wb") as f:
        f.write(build_test_pdf())
