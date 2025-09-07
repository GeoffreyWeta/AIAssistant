import os
import uuid
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
from app.Config.config import settings

def save_pdf_report(answer: str, citations: list[dict], question: str, web_sources: list[dict] | None = None) -> dict:
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    report_id = str(uuid.uuid4())
    path = os.path.join(settings.REPORTS_DIR, f"{report_id}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "AI Report")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Question: {question}")
    y -= 20

    # Answer
    c.setFont("Helvetica", 11)
    for line in wrap(answer, 95):
        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 11)
        c.drawString(40, y, line)
        y -= 14

    # Citations
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Citations")
    y -= 18
    c.setFont("Helvetica", 9)
    for cit in citations:
        line = f"- {cit.get('filename')} p.{cit.get('page')}: {cit.get('snippet', '')[:120]}"
        for l in wrap(line, 100):
            if y < 60:
                c.showPage()
                y = height - 40
                c.setFont("Helvetica", 9)
            c.drawString(40, y, l)
            y -= 12

    # Web sources if any
    if web_sources:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, "Web sources")
        y -= 18
        c.setFont("Helvetica", 9)
        for src in web_sources:
            line = f"- {src.get('title', 'source')} {src.get('url', '')}"
            for l in wrap(line, 100):
                if y < 60:
                    c.showPage()
                    y = height - 40
                    c.setFont("Helvetica", 9)
                c.drawString(40, y, l)
                y -= 12

    c.save()
    return {"report_id": report_id, "path": path}
