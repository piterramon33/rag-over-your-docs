"""Regenerates my_docs/cold_chain_handbook.pdf (fictional sample, 3 pages). Needs reportlab."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("1. Storage temperatures", [
        "This handbook is a fictional sample written for the demo of this repository.",
        "Fresh meat must be stored at or below 4 °C at all times. Chilled dairy is held between 2 °C and 6 °C.",
        "Frozen products must be kept at -18 °C or colder. Chambers are checked every four hours by the shift supervisor.",
    ]),
    ("2. Monitoring and records", [
        "Probe thermometers are calibrated once a month against a reference thermometer.",
        "Temperature records are retained for two years and must be available to inspectors on request.",
        "Each chamber has an automatic data logger that stores a reading every 15 minutes.",
    ]),
    ("3. Deviations and loading", [
        "When a critical limit is exceeded, the operator must notify the quality manager immediately and isolate the affected product.",
        "A dock door can stay open for a maximum of 10 minutes during loading.",
        "Product that stayed above its limit for more than two hours is placed on hold until the quality manager decides its disposition.",
    ]),
]

if __name__ == "__main__":
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph("Northwind Cold Storage - Cold Chain Handbook", styles["Title"]) if i == 0 else Spacer(1, 1))
        story.append(Paragraph(title, styles["Heading2"]))
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate("my_docs/cold_chain_handbook.pdf", pagesize=A4).build(story)
    print("wrote my_docs/cold_chain_handbook.pdf")
