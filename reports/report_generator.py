from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import darkblue
from datetime import datetime


def create_pdf(
    filename,
    prediction,
    confidence,
    severity,
    top3_predictions,
    recommendation,
):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    title = styles["Heading1"]
    title.alignment = TA_CENTER
    title.textColor = darkblue

    heading = styles["Heading2"]

    normal = styles["BodyText"]

    story = []

    story.append(Paragraph("SkinSense AI Report", title))

    story.append(
        Paragraph(
            datetime.now().strftime("%d %B %Y %H:%M"),
            normal,
        )
    )

    story.append(Paragraph("<br/><br/>", normal))

    story.append(Paragraph("Prediction", heading))
    story.append(
        Paragraph(
            f"<b>{prediction}</b>",
            normal,
        )
    )

    story.append(
        Paragraph(
            f"Confidence : {confidence*100:.2f}%",
            normal,
        )
    )

    story.append(
        Paragraph(
            f"Severity : {severity}",
            normal,
        )
    )

    story.append(Paragraph("<br/>", normal))

    story.append(Paragraph("Top Predictions", heading))

    for skin, score in top3_predictions:
        story.append(
            Paragraph(
                f"• {skin.title()} : {score*100:.2f}%",
                normal,
            )
        )

    story.append(Paragraph("<br/>", normal))

    story.append(
        Paragraph(
            "AI Dermatology Recommendation",
            heading,
        )
    )

    recommendation = recommendation.replace("\n", "<br/>")

    story.append(
        Paragraph(
            recommendation,
            normal,
        )
    )

    story.append(Paragraph("<br/><br/>", normal))

    story.append(
        Paragraph(
            "<b>Disclaimer</b><br/>"
            "This report is generated using AI and should not replace professional medical advice.",
            normal,
        )
    )

    doc.build(story)