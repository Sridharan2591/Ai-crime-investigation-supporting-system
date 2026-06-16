from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from datetime import datetime
import os

def generate_forensic_report(image_res, video_res, audio_res, filename="Forensic_Report.pdf"):
    report_path = os.path.join("uploads", filename)
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    # --- Header ---
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width/2, height - 50, "DIGITAL FORENSIC ANALYSIS REPORT")
    
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, height - 80, width - 50, height - 80)

    # --- Case Information ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 110, "Case Summary:")
    c.setFont("Helvetica", 11)
    c.drawString(60, height - 130, f"• Image Analysis: {image_res}")
    c.drawString(60, height - 150, f"• Video Analysis: {video_res}")
    c.drawString(60, height - 170, f"• Audio Analysis: {audio_res}")

    # --- Forensic Conclusion ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 210, "Final Verdict:")
    
    # Logic to determine if the overall evidence is suspicious
    if "Tampered" in image_res or "Fake" in video_res or "TAMPERED" in audio_res:
        verdict = "SUSPICIOUS - Evidence shows signs of digital manipulation."
        c.setFillColor(colors.red)
    else:
        verdict = "AUTHENTIC - No significant manipulation detected."
        c.setFillColor(colors.green)
        
    c.drawString(60, height - 230, verdict)

    # --- Footer ---
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width/2, 30, "Confidential - Crime AI Forensic Laboratory")

    c.save()
    return report_path