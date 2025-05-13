from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from ultralytics import YOLO
from PIL import Image
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
import textwrap
import uuid
import os
from datetime import datetime

app = FastAPI()

model = YOLO("best.pt")

OUTPUT_DIR = "outputs"
PDF_DIR = "pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Use landscape A4 size
PAGE_SIZE = landscape(A4)

def draw_cover_page(c, total_images):
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(PAGE_SIZE[0] / 2, PAGE_SIZE[1] - 40, "Damage Detection Report")

    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_SIZE[0] / 2, PAGE_SIZE[1] - 60, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica", 12)
    c.setFont("Helvetica", 12)
    text = (
        "This report presents the results of an advanced AI-driven vehicle damage detection system.\n\n"
        "The software is designed to automate the process of identifying vehicle damages by analyzing images of vehicles. "
        "It detects various types of visible damage, including dents, scratches, cracks, and other structural imperfections. "
        "By using state-of-the-art deep learning models trained on a wide range of vehicle damage data, the system offers highly accurate and reliable damage detection capabilities.\n\n"
        "The primary objective of this system is to streamline and accelerate the vehicle inspection process, allowing for faster, "
        "more consistent, and more accurate assessments compared to manual inspection methods. This technology can be particularly useful "
        "in industries such as insurance, automotive repair, fleet management, and vehicle rental services, where timely damage detection is critical.\n\n"
        "The process begins when an image is uploaded to the system, which is then processed by the AI model. The model identifies "
        "any damages in the image and highlights them with bounding boxes. These bounding boxes correspond to the location of the detected "
        "damage, and each is accompanied by a confidence score, indicating the likelihood of the damage being correctly identified.\n\n"
        "The system's AI model is constantly refined and improved to handle a wide variety of vehicle types, damages, and environmental conditions, "
        "ensuring accurate results in diverse scenarios.\n\n"
        f"Total Images Processed: {total_images}\n\n"
        "For each image submitted, this report provides the original image on the left and the processed image with detected damages on the right. "
        "Below each pair of images, you will find a table listing the detected objects, their class (type of damage), the confidence level for "
        "each detection, and the corresponding bounding box coordinates. This detailed information allows for easy inspection and validation of the results.\n\n"
        "By automating this process, the system reduces human error, accelerates claim processing times, and enhances the overall efficiency of vehicle inspections.\n\n"
        "We hope this report provides valuable insights into the performance and accuracy of the damage detection system, and that it supports the "
        "continuous improvement of vehicle assessment operations."
    )

    text_object = c.beginText(50, 500)
    text_object.setFont("Helvetica", 13)
    wrapped_lines = []
    for paragraph in text.split("\n\n"):
        lines = textwrap.wrap(paragraph, width=120)
        wrapped_lines.extend(lines + [""])

    for line in wrapped_lines:
        text_object.textLine(line)

    c.drawText(text_object)

    c.showPage()

def draw_images_and_table(c, index, total, original_path, result_path, result_data, y_start):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_start + 30, f"Image {index} of {total}")

    # Draw original image (larger)
    c.setFont("Helvetica", 12)
    c.drawString(50, y_start, "Original Image")
    c.drawImage(original_path, 50, y_start - 260, width=350, height=250)

    # Draw result image (larger)
    c.drawString(450, y_start, "Detected Image")
    c.drawImage(result_path, 450, y_start - 260, width=350, height=250)

    # Draw table below
    y = y_start - 280
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected Objects:")
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Class")
    c.drawString(250, y, "Confidence")

    c.setFont("Helvetica", 12)
    for r in result_data:
        y -= 15
        c.drawString(50, y, r['class'])
        c.drawString(250, y, f"{r['conf']:.2f}")

    # Just a 2-line gap instead of full page
    c.showPage()

@app.post("/detect/")
async def detect_damage(files: List[UploadFile] = File(...)):
    pdf_filename = os.path.join(PDF_DIR, f"{uuid.uuid4()}.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=PAGE_SIZE)
    y_position = PAGE_SIZE[1] - 100

    draw_cover_page(c, total_images=len(files))

    for index, file in enumerate(files, start=1):
        file_ext = file.filename.split('.')[-1]
        temp_filename = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(OUTPUT_DIR, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        results = model(temp_path, save=True, project=OUTPUT_DIR, name='runs', exist_ok=True)

        detected_path = os.path.join(OUTPUT_DIR, 'runs', os.path.basename(temp_path))
        if not os.path.exists(detected_path):
            detected_path = detected_path.replace("jpeg", "jpg")

        result_data = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xywh = box.xywh[0].tolist()
            result_data.append({
                "class": model.names[cls_id],
                "conf": conf,
                "bbox": [round(x, 2) for x in xywh]
            })

        draw_images_and_table(c, index, len(files), temp_path, detected_path, result_data, y_position)

    c.save()
    return FileResponse(pdf_filename, media_type="application/pdf", filename="damage_report.pdf")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8600)
