from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List

from reportlab.lib.utils import ImageReader
from ultralytics import YOLO
from PIL import Image
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
import uuid
import os
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import seaborn as sns

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("auto_damage_model.pt")

OUTPUT_DIR = "outputs"
PDF_DIR = "pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

PAGE_SIZE = landscape(A4)


def draw_cover_with_details(c, total_images, class_counts):
    # Header
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(PAGE_SIZE[0] / 2, PAGE_SIZE[1] - 40, "Damage Detection Report")

    # Timestamp & total image info
    c.setFont("Helvetica", 10)
    c.drawCentredString(
        PAGE_SIZE[0] / 2,
        PAGE_SIZE[1] - 60,
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    c.setFont("Helvetica", 12)
    c.drawCentredString(
        PAGE_SIZE[0] / 2, PAGE_SIZE[1] - 85, f"Total Images Processed: {total_images}"
    )

    # Policy Details
    y = PAGE_SIZE[1] - 120
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Policy Details:")

    y -= 25
    c.setFont("Helvetica", 12)
    policies = [
        "Comprehensive",
        "Zero Depreciation",
        "Own Damage",
        "Third-Party Liability Insurance",
        "Collision Insurance",
    ]
    c.drawString(60, y, "Policy Types:")
    for p in policies:
        y -= 18
        c.drawString(80, y, f"- {p}")

    y -= 30
    c.drawString(60, y, f"Policy Start Date: {datetime.now().strftime('%Y-%m-%d')}")
    y -= 20
    c.drawString(
        60,
        y,
        f"Policy Renewal Date: {(datetime.now().replace(year=datetime.now().year + 1)).strftime('%Y-%m-%d')}",
    )

    # Table Headers
    y -= 60
    table_top = y
    row_height = 25
    col1_x, col2_x, col3_x = 50, 300, 450
    col_widths = [250, 150, 150]
    table_width = sum(col_widths)

    # Header row
    c.setFillColorRGB(0.9, 0.9, 0.9)
    c.rect(col1_x, y, table_width, row_height, fill=True, stroke=True)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(col1_x + 10, y + 7, "Damage Type")
    c.drawString(col2_x + 10, y + 7, "Count")
    c.drawString(col3_x + 10, y + 7, "Covered")

    # Draw vertical column lines
    c.line(col2_x, y, col2_x, y - (row_height * (len(class_counts) + 1)))
    c.line(col3_x, y, col3_x, y - (row_height * (len(class_counts) + 1)))
    c.line(
        col1_x + table_width,
        y,
        col1_x + table_width,
        y - (row_height * (len(class_counts) + 1)),
    )

    # Table rows
    c.setFont("Helvetica", 12)
    y -= row_height
    for damage_type, count in class_counts.items():
        # Draw full row rectangle (optional)
        c.rect(col1_x, y, table_width, row_height, fill=False, stroke=True)

        # Add data in columns
        c.drawString(col1_x + 10, y + 7, damage_type)
        c.drawString(col2_x + 10, y + 7, str(count))
        c.drawString(col3_x + 10, y + 7, "Yes")
        y -= row_height

    c.showPage()


def draw_summary_chart(c, class_counts):
    if not class_counts:
        return

    labels = list(class_counts.keys())
    values = [class_counts[k] for k in labels]

    # Set seaborn style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))  # Slightly bigger chart

    # Create bar plot with seaborn
    ax = sns.barplot(x=labels, y=values, palette="viridis")
    ax.set_xlabel("Damage Type", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Summary of Detected Damages", fontsize=14)
    plt.xticks(rotation=30)
    plt.tight_layout()

    # Save to buffer and draw on PDF
    buf = BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()

    img_reader = ImageReader(buf)

    # Adjusted position and size for slightly bigger appearance
    chart_width = 680  # Increase width
    chart_height = 340  # Increase height
    chart_x = (PAGE_SIZE[0] - chart_width) / 2  # Centered horizontally
    chart_y = 180  # Adjusted Y position for fit

    c.drawImage(img_reader, chart_x, chart_y, width=chart_width, height=chart_height)
    c.showPage()


def draw_images_and_table(
    c, index, total, original_path, result_path, result_data, y_start
):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_start + 30, f"Image {index} of {total}")

    c.setFont("Helvetica", 12)
    c.drawString(50, y_start, "Original Image")
    c.drawImage(original_path, 50, y_start - 260, width=350, height=250)

    c.drawString(450, y_start, "Detected Image")
    c.drawImage(result_path, 450, y_start - 260, width=350, height=250)

    y = y_start - 280
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected Objects:")
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Class")
    c.drawString(250, y, "Accuracy")

    c.setFont("Helvetica", 12)
    for r in result_data:
        y -= 15
        c.drawString(50, y, r["class"])
        c.drawString(250, y, f"{r['conf'] * 100:.2f}%")

    c.showPage()


@app.post("/detection-report")
async def detect_damage(files: List[UploadFile] = File(...)):
    pdf_filename = os.path.join(PDF_DIR, f"{uuid.uuid4()}.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=PAGE_SIZE)

    total_class_counts = {}

    detection_results = []

    for index, file in enumerate(files, start=1):
        file_ext = file.filename.split(".")[-1]
        if file_ext.lower() not in {"jpg", "jpeg", "png", "bmp", "tiff"}:
            return {
                "error": "Invalid file type. Only JPEG, PNG, BMP, and TIFF files are allowed."
            }

        temp_filename = f"{uuid.uuid4()}.{file_ext}"
        temp_path = os.path.join(OUTPUT_DIR, temp_filename)

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        results = model(
            temp_path, save=True, project=OUTPUT_DIR, name="runs", exist_ok=True
        )

        detected_path = os.path.join(OUTPUT_DIR, "runs", os.path.basename(temp_path))
        if not os.path.exists(detected_path):
            detected_path = detected_path[: detected_path.rfind(".")] + ".jpg"

        result_data = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xywh = box.xywh[0].tolist()
            damage_class = model.names[cls_id]
            result_data.append(
                {
                    "class": damage_class,
                    "conf": conf,
                    "bbox": [round(x, 2) for x in xywh],
                }
            )
            total_class_counts[damage_class] = (
                total_class_counts.get(damage_class, 0) + 1
            )

        detection_results.append((index, temp_path, detected_path, result_data))

    # Draw the combined cover and details table
    draw_cover_with_details(c, total_images=len(files), class_counts=total_class_counts)

    # Then draw chart and image sections
    draw_summary_chart(c, total_class_counts)

    y_position = PAGE_SIZE[1] - 100
    for index, original_path, detected_path, result_data in detection_results:
        draw_images_and_table(
            c, index, len(files), original_path, detected_path, result_data, y_position
        )

    c.save()
    return FileResponse(
        pdf_filename, media_type="application/pdf", filename="damage_report.pdf"
    )
