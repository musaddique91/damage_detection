from ultralytics import YOLO
import cv2

# Load your trained model (best.pt from training)
model = YOLO('best.pt')

# Path to an image or a folder
image_path = 'test'

# Run detection
results = model(image_path, save=True)

# Show results
for r in results:
    # Display image with bounding boxes
    #r.show()

    # Or save the result image
    r.save(filename='output.jpg')

    # Print detected classes
    print("Detected parts:", r.names)
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Part: {r.names[cls]} | Confidence: {conf:.2f}")