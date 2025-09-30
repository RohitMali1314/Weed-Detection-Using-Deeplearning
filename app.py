from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ultralytics import YOLO
import cv2
import uuid

app = Flask(__name__)
CORS(app)   # allow frontend to access API

# Load your trained model
model = YOLO("best.pt")

# Create directories if not exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)


@app.route("/")
def home():
    return jsonify({"message": "Weed Detection API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    # Save uploaded image
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    # Run YOLO prediction
    results = model(filepath)

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                detections.append({
                    "label": label,
                    "confidence": round(confidence * 100, 1),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

    # Draw bounding boxes
    image = cv2.imread(filepath)
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
        label = detection["label"]
        confidence = detection["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{label} {confidence}%"
        cv2.putText(image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_filename = f"result_{filename}"
    result_path = os.path.join("results", result_filename)
    cv2.imwrite(result_path, image)

    # Use request.host_url for correct domain
    base_url = request.host_url

    return jsonify({
        "detections": detections,
        "result_image_url": f"{base_url}result/{result_filename}",
        "original_image_url": f"{base_url}uploads/{filename}"
    })


@app.route("/result/<filename>")
def get_result(filename):
    return send_from_directory("results", filename)


@app.route("/uploads/<filename>")
def get_upload(filename):
    return send_from_directory("uploads", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
