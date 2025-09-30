from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load your trained YOLO model
model = YOLO('best.pt')  # Ensure best.pt is in the backend folder

# Route to test if server is live
@app.route('/')
def index():
    return jsonify({"message": "YOLO Flask app is running!"})

# Route to detect objects in uploaded image
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_file = request.files['image']
    img_name = f"{uuid.uuid4()}.jpg"
    img_path = os.path.join("/tmp", img_name)
    img_file.save(img_path)

    # Perform detection
    results = model(img_path)
    detections = results[0].boxes.data.tolist()  # Convert to list

    return jsonify({"detections": detections})

# This part is optional if using Gunicorn; Flask fallback
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns this dynamically
    app.run(host="0.0.0.0", port=port)
