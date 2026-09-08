from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

WEIGHTS = "weights/best.pt"
UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
CONF_THRESH = 0.25

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

model = YOLO(WEIGHTS)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Oil Spill Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; text-align: center; }
        h1 { color: #222; }
        .box { border: 2px dashed #999; padding: 30px; border-radius: 10px; }
        img { max-width: 100%; margin-top: 20px; border-radius: 8px; }
        table { margin: 20px auto; border-collapse: collapse; }
        td, th { border: 1px solid #ccc; padding: 6px 14px; }
        button { padding: 8px 18px; margin-top: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🛢️ Oil Spill Detection</h1>
    <div class="box">
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br><button type="submit">Detect</button>
        </form>
    </div>

    {% if result_image %}
        <h2>Result</h2>
        <img src="{{ result_image }}">
        {% if detections %}
        <table>
            <tr><th>Class</th><th>Confidence</th></tr>
            {% for d in detections %}
            <tr><td>{{ d.cls }}</td><td>{{ "%.2f"|format(d.conf) }}</td></tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No detections found.</p>
        {% endif %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []

    if request.method == "POST":
        file = request.files["image"]
        if file:
            uid = uuid.uuid4().hex
            filename = f"{uid}_{file.filename}"
            upload_path = os.path.join(UPLOAD_DIR, filename)
            file.save(upload_path)

            results = model.predict(source=upload_path, conf=CONF_THRESH)
            r = results[0]

            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "cls": r.names[cls_id],
                    "conf": float(box.conf[0])
                })

            annotated = r.plot()  # BGR numpy array
            result_filename = f"result_{filename}"
            result_path = os.path.join(RESULT_DIR, result_filename)
            import cv2
            cv2.imwrite(result_path, annotated)

            result_image = f"/{result_path}"

    return render_template_string(HTML, result_image=result_image, detections=detections)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)