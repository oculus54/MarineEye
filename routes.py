"""
routes.py — Flask route handlers for MarineEye oil spill detection app.

Handles:
  - Home page
  - Detection page (upload + YOLO inference)
  - Dashboard (past detection history)
  - Delete record
"""

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app
)
from ultralytics import YOLO
import cv2
import os
import uuid
import json
from datetime import datetime

main = Blueprint("main", __name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model():
    """Lazy-load the YOLO model (cached on app config after first call)."""
    if "YOLO_MODEL" not in current_app.config:
        weights = current_app.config["WEIGHTS_PATH"]
        current_app.config["YOLO_MODEL"] = YOLO(weights)
    return current_app.config["YOLO_MODEL"]


def _load_history():
    """Load detection history from the JSON file."""
    path = current_app.config["HISTORY_FILE"]
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def _save_history(records):
    """Persist detection history to the JSON file."""
    path = current_app.config["HISTORY_FILE"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _append_record(record):
    """Append a single record and save."""
    records = _load_history()
    records.insert(0, record)  # newest first
    _save_history(records)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@main.route("/")
def index():
    """Landing / home page."""
    return render_template("index.html", active_page="home")


@main.route("/detect", methods=["GET", "POST"])
def detect():
    """Detection page — upload image and run YOLO inference."""
    result_image = None
    detections = []

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please select an image to upload.", "error")
            return redirect(url_for("main.detect"))

        # Read confidence from the slider (default 0.25)
        conf_thresh = float(request.form.get("confidence", 0.25))

        # Save uploaded file
        uid = uuid.uuid4().hex
        safe_name = file.filename.replace(" ", "_")
        filename = f"{uid}_{safe_name}"
        upload_dir = current_app.config["UPLOAD_DIR"]
        result_dir = current_app.config["RESULT_DIR"]
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        upload_path = os.path.join(upload_dir, filename)
        file.save(upload_path)

        # Run YOLO inference
        model = _get_model()
        results = model.predict(source=upload_path, conf=conf_thresh)
        r = results[0]

        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "cls": r.names[cls_id],
                "conf": float(box.conf[0]),
            })

        # Save annotated image
        annotated = r.plot()  # BGR numpy array
        result_filename = f"result_{filename}"
        result_path = os.path.join(result_dir, result_filename)
        cv2.imwrite(result_path, annotated)

        result_image = f"/{result_path}".replace("\\", "/")

        # Persist to history
        record = {
            "id": uid,
            "filename": safe_name,
            "upload_image": f"/{upload_path}".replace("\\", "/"),
            "result_image": result_image,
            "detections": detections,
            "confidence_threshold": conf_thresh,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        _append_record(record)

        flash(
            f"Detection complete — {len(detections)} region(s) found.",
            "success" if detections else "info",
        )

    return render_template(
        "detect.html",
        active_page="detect",
        result_image=result_image,
        detections=detections,
    )


@main.route("/dashboard")
def dashboard():
    """Dashboard showing all past detection records."""
    records = _load_history()

    total_scans = len(records)
    spill_detected = sum(
        1 for r in records
        if any(d["cls"] == "oil" for d in r.get("detections", []))
    )
    clean_scans = total_scans - spill_detected

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        records=records,
        total_scans=total_scans,
        spill_detected=spill_detected,
        clean_scans=clean_scans,
    )


@main.route("/delete/<record_id>", methods=["POST"])
def delete_record(record_id):
    """Delete a detection record and its images."""
    records = _load_history()
    updated = []
    deleted = False

    for rec in records:
        if rec["id"] == record_id:
            # Try to remove the files from disk
            for key in ("upload_image", "result_image"):
                path = rec.get(key, "").lstrip("/")
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            deleted = True
        else:
            updated.append(rec)

    _save_history(updated)

    if deleted:
        flash("Record deleted successfully.", "success")
    else:
        flash("Record not found.", "error")

    return redirect(url_for("main.dashboard"))
