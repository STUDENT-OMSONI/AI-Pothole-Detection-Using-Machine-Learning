from flask import Flask, render_template, request, send_file, Response, url_for
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
REPORT_FOLDER = BASE_DIR / "reports"
MODEL_PATH = BASE_DIR / "best.pt"

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(exist_ok=True)

# -----------------------------
# Load YOLO model
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# -----------------------------
# Helpers
# -----------------------------
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def detect_on_frame(frame):
    """
    Run pothole detection on a single frame.
    Returns:
        annotated_frame, detections
    """
    results = model.predict(source=frame, conf=0.2, imgsz=640, verbose=False)
    result = results[0]

    detections = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes, start=1):
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1

            detections.append({
                "id": i,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"pothole {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    return frame, detections


def process_image(image_path):
    file_name = Path(image_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read image.")

    annotated_img, detections = detect_on_frame(img.copy())
    pothole_count = len(detections)

    # Save output image
    output_image_name = f"detected_{timestamp}_{file_name}"
    output_image_path = OUTPUT_FOLDER / output_image_name
    cv2.imwrite(str(output_image_path), annotated_img)

    # Build report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("POTHOLE DETECTION REPORT - IMAGE")
    report_lines.append("=" * 60)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Image: {file_name}")
    report_lines.append(f"Potholes Detected: {pothole_count}")
    report_lines.append("=" * 60)

    if pothole_count > 0:
        report_lines.append("\nDetection Details:")
        report_lines.append("-" * 60)
        for d in detections:
            report_lines.append(
                f"Pothole {d['id']}: Confidence={d['confidence']:.2%}, "
                f"Width={d['width']}px, Height={d['height']}px, "
                f"Box=({d['x1']},{d['y1']}) to ({d['x2']},{d['y2']})"
            )
    else:
        report_lines.append("\nNo potholes detected.")

    report_name = f"image_report_{timestamp}.txt"
    report_path = REPORT_FOLDER / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "file_name": file_name,
        "pothole_count": pothole_count,
        "detections": detections,
        "output_image_name": output_image_name,
        "report_name": report_name
    }


def process_video(video_path):
    file_name = Path(video_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_name = f"detected_{timestamp}_{file_name.rsplit('.', 1)[0]}.mp4"
    output_video_path = OUTPUT_FOLDER / output_video_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    frames_with_potholes = 0
    max_potholes_in_frame = 0

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("POTHOLE DETECTION REPORT - VIDEO")
    report_lines.append("=" * 60)
    report_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Video: {file_name}")
    report_lines.append("=" * 60)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        annotated_frame, detections = detect_on_frame(frame.copy())
        potholes_this_frame = len(detections)

        total_detections += potholes_this_frame
        if potholes_this_frame > 0:
            frames_with_potholes += 1
        if potholes_this_frame > max_potholes_in_frame:
            max_potholes_in_frame = potholes_this_frame

        report_lines.append(
            f"Frame {frame_count}: Potholes Detected = {potholes_this_frame}"
        )

        out.write(annotated_frame)

    cap.release()
    out.release()

    report_lines.append("=" * 60)
    report_lines.append(f"Total Frames Processed: {frame_count}")
    report_lines.append(f"Frames With Potholes: {frames_with_potholes}")
    report_lines.append(f"Total Detections Across Video: {total_detections}")
    report_lines.append(f"Maximum Potholes In A Single Frame: {max_potholes_in_frame}")

    report_name = f"video_report_{timestamp}.txt"
    report_path = REPORT_FOLDER / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "file_name": file_name,
        "frame_count": frame_count,
        "frames_with_potholes": frames_with_potholes,
        "total_detections": total_detections,
        "max_potholes_in_frame": max_potholes_in_frame,
        "output_video_name": output_video_name,
        "report_name": report_name
    }


# -----------------------------
# Webcam stream generator
# -----------------------------
def generate_webcam_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, detections = detect_on_frame(frame.copy())

        pothole_count = len(detections)
        cv2.putText(
            annotated_frame,
            f"Potholes: {pothole_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return "No image file found."

    file = request.files["image"]

    if file.filename == "":
        return "No image selected."

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return "Invalid image file type."

    save_path = UPLOAD_FOLDER / file.filename
    file.save(str(save_path))

    result_data = process_image(save_path)

    return render_template(
        "result_image.html",
        original_name=result_data["file_name"],
        pothole_count=result_data["pothole_count"],
        detections=result_data["detections"],
        output_image_url=url_for("static", filename=f"outputs/{result_data['output_image_name']}"),
        report_name=result_data["report_name"]
    )


@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return "No video file found."

    file = request.files["video"]

    if file.filename == "":
        return "No video selected."

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return "Invalid video file type."

    save_path = UPLOAD_FOLDER / file.filename
    file.save(str(save_path))

    result_data = process_video(save_path)

    return render_template(
        "result_video.html",
        original_name=result_data["file_name"],
        frame_count=result_data["frame_count"],
        frames_with_potholes=result_data["frames_with_potholes"],
        total_detections=result_data["total_detections"],
        max_potholes_in_frame=result_data["max_potholes_in_frame"],
        output_video_url=url_for("static", filename=f"outputs/{result_data['output_video_name']}"),
        report_name=result_data["report_name"]
    )

@app.route("/image")
def image_page():
    return render_template("image.html")

@app.route("/video")
def video_page():
    return render_template("video.html")

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


@app.route("/webcam_feed")
def webcam_feed():
    return Response(
        generate_webcam_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/download-report/<report_name>")
def download_report(report_name):
    report_path = REPORT_FOLDER / report_name
    if not report_path.exists():
        return "Report not found."
    return send_file(report_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=8080)