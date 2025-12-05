#!/usr/bin/env python3
"""
Dental Cavity Detection - Professional Edition
Beautiful GUI for AI-powered cavity detection.
"""

import cv2
import csv
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import gradio as gr
from ultralytics import YOLO


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame_number: Optional[int] = None


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    frame_number: Optional[int] = None

    @property
    def cavity_count(self) -> int:
        return sum(1 for d in self.detections if d.class_id == 0)

    @property
    def normal_count(self) -> int:
        return sum(1 for d in self.detections if d.class_id == 1)


# ============================================================================
# DETECTOR CLASS
# ============================================================================

class CavityDetector:
    LABELS = ["Cavity", "Normal"]
    COLORS = [(255, 255, 255), (0, 255, 100)]

    def __init__(self, model_path: str = "cavity detection.pt", confidence: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.width = 1280
        self.height = 720

    def detect(self, frame: np.ndarray, frame_num: int = 0) -> DetectionResult:
        resized = cv2.resize(frame, (self.width, self.height))
        results = self.model.predict(resized, conf=self.confidence, verbose=False)

        detections = []
        output = resized.copy()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(Detection(cls, self.LABELS[cls], conf, (x1, y1, x2, y2), frame_num))

                color = self.COLORS[cls]

                # Draw box
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

                # Label with background
                label = f"{self.LABELS[cls]} {conf:.0%}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(output, (x1, y1 - h - 12), (x1 + w + 10, y1), color, -1)
                cv2.putText(output, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Pink overlay for cavities
                if cls == 0 and y2 > y1 and x2 > x1:
                    region = resized[y1:y2, x1:x2]
                    if region.size > 0:
                        overlay = np.full_like(region, (255, 180, 180), dtype=np.uint8)
                        output[y1:y2, x1:x2] = cv2.addWeighted(region, 0.4, overlay, 0.6, 0)

        return DetectionResult(detections, output, frame_number=frame_num)


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    def __init__(self, results: List[DetectionResult]):
        self.results = results
        self.timestamp = datetime.now()

    def summary(self) -> dict:
        all_det = [d for r in self.results for d in r.detections]
        cavities = sum(1 for d in all_det if d.class_id == 0)
        frames_cav = sum(1 for r in self.results if r.cavity_count > 0)
        return {
            'frames': len(self.results),
            'detections': len(all_det),
            'cavities': cavities,
            'normal': len(all_det) - cavities,
            'frames_with_cav': frames_cav,
            'rate': frames_cav / max(len(self.results), 1)
        }

    def to_csv(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Frame', 'Class', 'Confidence', 'Location'])
            for r in self.results:
                for d in r.detections:
                    w.writerow([r.frame_number, d.class_name, f"{d.confidence:.1%}", d.bbox])
            s = self.summary()
            w.writerow([])
            w.writerow(['SUMMARY'])
            w.writerow(['Total Frames', s['frames']])
            w.writerow(['Cavities Found', s['cavities']])
            w.writerow(['Detection Rate', f"{s['rate']:.1%}"])
        return path

    def to_pdf(self, path: str) -> str:
        try:
            from fpdf import FPDF
        except ImportError:
            return self.to_csv(path.replace('.pdf', '.csv'))

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()

        pdf = FPDF()
        pdf.add_page()

        # Purple header
        pdf.set_fill_color(102, 126, 234)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_y(12)
        pdf.cell(0, 10, 'Cavity Detection Report', align='C', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.cell(0, 8, self.timestamp.strftime('%B %d, %Y at %H:%M'), align='C', ln=True)

        pdf.set_text_color(0, 0, 0)
        pdf.ln(15)

        # Summary
        pdf.set_fill_color(245, 245, 250)
        pdf.rect(15, 50, 180, 40, 'F')
        pdf.set_xy(20, 55)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, 'Summary', ln=True)
        pdf.set_font('Helvetica', '', 11)
        pdf.set_x(20)
        pdf.cell(60, 7, f"Images Analyzed: {s['frames']}")
        pdf.cell(60, 7, f"Cavities Found: {s['cavities']}")
        pdf.cell(60, 7, f"Normal: {s['normal']}", ln=True)

        # Cavity details
        cavity_det = [(r, d) for r in self.results for d in r.detections if d.class_id == 0]
        if cavity_det:
            pdf.ln(15)
            pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 10, f'Cavity Details ({len(cavity_det)} found)', ln=True)

            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_fill_color(102, 126, 234)
            pdf.set_text_color(255)
            pdf.cell(60, 8, 'Source', 1, 0, 'C', True)
            pdf.cell(40, 8, 'Confidence', 1, 0, 'C', True)
            pdf.cell(70, 8, 'Location', 1, 1, 'C', True)

            pdf.set_text_color(0)
            pdf.set_font('Helvetica', '', 10)
            for i, (r, d) in enumerate(cavity_det[:25]):
                fill = i % 2 == 0
                pdf.set_fill_color(248, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
                pdf.cell(60, 7, str(r.source_path or f"Frame {r.frame_number}")[:25], 1, 0, 'C', fill)
                pdf.cell(40, 7, f"{d.confidence:.1%}", 1, 0, 'C', fill)
                pdf.cell(70, 7, str(d.bbox), 1, 1, 'C', fill)

        pdf.ln(10)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(128)
        pdf.multi_cell(0, 5, "DISCLAIMER: For educational purposes only. Consult a dental professional for diagnosis.")

        pdf.output(path)
        return path


# ============================================================================
# GLOBAL STATE
# ============================================================================

detector = None

def get_detector(conf: float = 0.25) -> CavityDetector:
    global detector
    if detector is None or abs(detector.confidence - conf) > 0.01:
        detector = CavityDetector(confidence=conf)
    return detector


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def analyze_image(image: np.ndarray, confidence: float):
    if image is None:
        return None, "**Please upload an image**"

    det = get_detector(confidence)
    result = det.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    output = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

    if result.cavity_count > 0:
        status = f"""### Detection Complete

| Metric | Value |
|--------|-------|
| Cavities Found | **{result.cavity_count}** |
| Normal Regions | {result.normal_count} |

**Details:**
"""
        for d in result.detections:
            icon = "CAVITY" if d.class_id == 0 else "Normal"
            status += f"- **{icon}** - {d.confidence:.0%} confidence\n"
    else:
        status = f"""### No Cavities Detected

| Metric | Value |
|--------|-------|
| Cavities | 0 |
| Normal Regions | {result.normal_count} |

*Try lowering sensitivity to detect more.*"""

    return output, status


def analyze_video(video_path: str, confidence: float):
    if not video_path:
        return None, "**Please upload a video**"

    det = get_detector(confidence)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "**Error: Cannot open video**"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_out = os.path.join(tempfile.gettempdir(), "detected_output.mp4")
    writer = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (det.width, det.height))

    frame_num = 0
    total_cav = 0
    frames_with_cav = 0

    print(f"\n{'='*50}")
    print(f"  Processing {total} frames...")
    print('='*50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = det.detect(frame, frame_num)
        writer.write(result.annotated_image)

        if result.cavity_count > 0:
            frames_with_cav += 1
            total_cav += result.cavity_count

        frame_num += 1
        if frame_num % 30 == 0 or frame_num == total:
            pct = frame_num * 100 // max(total, 1)
            bar = "=" * (pct // 2) + "-" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n{'='*50}\n")

    status = f"""### Video Complete

| Metric | Value |
|--------|-------|
| Frames | {frame_num} |
| Cavities | **{total_cav}** |
| Affected Frames | {frames_with_cav} |
| Rate | {frames_with_cav * 100 // max(frame_num, 1)}% |"""

    return temp_out, status


def analyze_batch(files, confidence: float, report_fmt: str):
    if not files:
        return [], "**Upload images first**", None

    det = get_detector(confidence)
    results = []
    gallery = []

    for f in files:
        try:
            path = f if isinstance(f, str) else f.name
            img = cv2.imread(path)
            if img is None:
                continue

            result = det.detect(img)
            result.source_path = Path(path).name
            results.append(result)

            rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            label = f"{result.source_path}: {result.cavity_count} cavities"
            gallery.append((rgb, label))
        except Exception as e:
            print(f"Error: {e}")

    if not results:
        return [], "**No images processed**", None

    total_cav = sum(r.cavity_count for r in results)

    status = f"""### Batch Complete

| Metric | Value |
|--------|-------|
| Images | {len(results)} |
| Total Cavities | **{total_cav}** |
| With Cavities | {sum(1 for r in results if r.cavity_count > 0)} |

**Per Image:**
"""
    for r in results:
        status += f"- **{r.source_path}**: {r.cavity_count} cavities\n"

    gen = ReportGenerator(results)
    temp = tempfile.gettempdir()
    report_path = gen.to_pdf(os.path.join(temp, "report.pdf")) if report_fmt == "PDF" else gen.to_csv(os.path.join(temp, "report.csv"))

    return gallery, status, report_path


def create_report(image: np.ndarray, confidence: float, fmt: str):
    if image is None:
        return None

    det = get_detector(confidence)
    result = det.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    result.source_path = "uploaded_image"

    gen = ReportGenerator([result])
    temp = tempfile.gettempdir()

    if fmt == "PDF":
        return gen.to_pdf(os.path.join(temp, "cavity_report.pdf"))
    return gen.to_csv(os.path.join(temp, "cavity_report.csv"))


# ============================================================================
# CREATE APP - GRADIO 6 COMPATIBLE
# ============================================================================

def create_app():
    with gr.Blocks() as app:

        # Header
        gr.Markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">Dental Cavity Detection</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 10px; font-size: 1.1em;">AI-Powered Analysis for Dental X-rays and Videos</p>
        </div>
        """)

        with gr.Tabs():

            # ===== IMAGE TAB =====
            with gr.Tab("Image Analysis"):
                gr.Markdown("### Upload a dental X-ray or photo for instant analysis")

                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="Upload Image", type="numpy")
                        img_conf = gr.Slider(0.1, 1.0, 0.25, step=0.05,
                                            label="Detection Sensitivity",
                                            info="Lower = more sensitive, Higher = stricter")
                        img_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                        gr.Markdown("---")
                        with gr.Row():
                            rpt_fmt = gr.Radio(["PDF", "CSV"], value="PDF", label="Report Format")
                            rpt_btn = gr.Button("Download Report", variant="secondary")
                        rpt_file = gr.File(label="Report")

                    with gr.Column():
                        img_output = gr.Image(label="Detection Result")
                        img_status = gr.Markdown()

                img_btn.click(analyze_image, [img_input, img_conf], [img_output, img_status])
                rpt_btn.click(create_report, [img_input, img_conf, rpt_fmt], [rpt_file])

            # ===== VIDEO TAB =====
            with gr.Tab("Video Analysis"):
                gr.Markdown("### Upload a dental video for frame-by-frame detection")

                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Upload Video")
                        vid_conf = gr.Slider(0.1, 1.0, 0.25, step=0.05,
                                            label="Detection Sensitivity",
                                            info="Lower = more sensitive")
                        vid_btn = gr.Button("Process Video", variant="primary", size="lg")
                        gr.Markdown("*Check console for progress bar*")

                    with gr.Column():
                        vid_output = gr.Video(label="Processed Video")
                        vid_status = gr.Markdown()

                vid_btn.click(analyze_video, [vid_input, vid_conf], [vid_output, vid_status])

            # ===== BATCH TAB =====
            with gr.Tab("Batch Processing"):
                gr.Markdown("### Process multiple images at once with combined reporting")

                with gr.Row():
                    with gr.Column():
                        batch_input = gr.File(label="Upload Multiple Images", file_count="multiple")
                        batch_conf = gr.Slider(0.1, 1.0, 0.25, step=0.05, label="Detection Sensitivity")
                        batch_fmt = gr.Radio(["PDF", "CSV"], value="PDF", label="Report Format")
                        batch_btn = gr.Button("Analyze All Images", variant="primary", size="lg")

                    with gr.Column():
                        batch_status = gr.Markdown()
                        batch_report = gr.File(label="Download Report")

                batch_gallery = gr.Gallery(label="Detection Results", columns=2, height=400)

                batch_btn.click(analyze_batch, [batch_input, batch_conf, batch_fmt],
                               [batch_gallery, batch_status, batch_report])

            # ===== HELP TAB =====
            with gr.Tab("Help"):
                gr.Markdown("""
                ## Quick Guide

                ### Image Analysis
                1. Upload a dental X-ray or intraoral image
                2. Adjust the sensitivity slider
                3. Click **Analyze Image**
                4. Download PDF/CSV report if needed

                ### Video Analysis
                1. Upload a dental video (MP4, AVI, MOV)
                2. Click **Process Video**
                3. Watch progress in console
                4. Download processed video

                ### Batch Processing
                1. Upload multiple images at once
                2. Click **Analyze All Images**
                3. View results in gallery
                4. Download combined report

                ---

                ## Sensitivity Guide

                | Level | Value | Use Case |
                |-------|-------|----------|
                | High Sensitivity | 0.1 - 0.3 | Find all potential issues |
                | Balanced | 0.3 - 0.5 | General use (recommended) |
                | High Precision | 0.5 - 0.8 | Reduce false positives |
                | Strict | 0.8 - 1.0 | Only clear cavities |

                ---

                ## Visual Guide
                - **White/Pink overlay** = Detected cavity
                - **Green box** = Normal tissue
                - **Percentage** = AI confidence

                ---

                ## Disclaimer
                This tool is for **educational purposes only**.
                Always consult a qualified dental professional.
                """)

        # Footer
        gr.Markdown("""
        <div style="text-align: center; padding: 15px; margin-top: 20px; border-top: 1px solid #eee; color: #666;">
            Built with YOLO & Gradio | For Educational Use Only
        </div>
        """)

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("       DENTAL CAVITY DETECTION")
    print("="*60)
    print("\nLoading AI model...")

    get_detector()

    print("Model ready!")
    print("\nStarting web interface...")
    print("="*60 + "\n")

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )
