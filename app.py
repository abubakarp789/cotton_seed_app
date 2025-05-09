import os
from typing import Tuple, Dict, List
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session, jsonify, make_response
from werkzeug.utils import secure_filename
import numpy as np
import csv
import io
import json
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO
import cv2

# Constants
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'cotton_seeds_yolo.pt'
LABELS_PATH = 'labels.txt'
BAR_CHART_PATH = 'static/bar_chart.png'

# Flask app setup
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
torch.set_num_threads(1)
model = YOLO(MODEL_PATH)

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_class_names(labels_path: str = LABELS_PATH) -> Dict[int, str]:
    """Load class names from labels.txt."""
    with open(labels_path, 'r') as f:
        return {int(line.split()[0]): line.split()[1] for line in f.readlines()}

CLASS_NAMES = load_class_names()


def predict_image(image_path: str) -> Dict[str, float]:
    """Predict class probabilities for the given image using YOLO model."""
    results = model(image_path)[0]
    confidences = {name: 0.0 for name in CLASS_NAMES.values()}
    if len(results.boxes) > 0:
        for box in results.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            name = CLASS_NAMES[class_id]
            if conf > confidences[name]:
                confidences[name] = conf
    # Normalize to sum to 1 if any prediction
    total = sum(confidences.values())
    if total > 0:
        confidences = {k: v / total for k, v in confidences.items()}
    return confidences

def plot_bar_chart(confidences: Dict[str, float], output_path: str = BAR_CHART_PATH) -> None:
    """Plot and save a bar chart of class probabilities."""
    plt.figure(figsize=(8, 4))
    bars = plt.bar(confidences.keys(), [v * 100 for v in confidences.values()], color=['#003366', '#008080', '#FFD700', '#FFFFFF', '#005f73', '#0a9396', '#e9d8a6'])
    plt.ylabel('Confidence (%)')
    plt.title('Cotton Seed Species Prediction')
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()

def save_bar_chart(confidences: Dict[str, float], output_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(confidences.keys(), [v * 100 for v in confidences.values()], color=['#7bb661', '#a3c585', '#e6e2d3', '#b7a16a', '#f6f8f3', '#c2b280', '#b6d7a8'])
    plt.ylabel('Confidence (%)')
    plt.title('Cotton Seed Species Prediction')
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()

def predictions_to_csv(predictions: List[Dict], class_names: List[str]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Image'] + class_names)
    for pred in predictions:
        row = [pred['filename']] + [f"{pred['confidences'].get(name, 0.0)*100:.2f}" for name in class_names]
        writer.writerow(row)
    return output.getvalue()

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'history' not in session:
        session['history'] = []
    predictions = []
    filenames = []
    if request.method == 'POST':
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            flash('No selected file(s)')
            return redirect(request.url)
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                confidences = predict_image(file_path)
                predictions.append({'filename': filename, 'confidences': confidences})
                filenames.append(filename)
            else:
                flash('Allowed file types are png, jpg, jpeg')
                return redirect(request.url)
        # Add to session history
        session['history'] += predictions
        session.modified = True
    else:
        predictions = []
        filenames = []
    return render_template('index.html', predictions=predictions, filenames=filenames, class_names=list(CLASS_NAMES.values()), history=session.get('history', []))

@app.route('/reset', methods=['GET', 'POST'])
def reset():
    session.pop('history', None)
    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    history = session.get('history', [])
    class_names = list(CLASS_NAMES.values())
    csv_data = predictions_to_csv(history, class_names)
    response = make_response(csv_data)
    response.headers['Content-Disposition'] = 'attachment; filename=predictions.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/model_info')
def model_info():
    info = {
        'title': 'Cotton Seed Species Predictor',
        'model': 'YOLOv8',
        'classes': list(CLASS_NAMES.values()),
        'description': 'This model predicts the species of a cotton seed from an image using a YOLOv8 deep learning model trained on labeled cotton seed images.'
    }
    return jsonify(info)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Removed app.run(debug=True) for Azure deployment; use gunicorn instead. 