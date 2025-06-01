import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime
from huggingface_hub import hf_hub_download # New import!
import time

app = Flask(__name__)

# --- Model Loading Configuration ---
MODEL_FILE_NAME = "model.keras"
# REPLACE THIS WITH YOUR HUGGING FACE MODEL REPO ID
# Format: "your-username/your-model-repo-name"
HF_MODEL_REPO_ID = "YOUR_USERNAME/garbage-detection-model" # Example!

# Check if model exists, if not, try to download it from Hugging Face Hub
if not os.path.exists(MODEL_FILE_NAME):
    print(f"'{MODEL_FILE_NAME}' not found locally. Attempting to download from Hugging Face Hub...")
    try:
        # Download the model from Hugging Face Hub
        # The downloaded file will be in a cache directory by default,
        # so we'll move it to the current directory for easier loading.
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILE_NAME)
        # Move the downloaded file to the root directory for app.py to find it easily
        os.rename(model_path, MODEL_FILE_NAME)
        print(f"'{MODEL_FILE_NAME}' downloaded successfully from Hugging Face Hub.")
    except Exception as e:
        print(f"FATAL: Could not download model from Hugging Face Hub: {e}")
        # If download fails, the model will remain None, and prediction attempts will fail.
        model = None

# Load the trained model
model = None # Initialize model to None
try:
    if os.path.exists(MODEL_FILE_NAME):
        model = tf.keras.models.load_model(MODEL_FILE_NAME)
        print(f"Model loaded successfully from {MODEL_FILE_NAME}")
    else:
        print(f"Model file '{MODEL_FILE_NAME}' still not found after download attempt.")
except Exception as e:
    print(f"Error loading model from {MODEL_FILE_NAME}: {e}")
    model = None # Ensure model is None if loading fails

# Configurations
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure uploads directory exists

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    results = []
    for file in files:
        file_path = None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            try:
                img_array = preprocess_image(file_path)
                prediction = model.predict(img_array)[0][0]
                label = "Dirty" if prediction > 0.5 else "Clean"
                confidence = prediction if label == "Dirty" else 1 - prediction

                results.append({
                    'label': label,
                    'confidence': f"{confidence:.2%}",
                    'image_url': f"/static/uploads/{unique_filename}"
                })
            except Exception as e:
                results.append({
                    'label': 'Error',
                    'confidence': 'N/A',
                    'image_url': None,
                    'error': str(e)
                })
            finally:
                # Clean up the uploaded file after processing
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted uploaded file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
        else:
            results.append({
                'label': 'Error',
                'confidence': 'N/A',
                'image_url': None,
                'error': f"Invalid file type: {file.filename}"
            })

    return render_template('results.html', results=results)

if __name__ == '__main__':
    # Hugging Face Spaces sets the PORT environment variable
    # Default to 7860 as it's common for HF Spaces apps
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
