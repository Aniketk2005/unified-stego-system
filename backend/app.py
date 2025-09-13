# backend/app.py

import os
import uuid
import threading
from datetime import datetime # Import the datetime module
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from steganography.lsb import encode_text_in_image, decode_image_to_text

# --- Configuration ---
ENCODED_DIR = 'encoded'
TEMP_UPLOAD_DIR = 'temp_uploads'
os.makedirs(ENCODED_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

jobs = {}

def run_encoding_task(job_id, message, temp_path, output_path, password):
    """This function will run in a separate thread."""
    try:
        jobs[job_id] = {'status': 'processing', 'progress': 10}
        encode_text_in_image(message, temp_path, output_path, password)
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['encoded_image_url'] = f'http://localhost:5000/encoded/{os.path.basename(output_path)}'
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Routes ---

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username, password = data.get('username'), data.get('password')
    USERS = {"user1": "password123"}
    if username in USERS and USERS[username] == password:
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/encode', methods=['POST'])
def encode():
    if 'text' not in request.form or 'coverImage' not in request.files:
        return jsonify({'success': False, 'message': 'Missing text or cover image.'}), 400

    message_content = request.form['text']
    cover_image_file = request.files['coverImage']
    password = request.form.get('password')
    # Your frontend sends 'lsb_image' or 'dct_image', but we'll get it from the form
    # Note: We need to update the frontend to send this again. Let's assume 'lsb_image' for now.
    encoding_method = request.form.get('encoding', 'lsb_text_to_image') 

    cover_filename = secure_filename(cover_image_file.filename)
    temp_cover_path = os.path.join(TEMP_UPLOAD_DIR, cover_filename)
    cover_image_file.save(temp_cover_path)

    # --- NEW DESCRIPTIVE NAMING SCHEME ---
    # 1. Get the original filename without its extension
    original_filename_base = os.path.splitext(cover_filename)[0]
    
    # 2. Get the current timestamp in a file-friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3. Combine the parts to create the new name
    output_filename = f"{original_filename_base}_{encoding_method}_{timestamp}.png"
    # --- END OF CHANGES ---

    output_path = os.path.join(ENCODED_DIR, output_filename)

    job_id = str(uuid.uuid4())
    
    thread = threading.Thread(
        target=run_encoding_task,
        args=(job_id, message_content, temp_cover_path, output_path, password)
    )
    thread.start()
    
    jobs[job_id] = {'status': 'processing', 'progress': 0}
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({'status': 'error', 'message': 'Job not found.'}), 404
    return jsonify(job)

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided.'}), 400

    image_file = request.files['image']
    password = request.form.get('password')
    filename = secure_filename(image_file.filename)
    temp_image_path = os.path.join(TEMP_UPLOAD_DIR, filename)
    image_file.save(temp_image_path)

    try:
        decoded_text = decode_image_to_text(temp_image_path, password)
        return jsonify({
            'success': True,
            'decoded_type': 'text',
            'decoded_content': decoded_text,
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.route('/encoded/<filename>')
def serve_encoded_image(filename):
    return send_file(os.path.join(ENCODED_DIR, filename))

if __name__ == "__main__":
    app.run(debug=True)