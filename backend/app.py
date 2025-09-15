import os
import uuid
import threading
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import functions from both LSB and DCT modules
from steganography.lsb import encode_text_in_image, decode_image_to_text
from steganography.dct import encode_image_in_image, decode_image_from_image

# --- Configuration ---
ENCODED_DIR = 'encoded'
DECODED_DIR = 'decoded_output'
TEMP_UPLOAD_DIR = 'temp_uploads'
os.makedirs(ENCODED_DIR, exist_ok=True)
os.makedirs(DECODED_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

jobs = {} # In-memory job store

# --- Background Task Runner ---
# This function is the "worker". It receives file paths and does the heavy lifting.
def run_encode_task(job_id, form_data, temp_cover_path, temp_secret_path=None):
    """
    Runs the steganography encoding in a background thread.
    Cleans up its own temporary files.
    """
    try:
        jobs[job_id] = {'status': 'processing', 'progress': 10}
        
        input_type = form_data.get('inputType')
        password = form_data.get('password')

        # Generate descriptive output filename
        base_name = os.path.splitext(os.path.basename(temp_cover_path).split('_', 1)[1])[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if input_type == 'text':
            secret_text = form_data.get('text')
            output_filename = f"{base_name}_lsb_text_{timestamp}.png"
            output_path = os.path.join(ENCODED_DIR, output_filename)
            encode_text_in_image(secret_text, temp_cover_path, output_path, password)

        elif input_type == 'image' and temp_secret_path:
            output_filename = f"{base_name}_dct_image_{timestamp}.png"
            output_path = os.path.join(ENCODED_DIR, output_filename)
            # The function expects a file-like object, so we open the temp path
            with open(temp_secret_path, 'rb') as secret_file:
                encode_image_in_image(secret_file, temp_cover_path, output_path, password)
        else:
            raise ValueError("Invalid input type or missing secret image path.")

        # If successful, update the job status
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['progress'] = 100
        jobs[job_id]['encoded_image_url'] = f'http://localhost:5000/encoded/{output_filename}'

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['message'] = str(e)
        print(f"Error in job {job_id}: {e}") # Log the actual error
    finally:
        # This block ALWAYS runs, ensuring cleanup
        print(f"Cleaning up files for job {job_id}")
        if os.path.exists(temp_cover_path):
            os.remove(temp_cover_path)
        if temp_secret_path and os.path.exists(temp_secret_path):
            os.remove(temp_secret_path)

# --- API Routes ---

@app.route('/login', methods=['POST'])
def login():
    # Your login logic...
    return jsonify({"success": True})

@app.route('/encode', methods=['POST'])
def encode():
    """Receives files, saves them temporarily, and dispatches a background job."""
    if 'coverImage' not in request.files:
        return jsonify({'success': False, 'message': 'Cover image is required.'}), 400

    job_id = str(uuid.uuid4())
    input_type = request.form.get('inputType')
    temp_secret_path = None # Initialize to None

    # 1. Save cover image temporarily, get its path
    cover_image = request.files['coverImage']
    # Save with a unique name based on the job ID
    temp_cover_path = os.path.join(TEMP_UPLOAD_DIR, f"{job_id}_cover_{secure_filename(cover_image.filename)}")
    cover_image.save(temp_cover_path)

    # 2. If it's an image-in-image job, also save the secret image
    if input_type == 'image':
        if 'secretImage' not in request.files:
            return jsonify({'success': False, 'message': 'Secret image is required for this input type.'}), 400
        secret_image = request.files['secretImage']
        temp_secret_path = os.path.join(TEMP_UPLOAD_DIR, f"{job_id}_secret_{secure_filename(secret_image.filename)}")
        secret_image.save(temp_secret_path)
    
    # 3. Start the background thread, passing the file paths
    thread = threading.Thread(
        target=run_encode_task,
        args=(job_id, request.form, temp_cover_path, temp_secret_path)
    )
    thread.start()
    
    # 4. Immediately return the job ID to the frontend
    jobs[job_id] = {'status': 'processing', 'progress': 0}
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files: return jsonify({'success': False, 'message': 'No image provided.'}), 400
    decode_type = request.form.get('decodeType')
    if not decode_type: return jsonify({'success': False, 'message': 'Decode type must be specified.'}), 400

    image_file = request.files['image']
    password = request.form.get('password')
    filename = secure_filename(image_file.filename)
    temp_image_path = os.path.join(TEMP_UPLOAD_DIR, filename)
    image_file.save(temp_image_path)

    try:
        if decode_type == 'text':
            decoded_text = decode_image_to_text(temp_image_path, password)
            return jsonify({'success': True, 'message': 'Text decoded successfully!', 'decoded_type': 'text', 'decoded_content': decoded_text})
        elif decode_type == 'image':
            output_filename = f"decoded_{uuid.uuid4()}.png"
            output_path = os.path.join(DECODED_DIR, output_filename)
            decode_image_from_image(temp_image_path, output_path, password)
            return jsonify({'success': True, 'message': 'Image decoded successfully!', 'decoded_type': 'image', 'decoded_content_url': f'http://localhost:5000/decoded_output/{output_filename}'})
        else:
            return jsonify({'success': False, 'message': 'Invalid decode type specified.'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Decoding failed: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_image_path): os.remove(temp_image_path)

@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({'status': 'error', 'message': 'Job not found.'}), 404
    return jsonify(job)

@app.route('/encoded/<filename>')
def serve_encoded_image(filename):
    return send_file(os.path.join(ENCODED_DIR, filename))

@app.route('/decoded_output/<filename>')
def serve_decoded_output(filename):
    return send_file(os.path.join(DECODED_DIR, filename))

if __name__ == "__main__":
    app.run(debug=True)

