from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import os
import re

# Initialize Blueprint
candidate_bp = Blueprint('candidate', __name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png'}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

@candidate_bp.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # TODO: Extract text, phone number, email, etc., from the uploaded resume

        # Save resume in MongoDB
        candidates_collection.insert_one({
            "name": request.form['name'],
            "email": request.form['email'],
            "resume_text": "Extracted resume text here...",
            "skills": ["Skill1", "Skill2"],  # Extracted skills
            "uploaded_at": datetime.now()
        })

        return jsonify({"message": "Resume uploaded successfully"}), 200
    else:
        return jsonify({"error": "Unsupported file type"}), 400
