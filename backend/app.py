from flask import Flask, render_template, request, redirect #to make app
import os   
import pandas as pd #to read csv
import re   #for cleaning
from sklearn.feature_extraction.text import CountVectorizer #convert to vector
from sklearn.metrics.pairwise import cosine_similarity  #for comparision
from werkzeug.utils import secure_filename
from pdfplumber import open as open_pdf
from docx2txt import process as docx_process
from PIL import Image   #to read and upload image
import pytesseract  #to extract text form image
from models.skills_data import skills
import spacy    
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import base64 
import io
from flask import send_file
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

#import axios from 'axios';

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Allow only your frontend


client = MongoClient('mongodb+srv://khushalibegde18:y1ESj3rHlikmgxu6@cluster0.9q3rt.mongodb.net/')
db = client['resume_analyzer']  
candidates_collection = db['candidates']  
jobs_collection = db['jobs']
details_db = client['userid_pass'] 
users_collection = details_db['users']
company_collection = details_db['company']

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

nlp = spacy.load("en_core_web_sm")


# Create the uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load CSV data into a pandas DataFrame
csv_path = 'dataa/Jobdataset.csv'  
data = pd.read_csv(csv_path)

#skills_path = 'dataa/TechnologySkills.csv'
#skills_data = pd.read_csv(skills_path)


#get file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

custom_stopwords = ["the", "a", "of", "and", "in", "for", "with", "I", "you", "he", "she", "we", "they"]
#cleaning the text
def clean_text(text):
    cleaned_text = re.sub(r'\d{10,}', '', text)
    cleaned_text = re.sub(r'http\S+\s*', ' ', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'RT|cc', ' ', cleaned_text)  # Remove 'RT' and 'cc'
    cleaned_text = re.sub(r'#\S+', '', cleaned_text)  # Remove hashtags
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)  # Replace mentions with spaces
    cleaned_text = re.sub(r"[,./?\":;+=\[\](){}]", "", cleaned_text)  # Remove special characters
    cleaned_text = re.sub(r'[^\x00-\x7f]', ' ', cleaned_text)  # Remove non-ASCII characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra whitespace
    cleaned_text = cleaned_text.lower()    

    return cleaned_text


def calculate_similarity(cleaned_text, resume_text, skill_list):
    # Extract skills present in both cleaned_text and resume_text
    cleaned_skills = set([skill for skill in skill_list if skill.lower() in cleaned_text.lower()])
    resume_skills = set([skill for skill in skill_list if skill.lower() in resume_text.lower()])
    
    # Convert the skills to a string for vectorization
    cleaned_text_skills = ' '.join(cleaned_skills)
    resume_text_skills = ' '.join(resume_skills)
    
    if cleaned_text_skills and resume_text_skills:
        # Vectorize the skill strings
        vectorizer = CountVectorizer().fit_transform([cleaned_text_skills, resume_text_skills])
        vectors = vectorizer.toarray()
        # Calculate cosine similarity
        similarity = cosine_similarity(vectors)[0, 1]
        return similarity
    else:
        return 0  # Return 0 if no skills match


def exact_match_skills(cleaned_text, skills_list):
    matched_skills = set()
    for skill in skills_list:
        if f"{skill.lower()}" in cleaned_text:
            matched_skills.add(skill)
    return matched_skills

def tokenize_and_match_skills(cleaned_text, job_skills):
    resume_doc = nlp(cleaned_text)
    matched_skills = set()
    for skill in job_skills:
        skill_doc = nlp(skill)
        for token in resume_doc:
            if token.similarity(skill_doc) > 0.75:
                matched_skills.add(skill)
                break
    return matched_skills

def extract_name(text):
    # Split text into lines and take the first few lines (assuming name is near the top)
    lines = text.splitlines()
    name_candidates = []
    
    # Loop through the first few lines to find a probable name
    for line in lines[:10]:  # Assuming the name will be in the first 10 lines
        # Check if the line has at least two words and both are capitalized
        words = line.split()
        if len(words) > 1 and all(word[0].isupper() for word in words):
            name_candidates.append(line)
    
    if name_candidates:
        return name_candidates[0]  # Return the first candidate as the name
    return None

def get_next_id():
    last_candidate = candidates_collection.find_one(
        sort=[("_id", -1)] 
    )
    return (last_candidate['_id'] + 1) if last_candidate else 1

# Function to extract phone numbers
def extract_phone_number(text):
    phone_number = re.findall(r'\b\d{10,}\b', text)  # Match 10+ digit numbers
    if phone_number:
        return phone_number[0]  # Return the first phone number found
    return None

# Function to extract email addresses
def extract_email(text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email:
        return email[0]  # Return the first email found
    return None

from flask import send_file, abort

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return abort(404, description="File not found")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    if file_ext in {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)


        with open(filepath, 'rb') as f:
            encoded_file = base64.b64encode(f.read()).decode('utf-8')
        # Determine file type and convert to text
        if file_ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_ext == 'pdf':
            with open_pdf(filepath) as pdf:
                pages = pdf.pages
                text = ''
                for page in pages:
                    text += page.extract_text()
        elif file_ext == 'docx':
            text = docx_process(filepath)
        elif file_ext in {'png', 'jpg', 'jpeg'}:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)


# Extract phone number, email, and name
        phone_number = extract_phone_number(text)
        email = extract_email(text)
        name = extract_name(text)
        # Clean the extracted text
        cleaned_text = clean_text(text)
        
        
        
        # Extract skills using both exact match and token similarity-based skill matching
        exact_matches = exact_match_skills(cleaned_text, skills)
        token_matches = tokenize_and_match_skills(cleaned_text, skills)
        
        # Combine both sets of matches
        all_matches = sorted(list(exact_matches.union(token_matches)))
        candidate_id = get_next_id()
        dictory = {
            "_id": candidate_id,
            "name": name,
            "phone": phone_number,
            "email": email,
            "skills": all_matches,
            "filename": filename,
            "file_content": encoded_file,  # Store the file as Base64
            "file_extension": file_ext  
            }
        
        candidates_collection.insert_one(dictory)
        # Render the result page with name, phone, email, and extracted skills
        #return render_template('skills.html', skills=all_matches, name=name, phone=phone_number, email=email,candidate=dictory)
        return jsonify({
            "id": candidate_id,
            "name": name,
            "phone": phone_number,
            "email": email,
            "skills": all_matches
        })

    else:
        
        return jsonify({"error": "Unsupported file type"}), 400

def fetch_data(collection):
    # Fetch all documents from the collection
    data = collection.find()
    return data




def cv_skills():
    cleaned_text = request.form['text'].lower()
    exact_matches = exact_match_skills(cleaned_text, skills)
    token_matches = tokenize_and_match_skills(cleaned_text, skills)
    all_matches = sorted(list(exact_matches.union(token_matches)))
    cv_skills = ' '.join(all_matches)
    return cv_skills

@app.route('/clean_text', methods=['POST'])
def clean_text_route():
    text = request.form['text']
    cleaned_text = clean_text(text)
    return jsonify({"cleaned_text": cleaned_text})



@app.route('/skills', methods=['POST'])
def cv_skill_match():
    cleaned_text = request.form['text'].lower()

    # Perform exact skill matching
    exact_matches = exact_match_skills(cleaned_text, skills)
    
    # Perform token similarity-based skill matching
    token_matches = tokenize_and_match_skills(cleaned_text, skills)
    
    # Combine both sets of matches
    all_matches = sorted(list(exact_matches.union(token_matches)))
    
    return jsonify({"skills": all_matches})


def job_skill_match():
    for index, row in data.iterrows():
        resume_text = clean_text(row['Skills']).lower() 

    # Perform exact skill matching
    exact_matches = exact_match_skills(resume_text, skills)
    
    # Perform token similarity-based skill matching
    token_matches = tokenize_and_match_skills(resume_text, skills)
    # Combine both sets of matches
    all_matches = sorted(list(exact_matches.union(token_matches)))
    job_skills = ' '.join(all_matches)
    
    return all_matches,job_skills

@app.route('/compare_text', methods=['POST'])
def compare_text():
    if 'text' not in request.json:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    cleaned_text = request.json['text'].lower() # Get the cleaned resume text from the form
    matches = []

    for index, row in data.iterrows():
        resume_text = clean_text(row['Skills']).lower()  # Clean and lowercase the resume text from the dataset
        similarity = calculate_similarity(cleaned_text, resume_text, skills)

        # Find matched and unmatched skills
        matched_skills = set()
        unmatched_skills = set()

        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in cleaned_text and skill_lower in resume_text:
                matched_skills.add(skill)
            elif skill_lower in resume_text and skill_lower not in cleaned_text:
                unmatched_skills.add(skill)

        if similarity >= 0.01 and matched_skills:  # Only consider matches with similarity >= 1% and skills found
            matches.append({
                "category": row['Job'],
                "percentage": round(similarity * 100, 2),
                "matched_skills": ', '.join(matched_skills),
                "unmatched_skills": ', '.join(unmatched_skills)
            })

    # Remove duplicate categories with lower percentage matches
    unique_matches = {}
    for match in matches:
        category = match['category']
        if category not in unique_matches or unique_matches[category]['percentage'] < match['percentage']:
            unique_matches[category] = match

    # Prepare the final matches for display
    sorted_matches = sorted(unique_matches.values(), key=lambda x: x['percentage'], reverse=True)

    return jsonify({"matches": sorted_matches})


#to get candidates for job description
@app.route('/get_candidates', methods=['POST'])
def get_candidates():
    data = request.json  
    job_description = data.get('job_description')

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    cleaned_description = clean_text(job_description)

    data_list = candidates_collection.find({})
    candidates = []

    # Iterate through candidates and calculate similarity with job description
    for candidate in data_list:
        candidate_name = candidate.get("name")
        resume_text = " ".join(candidate.get("skills", []))
        similarity_score = calculate_similarity(cleaned_description, resume_text, skills) * 100

        # Safely access 'filename', 'file_content', and 'file_extension'
        candidates.append({
            "name": candidate_name,
            "email": candidate.get("email"),
            "phone": candidate.get("phone"),
            "skills": candidate.get("skills", []),
            "similarity_score": round(similarity_score, 2),
            'filename': candidate.get('filename', 'N/A'),  # Use default 'N/A' if missing
            'file_content': candidate.get('file_content', ''),  # Default to empty string
            'file_extension': candidate.get('file_extension', 'unknown')  # Default 'unknown'
        })

    # Sort candidates by similarity score in descending order
    sorted_candidates = sorted(candidates, key=lambda x: x['similarity_score'], reverse=True)

    return jsonify({
        "job_description": cleaned_description,
        "candidates": sorted_candidates
    })

@app.route('/companysignup', methods=['POST'])
def companysignup():
    data = request.json

    # Check if the user already exists in the 'users' collection
    if company_collection.find_one({"email": data['email']}):
        return jsonify({"message": "User already exists"}), 400

    # Hash the password for security
    hashed_password = generate_password_hash(data['password'])

    # Store the user details in the database
    user_data = {
        "companyName": data['companyName'],
        "email": data['email'],
        "contact": data['contact'],
        "location": data['location'],
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }
    company_collection.insert_one(user_data)

    return jsonify({"message": "Sign up successful"}), 201

@app.route('/candidatesignup', methods=['POST'])
def candidatesignup():
    data = request.json

    # Check if the user already exists in the 'users' collection
    if users_collection.find_one({"email": data['email']}):
        return jsonify({"message": "User already exists"}), 400

    # Hash the password for security
    hashed_password = generate_password_hash(data['password'])

    # Store the user details in the database
    user_data = {
        "name": data['name'],
        "phone": data['phone'],
        "email": data['email'],
        "password": hashed_password,
        "education": data['education'],
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user_data)

    return jsonify({"message": "Sign up successful"}), 201

# --- LOGIN ROUTE ---
@app.route('/candidatelogin', methods=['POST'])
def candidatelogin():
    data = request.json

    # Retrieve user by email
    user = users_collection.find_one({"email": data['email']})

    if user:
        print("Stored Hash:", user['password'])
        print("Entered Password:", data['password'])
        # Check if password is valid
        if check_password_hash(user['password'], data['password']):
            users_collection.insert_one({
                "email": data['email'],
                "login_time": datetime.utcnow()
            })
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"message": "Invalid password"}), 401
    return jsonify({"message": "User not found"}), 404

@app.route('/companylogin', methods=['POST'])
def companylogin():
    data = request.json

    # Retrieve user by email
    user = company_collection.find_one({"email": data['email']})

    if user:
        print("Stored Hash:", user['password'])
        print("Entered Password:", data['password'])
        # Check if password is valid
        if check_password_hash(user['password'], data['password']):
            company_collection.insert_one({
                "email": data['email'],
                "login_time": datetime.utcnow()
            })
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"message": "Invalid password"}), 401
    return jsonify({"message": "User not found"}), 404


#get resume uploaded 
@app.route('/candidates/<int:candidate_id>', methods=['GET'])
def get_candidate_skills(candidate_id):
    candidate = candidates_collection.find_one({"_id": candidate_id})
    if candidate:
        # Return the candidate's details in JSON format
        return jsonify({
            "name": candidate.get("name", "N/A"),
            "phone": candidate.get("phone", "N/A"),
            "email": candidate.get("email", "N/A"),
            "skills": candidate.get("skills", []),
            "file_content": candidate.get("file_content", "")  # Assuming it's base64
        })
    else:
        return jsonify({"error": "Candidate not found"}), 404




if __name__ == '__main__':
    app.run(debug=True)