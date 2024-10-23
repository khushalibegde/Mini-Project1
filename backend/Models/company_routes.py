from flask import Blueprint, request, jsonify
from pymongo import MongoClient

# Initialize Blueprint
company_bp = Blueprint('company', __name__)

@company_bp.route('/post_job', methods=['POST'])
def post_job():
    data = request.json
    jobs_collection.insert_one({
        "company_name": data['company_name'],
        "job_title": data['job_title'],
        "description": data['description'],
        "required_skills": data['required_skills'],
        "posted_at": datetime.now()
    })
    return jsonify({"message": "Job posted successfully"}), 200

@company_bp.route('/get_matching_resumes/<job_id>', methods=['GET'])
def get_matching_resumes(job_id):
    job = jobs_collection.find_one({"_id": ObjectId(job_id)})
    required_skills = job['required_skills']

    matching_candidates = candidates_collection.find({
        "skills": {"$in": required_skills}
    })

    result = [
        {
            "name": candidate['name'],
            "email": candidate['email'],
            "skills": candidate['skills']
        }
        for candidate in matching_candidates
    ]
    return jsonify(result), 200
