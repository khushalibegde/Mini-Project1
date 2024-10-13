# test.py
#import training
import spacy

# Load the trained NLP model
nlp_model = spacy.load('nlp_model')

def predict_entities(text):
    doc = nlp_model(text)  # Process the custom text
    for ent in doc.ents:
        print(f'{ent.label_.upper():{30}}- {ent.text}')

if __name__ == "__main__":
    # Example input text
    custom_text = "KHUSHALI BEGDE I am currently pursuing a Bachelor’s degree in Computer Science from Ramdeobaba College of Engineering & Management, Nagpur. As a dedicated and hardworking student, I am passionate about applying my knowledge to real-world challenges. I am detail-oriented, organized, and continuously strive to enhance my technical abilities. I am eager to take on new challenges and contribute meaningfully to any team or project. CONTACT EXPERIENCE 9730986147 khushalibegde18@gmail.com Resume Parsing and Job Matching Application: begdekm@rknec.edu July 2024 - Present Developed a Flask-based application that automates EDUCATION the parsing of resumes (PDF, DOCX) to extract skills and personal information using NLP BACHELOR'S DEGREE IN COMPUTER SCIENCE techniques. Implemented a skill matching algorithm that Shri Ramdeobaba College of Engineering and Management compares extracted skills with job descriptions, 2022-2026 providing personalized job recommendations based on skill relevance. TECHNICAL SKILLS Online Food Ordering Website: April 2024 - Present Programming Languages: Python, Java, C, R, C++ Designed and developed a responsive website allowing users to browse menus and get to know Web Development: HTML, CSS, about the website. JavaScript, ReactJS, Flask Database Management: SQL (Oracle SOCIAL PROFILES SQL Developer) Machine Learning www.linkedin.com/in/khushali-begde-5b19bb276 Data Science www.codechef.com/users/khushalibegde www.hackerrank.com/profile/khushalibegde18 www.leetcode.com/u/khushalibegde/ SOFT SKILLS VOLUNTEER EXPERIENCE Organized Hardworking Make A Difference(MAD): Academic Volunteer NSS volunteer (2023-2024) Attention to details Academic mentor for First Year Student (III - IV Sem) Time management Adaptability Problem Solving HOBBIES & LANGUAGES Teamwork English, Hindi, Marathi Drawing, Reading, Cleaning, Playing"

    predict_entities(custom_text)
