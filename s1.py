import streamlit as st
import pdfplumber
import docx
import os
import joblib
import re

# Load the model and vectorizer
clf = joblib.load('resume_classifier_model.pkl')  # Classification model
word_vectorizer = joblib.load('word_vectorizer.pkl')  # Text vectorizer

# Define category names and recommendations
category_names = [
    "Java Developer", "Database", "HR", "Data Science", "Advocate",
    "DotNet Developer", "Hadoop", "DevOps Engineer", "Automation Testing",
    "Testing", "Civil Engineer", "Business Analyst", "SAP Developer",
    "Health and fitness", "Python Developer", "Arts", "Electrical Engineering",
    "Sales", "Network Security Engineer", "Mechanical Engineer", "Web Designing",
    "ETL Developer", "Blockchain", "Operations Manager", "PMO"
]

job_recommendations = {
    "Java Developer": "Apply for Java Developer roles in companies focusing on backend development.",
    "Database": "Look for opportunities as a Database Administrator or Data Architect.",
    "HR": "Explore positions in Human Resources or Talent Management.",
    "Data Science": "Consider roles such as Data Scientist, Machine Learning Engineer, or Analyst.",
    "Advocate": "Search for legal firms or organizations hiring advocates or legal consultants.",
    "DotNet Developer": "Focus on .NET Developer roles in software development companies.",
    "Hadoop": "Look for roles in Big Data Engineering or Hadoop Administration.",
    "DevOps Engineer": "Explore DevOps or Cloud Engineer positions in IT companies.",
    "Automation Testing": "Search for QA Automation roles in testing teams.",
    "Testing": "Focus on roles in manual or automated software testing.",
    "Civil Engineer": "Look for Civil Engineering roles in construction or infrastructure firms.",
    "Business Analyst": "Explore positions as a Business Analyst or Consultant.",
    "SAP Developer": "Look for SAP Development or SAP Functional Consultant roles.",
    "Health and fitness": "Consider roles as a Fitness Trainer or Health Consultant.",
    "Python Developer": "Apply for Python Developer roles in AI, web development, or backend projects.",
    "Arts": "Look for creative roles in graphic design, animation, or arts-related fields.",
    "Electrical Engineering": "Explore roles in power systems, electronics, or electrical design.",
    "Sales": "Search for opportunities as a Sales Manager or Executive.",
    "Network Security Engineer": "Focus on Cybersecurity roles in network security.",
    "Mechanical Engineer": "Look for Mechanical Engineering positions in manufacturing or design.",
    "Web Designing": "Apply for Web Designer or UI/UX roles.",
    "ETL Developer": "Search for ETL Development roles in data warehousing projects.",
    "Blockchain": "Explore opportunities in Blockchain Development or Crypto Analysis.",
    "Operations Manager": "Focus on Operations Manager roles in business management.",
    "PMO": "Look for roles in Project Management Offices or as a Project Manager."
}

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract text from a Word document
def extract_text_from_docx(file):
    text = ""
    doc = docx.Document(file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract specific information using regular expressions
def extract_resume_details(text):
    details = {}
    lines = text.split('\n')
    if lines:
        details['Name'] = lines[0].strip()

    phone_match = re.search(r'\b\d{10}\b|\b(\+\d{1,2}\s)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    details['Phone'] = phone_match.group(0) if phone_match else "Not found"

    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    details['Email'] = email_match.group(0) if email_match else "Not found"

    education_patterns = [
        r'(?i)(B\.?(Tech|E)|M\.?(Tech|CA|BA)|BSc|MSc|PhD|Bachelor|Master|Doctorate).*?(from\s+)?(\b[A-Z][a-zA-Z.&\s]+(University|Institute|College|School)\b)',
        r'(?i)(\b[A-Z][a-zA-Z.&\s]+(University|Institute|College|School)\b).*?(B\.?(Tech|E)|M\.?(Tech|CA|BA)|BSc|MSc|PhD|Bachelor|Master|Doctorate)?'
    ]
    qualifications = []
    for pattern in education_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            qualifications.append(' '.join(match).strip())
    details['Education'] = [q for q in qualifications if q.strip()] if qualifications else ["Not found"]

    skills_keywords = ['skills', 'technologies', 'proficient in', 'expertise']
    skills = [line for line in lines if any(keyword in line.lower() for keyword in skills_keywords)]
    details['Skills'] = skills if skills else ["Not found"]

    return details

# Streamlit app
st.set_page_config(page_title="Resume Screening", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: black;
        color: orange;
    }
    .stButton>button {
        background-color: black;
        color: yellow;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("HIREGENIE")
st.write("Upload a .pdf or .docx file, and the model will predict the job category and provide recommendations.")

uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.pdf'):
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.docx'):
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or Word document.")
        resume_text = ""

    if not resume_text.strip():
        st.error("The uploaded file appears to be empty. Please try another file.")
    else:
        details = extract_resume_details(resume_text)

        st.subheader("Extracted Resume Details:")
        for key, value in details.items():
            st.write(f"**{key}:** {', '.join(value) if isinstance(value, list) else value}")

        transformed_text = word_vectorizer.transform([resume_text])
        prediction = clf.predict(transformed_text)
        predicted_category = category_names[prediction[0]]

        st.success(f"Predicted Category: {predicted_category}")

        st.subheader("Recommended Job Role:")
        st.write(job_recommendations[predicted_category])
