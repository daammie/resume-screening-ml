import streamlit as st
import pickle
import re
import string
import PyPDF2
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load saved model and vectorizer
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Preset job descriptions for the dropdown
PRESET_JOBS = {
    "Software Engineer": "We are looking for a Software Engineer with experience in Python, data structures, algorithms, and object-oriented design. Responsibilities include writing clean code, debugging, testing, and collaborating with cross-functional teams to build scalable applications.",
    "Data Analyst": "We are seeking a Data Analyst skilled in SQL, Excel, data visualization, and statistical analysis. The role involves collecting, cleaning, and interpreting data to support business decisions and building dashboards and reports.",
    "Mechanical Engineer": "We are hiring a Mechanical Engineer with knowledge of CAD design, thermodynamics, manufacturing processes, and product testing. Responsibilities include designing mechanical components, running simulations, and supporting production teams.",
}


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])


def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


st.title("Resume Fit Checker")
st.caption("Predict candidate-job fit in seconds.")
st.write("Select a role, upload resume, and check fit.")

role_choice = st.selectbox("Select a role", list(PRESET_JOBS.keys()) + ["Other (upload your own job description)"])

jd_text = None

if role_choice == "Other (upload your own job description)":
    jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
    if jd_file:
        jd_text = extract_text(jd_file)
else:
    jd_text = PRESET_JOBS[role_choice]
    with st.expander("View job description"):
        st.write(jd_text)

resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if resume_file and jd_text:
    resume_text = extract_text(resume_file)

    combined = resume_text + " " + jd_text
    cleaned = clean_text(combined)

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.success(f"Fit — confidence: {probability:.2%}")
    else:
        st.error(f"Not Fit — confidence: {(1 - probability):.2%}")
