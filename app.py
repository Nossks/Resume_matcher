import streamlit as st
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import fitz 

with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def get_recommendation(resume):
    resume = resume.lower()
    resume = re.sub(r"[^a-z0-9\s]", "", resume)
    words = resume.split()
    word_vec = [model.wv[word] for word in words if word in model.wv]
    final_res = np.mean(word_vec, axis=0) if word_vec else np.zeros(model.vector_size)

    recommendation = []

    for idx, emb in enumerate(dataset["embeddings"]):
        recommendation.append((cosine_similarity([final_res], [emb])[0][0], int(idx)))

    recommendation.sort(reverse=True, key=lambda x: x[0])
    seen_titles = set()
    results = []

    for it in recommendation[:100]:
        title = dataset.iloc[it[1]]["title"]
        score = it[0]
        if len(seen_titles) == 5:
            break
        if title not in seen_titles and score >= .80:
            seen_titles.add(title)
            results.append((title, score))

    return results

st.title("ResumeMatcher")
st.write("Welcome to the Resume Matcher! Please upload or paste your resume below:")


uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
resume_text = st.text_area("Or paste your resume text here:", height=300)

if st.button("Find Matching Jobs"):
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
    if resume_text:
        results = get_recommendation(resume_text)
        if not results:
            st.write("No matching jobs found.")
        else:
            for title, score in results:
                st.write(f"**{title}** - Similarity Score: {score:.2f}")
    else:
        st.write("Please upload a PDF or enter text.")

