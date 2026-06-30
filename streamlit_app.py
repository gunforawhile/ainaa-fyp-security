import streamlit as st
import pandas as pd
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import torch

# download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

#MODEL--------------------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    # Phase I: Binary Classification (Functional vs Security)
    phase1_classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli"
    )
    # Phase II: CIA Triad Classification
    phase2_classifier = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli"
    )
    return phase1_classifier, phase2_classifier

# Phase 1
def phase1_classify(sentence, classifier):
    candidate_labels = ["security requirement", "functional requirement"]
    
    result = classifier(sentence, candidate_labels)
    label = result['labels'][0]
    score = result['scores'][0]
    
    if label == "security requirement":
        return "Security", score
    else:
        return "Functional", score

# Phase 2
def phase2_classify(sentence, classifier):
    candidate_labels = [
        "confidentiality - data access control and privacy",
        "integrity - data accuracy and prevention of unauthorized modification",
        "availability - system and data accessibility and uptime"
    ]
    result = classifier(sentence, candidate_labels)
    label_map = {
        "confidentiality - data access control and privacy": "Confidentiality",
        "integrity - data accuracy and prevention of unauthorized modification": "Integrity",
        "availability - system and data accessibility and uptime": "Availability"
    }
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    return label_map[top_label], top_score

# Two-Phase Classification
def classify_requirements(sentences):
    phase1_classifier, phase2_classifier = load_models()
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, sentence in enumerate(sentences):
        status.write(f"Analysing sentence {i+1} of {len(sentences)}...")
        progress.progress((i + 1) / len(sentences))

        if len(sentence.split()) < 3:
            continue

        # Phase I
        phase1_label, phase1_score = phase1_classify(sentence, phase1_classifier)

        # Phase II (only for security requirements)
        if phase1_label == "Security":
            phase2_label, phase2_score = phase2_classify(sentence, phase2_classifier)
        else:
            phase2_label = "N/A"
            phase2_score = None

        results.append({
            "Sentence": sentence,
            "Phase I (Type)": phase1_label,
            "Phase I Confidence": f"{phase1_score:.0%}",
            "Phase II (CIA)": phase2_label,
            "Phase II Confidence": f"{phase2_score:.0%}" if phase2_score else "N/A"
        })

    progress.empty()
    status.empty()
    return pd.DataFrame(results)

#UI part------------------------------------------------------------------------------------------
st.title('Software Requirement Specification for Security-Related Requirements')

st.write('The system is to identify security-related requirements in the SRS')

#TEXT/FILE UPLOAD SECTION---------------------------------------------------------------
st.info('Text / SRS File Upload')

#user input text
st.write('Text Input')
txt=st.text_area(
  "Text to identify the requirements"
)
st.write(f" {len(txt)} characters | {len(txt.split())} words")

#user file upload
st.write('File Upload')
uploaded_file=st.file_uploader(
  "Choose file(s)", accept_multiple_files=True,type=["csv","txt"]
)

#for text input
all_text=[]
if txt.strip():
  all_text.append(txt)
  
#preprocess the files
if uploaded_file:
  
  for uploaded_files in uploaded_file:
    st.write(f"{uploaded_files.name}")
    #if text
    if uploaded_files.name.endswith(".csv"):
        df=pd.read_csv(uploaded_files)
        string_data=df.to_string(index=False)
  
    #if csv file
    elif uploaded_files.name.endswith(".txt"):
      string_data=uploaded_files.read().decode("utf-8",errors="ignore")

    all_text.append(string_data)
    st.success(f"Loaded: {uploaded_files.name}")

if all_text:
  combined_text="\n\n".join(all_text)
  #view text
  with st.expander("Raw Text Preview"):
      st.text_area("Raw", combined_text[:1000] + "...", height=150)

  #sentence tokenization
  sentences = sent_tokenize(combined_text)
      
  #noise reduction
  def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z\s]", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    tokens = sentence.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

  cleaned_sentences = [clean_sentence(s) for s in sentences]
  cleaned_sentences = [s for s in cleaned_sentences if s]

  #before and after sentences comparison
  with st.expander("Before vs After Comparison"):
    comparison_df = pd.DataFrame({
        "Original Sentence": sentences[:len(cleaned_sentences)],
        "Cleaned Sentence": cleaned_sentences
    })
    st.dataframe(comparison_df)
      
 st.write(f"{len(cleaned_sentences)} sentences ready for analysis")

 #classification SECTION---------------------------------------------------------------
 st.info("Classification of Requirements")

 if st.button("Run"):
    with st.spinner("Loading (It may take a few minutes...)"):
        results_df = classify_requirements(sentences)
else:
  st.warning("Please enter text or upload file(s) to proceed.")



  
