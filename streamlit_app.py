import streamlit as st
import pandas as pd
from io import StringIO #remove?
import pdfplumber
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re

#spacy model
nlp=spacy.load("en_core_web_sm")

st.title('Software Requirement Specification for Security-Related Requirements')

st.info('The system is to identify security-related requirements in the SRS')

st.info('Text / File Upload')
#user input text
st.write('Text')
txt=st.text_area(
  "Text to identify the requirements"
)

st.write(f" {len(txt)} characters | {len(txt.split())} words")
#user file upload
st.write('File upload')
#support csv and pdf, multiple files
uploaded_file=st.file_uploader(
  "Choose file(s)", accept_multiple_files=True,type=["csv","txt"]
)

#for text input
all_text=[]
if txt.strip():
  all_text.append(txt)
  
#preprocess the files
if uploaded_file is not None:
  
  for uploaded_files in uploaded_file:
    st.write(f"{uploaded_file.name}")
    #if text
    if uploaded_file.name.endswith(".csv"):
        df=pd.read_csv(uploaded_file)
        string_data=df.to_string(index=False)
  
    #if csv file
    elif uploaded_file.name.endswith(".txt"):
      string_data=uploaded_file.read().decode("utf-8",errors="ignore")

    all_text.append(string_data)
    st.success(f"Loaded: {uploaded_file.name}")

if all_text:
  combined_text="\n\n".join(all_text)
  #view text
  with st.expander("Raw Text Preview"):
      st.text_area("Raw", combined_text[:1000] + "...", height=200)

  #sentence tokenization
  st.write("1. Sentence tokenization...")
  doc = nlp(combined_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    st.write(f"**{len(sentences)} sentences extracted**")
    with st.expander("View Tokenized Sentences"):
        for i, sent in enumerate(sentences, 1):
            st.write(f"`{i}.` {sent}")
  #noise reduction
  st.write("2. Noise reduction...")
  def clean_sentence(sentence):
        # for lowercase
        sentence = sentence.lower()
        # remove special characters, numbers, extra whitespace
        sentence = re.sub(r"[^a-z\s]", "", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        # remove stop words
        tokens = sentence.split()
        tokens = [word for word in tokens if word not in STOP_WORDS]
        return " ".join(tokens)

    cleaned_sentences = [clean_sentence(s) for s in sentences]
    # filter out empty sentences after cleaning
    cleaned_sentences = [s for s in cleaned_sentences if s]

    st.write(f"**{len(cleaned_sentences)} sentences after noise reduction**")
    with st.expander("View Cleaned Sentences"):
        for i, sent in enumerate(cleaned_sentences, 1):
            st.write(f"`{i}.` {sent}")

  #before and after comparison, can delete after testing
  st.write("Before vs After Comparison")
    comparison_df = pd.DataFrame({
        "Original Sentence": sentences[:len(cleaned_sentences)],
        "Cleaned Sentence": cleaned_sentences
    })
    st.dataframe(comparison_df)

  #ready for analysis
  st.write("## ✅ Ready for Requirement Analysis")
  st.write(f"- **Total sentences:** {len(sentences)}")
  st.write(f"- **After cleaning:** {len(cleaned_sentences)}")
  st.write(f"- **Total words (cleaned):** {sum(len(s.split()) for s in cleaned_sentences)}")
else:
  st.warning("Please enter text or upload file(s) to proceed.")




  
