import streamlit as st
import pandas as pd
import pdfplumber
import nltk
import re

# download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

st.title('Software Requirement Specification for Security-Related Requirements')

st.write('The system is to identify security-related requirements in the SRS')

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

  #before and after comparison, can delete after testing
  with st.expander("Before vs After Comparison"):
    comparison_df = pd.DataFrame({
        "Original Sentence": sentences[:len(cleaned_sentences)],
        "Cleaned Sentence": cleaned_sentences
    })
    st.dataframe(comparison_df)

  st.write(f"{len(cleaned_sentences)} sentences ready for analysis")
else:
  st.warning("Please enter text or upload file(s) to proceed.")




  
