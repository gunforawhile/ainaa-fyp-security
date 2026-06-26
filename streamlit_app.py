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

st.write('Text')
txt=st.text_area(
  "Text to identify the requirements"
)

st.write(f" {len(txt)} characters | {len(txt.split())} words")

st.write('File upload')
#support csv and pdf, multiple files
uploaded_file=st.file_uploader(
  "Choose file(s)", accept_multiple_files=True,type=["csv","txt"]
)

all_text=[]

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

  combined_text="\n\n".join(all_text)
  #view text
   with st.expander("Raw Text Preview"):
        st.text_area("Raw", combined_text[:1000] + "...", height=200)

  #sentence tokenization
  
