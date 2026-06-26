import streamlit as st
import pandas as pd

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
  "Upload file", accept_multiple_files=True,type=["csv","pdf"]
)
