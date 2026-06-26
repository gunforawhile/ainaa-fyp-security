import streamlit as st

st.title('Software Requirement Specification for Security-Related Requirements')

st.info('The system is to identify security-related requirements in the SRS')

txt=st.text_area(
  "Text to identify requirements"
)

st.write(f" {len(txt)} characters are inserted.")
