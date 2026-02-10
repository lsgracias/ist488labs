import streamlit as st

# Pages
lab1 = st.Page("Labs/Lab1.py", title= "Lab 1 - Document Q & A", icon = ":material/description:")
lab2 = st.Page("Labs/Lab2.py", title= "Lab 2 - Document Summarizer", icon = ":material/description:")
lab3 = st.Page("Labs/Lab3.py", title= "Lab 3 - Chatbot", icon = ":material/description:")
lab4 = st.Page("Labs/Lab4.py", title= "Lab 4 - RAG Chatbot", icon = ":material/description:", default= True)
somebs = st.Page("Labs/lab4_losingmymind.py", title= "somebs", icon = ":material/description:")
# Navigation
pg = st.navigation([lab1, lab2, lab3, lab4, somebs])

# Configuration
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

pg.run()