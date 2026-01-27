import streamlit as st

# Pages
lab1 = st.Page("Lab1.py", title= "Lab 1", icon = ":material/description:")
lab2 = st.Page("Lab2.py", title= "Lab 2", icon = ":material/summarize:", default= True)

# Navigation
pg = st.navigation([lab1, lab2])

# Configuration
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

pg.run()