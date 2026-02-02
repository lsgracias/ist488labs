import streamlit as st

# Pages
lab1 = st.Page("Labs/Lab1.py", title= "Lab 1", icon = ":material/description:")
lab2 = st.Page("Labs/Lab2.py", title= "Lab 2", icon = ":material/description:")
lab3 = st.Page("Labs/Lab3.py", title= "Lab 3", icon = ":material/description:", default= True)

# Navigation
pg = st.navigation([lab1, lab2, lab3])

# Configuration
st.set_page_config(page_title="IST 488 Labs", page_icon=":material/school:")

pg.run()