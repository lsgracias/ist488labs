import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Lab 6b - Movie Recommendation Chatbot")
st.caption("Powered by LangChain · GPT-4o Mini")

# LLM init (Part A) 
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
)

#  Session state 
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = ""

# Sidebar controls (Part B) 
st.sidebar.header("Customize Your Picks")

genre = st.sidebar.selectbox(
    "Genre",
    ["Action", "Comedy", "Horror", "Drama", "Sci-Fi", "Thriller", "Romance"],
)

mood = st.sidebar.selectbox(
    "Mood",
    ["Excited", "Happy", "Sad", "Bored", "Scared", "Romantic", "Curious", "Tense", "Melancholy"],
)

persona = st.sidebar.selectbox(
    "Recommender Persona",
    ["Film Critic", "Casual Friend", "Movie Journalist"],
)

# Chain 1 — Recommendation chain (Part B) 
rec_template = PromptTemplate(
    input_variables=["genre", "mood", "persona"],
    template=(
        "You are a {persona}. A user is feeling {mood} and wants to watch a {genre} movie. "
        "Recommend exactly 3 movies that fit both the genre and the mood. "
        "For each movie, give the title, year, a one-sentence synopsis, and a brief reason "
        "why it matches the user's current mood. "
        "Match the tone and writing style of a {persona} throughout your response."
    ),
)

rec_chain = rec_template | llm | StrOutputParser()

# Recommendation button 
if st.button(" Get Recommendations", type="primary"):
    with st.spinner("Finding the perfect movies for you…"):
        result = rec_chain.invoke({"genre": genre, "mood": mood, "persona": persona})
        st.session_state.last_recommendation = result

if st.session_state.last_recommendation:
    st.subheader("Your Recommendations")
    st.markdown(st.session_state.last_recommendation)

# Chain 2 — Follow-up chain (Part C) 
st.divider()
follow_up = st.text_input("Ask a follow-up question about these movies:")

followup_template = PromptTemplate(
    input_variables=["recommendations", "question"],
    template=(
        "Here are some movie recommendations that were just given to a user:\n\n"
        "{recommendations}\n\n"
        "The user now has this follow-up question: {question}\n\n"
        "Answer the question clearly and helpfully using only the context of the "
        "recommended movies above."
    ),
)

followup_chain = followup_template | llm | StrOutputParser()

if follow_up:
    if not st.session_state.last_recommendation:
        st.warning("Please get recommendations first before asking a follow-up question.")
    else:
        with st.spinner("Thinking…"):
            followup_result = followup_chain.invoke(
                {
                    "recommendations": st.session_state.last_recommendation,
                    "question": follow_up,
                }
            )
        st.subheader("💬 Follow-Up Answer")
        st.markdown(followup_result)