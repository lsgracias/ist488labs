import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

st.title("Lab 6 - Research Agent")
st.write("Ask me anything! I have web search enabled and can remember your follow-up questions.")
st.caption("Web search enabled — responses will cite up-to-date sources.")

# API Setup
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Pydantic model for structured output (Part D)
class ResearchSummary(BaseModel):
    main_answer: str
    key_facts: list[str]
    source_hint: str

# Session state 
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None
if "first_answer" not in st.session_state:
    st.session_state.first_answer = None

# Sidebar
st.sidebar.header("Settings")

use_streaming   = st.sidebar.toggle("Stream response", value=True)
use_structured  = st.sidebar.checkbox("Return structured summary")

st.sidebar.divider()
st.sidebar.header("Agent Info")
st.sidebar.write("**Model:** gpt-4o")
st.sidebar.write("**Tool:** web_search_preview")
st.sidebar.write("**Memory:** previous_response_id chaining")

st.sidebar.divider()
if st.sidebar.button("Reset conversation"):
    st.session_state.last_response_id = None
    st.session_state.first_answer     = None
    st.rerun()

# Helper: display structured output
def display_structured(parsed: ResearchSummary):
    st.markdown(parsed.main_answer)
    st.markdown("**Key Facts:**")
    for fact in parsed.key_facts:
        st.markdown(f"- {fact}")
    st.caption(f"Source hint: {parsed.source_hint}")

# Helper: make a Responses API call
def call_responses_api(user_input: str, previous_id: str | None) -> str | ResearchSummary:
    common_kwargs = dict(
        model="gpt-4o",
        instructions="You are a helpful research assistant. Always cite your sources.",
        input=user_input,
        tools=[{"type": "web_search_preview"}],
        previous_response_id=previous_id,
    )

    if use_structured:
        # Part D: structured output via .parse()
        response = client.responses.parse(
            **common_kwargs,
            text_format=ResearchSummary,
        )
        st.session_state.last_response_id = response.id
        return response.output_parsed

    elif use_streaming:
        # Streaming via .create() with stream=True
        answer_placeholder = st.empty()
        full_text = ""
        with client.responses.stream(**common_kwargs) as stream:
            for event in stream:
                # Accumulate text delta events
                if hasattr(event, "delta") and hasattr(event.delta, "text"):
                    full_text += event.delta.text
                    answer_placeholder.markdown(full_text + "▌")
        answer_placeholder.markdown(full_text)
        # Retrieve final response ID from the completed response
        st.session_state.last_response_id = stream.get_final_response().id
        return full_text

    else:
        # Non-streaming
        response = client.responses.create(**common_kwargs)
        st.session_state.last_response_id = response.id
        return response.output_text

# Part A & C: First question
st.subheader("Ask a question")
user_question = st.text_input("Your question:", placeholder="e.g. What are the latest developments in AI regulation?")

if user_question and st.session_state.first_answer is None:
    with st.spinner("Researching..."):
        with st.chat_message("assistant"):
            result = call_responses_api(user_question, previous_id=None)
            if use_structured and isinstance(result, ResearchSummary):
                display_structured(result)
                st.session_state.first_answer = result.main_answer
            elif not use_streaming:
                st.markdown(result)
                st.session_state.first_answer = result
            else:
                st.session_state.first_answer = result  # already streamed above

elif st.session_state.first_answer:
    with st.chat_message("assistant"):
        st.markdown(f"*(Previous answer on record — ask a follow-up below or reset to start over.)*")

# Part B: Follow-up question
if st.session_state.last_response_id:
    st.divider()
    st.subheader("Ask a follow-up question")
    followup = st.text_input(
        "Follow-up:",
        placeholder="e.g. Can you give me more detail on the second point?",
        key="followup_input",
    )

    if followup:
        with st.spinner("Following up..."):
            with st.chat_message("assistant"):
                result = call_responses_api(followup, previous_id=st.session_state.last_response_id)
                if use_structured and isinstance(result, ResearchSummary):
                    display_structured(result)
                elif not use_streaming:
                    st.markdown(result)
                # streaming already rendered inside call_responses_api