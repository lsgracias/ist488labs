import streamlit as st
from openai import OpenAI

st.title("🤖 Lab 3 - Chatbot with Memory")
st.write("A friendly chatbot that explains things so a 10-year-old can understand!")

# Get API key from secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.toml file.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Configuration
MAX_BUFFER_MESSAGES = 4  # Last 2 user messages + 2 assistant responses
MODEL = "gpt-4o-mini"

# System prompt
SYSTEM_PROMPT = """You are a friendly and helpful chatbot assistant. You must follow these rules:

1. ALWAYS explain things in a simple way that a 10-year-old child can easily understand.
2. Use simple words, fun examples, and comparisons to everyday things kids know about.
3. Avoid technical jargon - if you must use a big word, explain what it means.
4. Keep your answers short and easy to read.

5. After answering ANY question, you MUST ask: "Do you want more info?"

6. If the user says "Yes" (or similar like "yeah", "sure", "please", "tell me more"):
   - Provide additional interesting information about the topic
   - Use fun facts or examples
   - Then ask again: "Do you want more info?"

7. If the user says "No" (or similar like "nope", "no thanks", "I'm good"):
   - Say something friendly like "Okay! What else can I help you with?"
   - Wait for their next question

Remember: Be enthusiastic, friendly, and make learning fun!"""

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get buffered messages (keeps system prompt + last N messages)
def get_buffered_messages():
    """Returns system prompt + last MAX_BUFFER_MESSAGES from conversation history."""
    buffered = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add only the last MAX_BUFFER_MESSAGES (2 user + 2 assistant = 4 messages)
    if len(st.session_state.messages) > MAX_BUFFER_MESSAGES:
        buffered.extend(st.session_state.messages[-MAX_BUFFER_MESSAGES:])
    else:
        buffered.extend(st.session_state.messages)
    
    return buffered

# Function to count tokens
def count_tokens(messages):
    """Approximate token count for messages."""
    total_text = ""
    for msg in messages:
        total_text += msg.get("content", "")
    # Estimate: 1 token ≈ 4 characters
    return len(total_text) // 4

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get buffered messages for API call
    buffered_messages = get_buffered_messages()
    
    # Display token count in sidebar
    token_count = count_tokens(buffered_messages)
    st.sidebar.metric("Estimated Tokens", token_count)
    st.sidebar.caption(f"Messages in buffer: {len(buffered_messages) - 1}")  # -1 for system prompt
    
    # Generate response with streaming
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=buffered_messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar info
st.sidebar.header("Chat Info")
st.sidebar.write(f"**Model:** {MODEL}")
st.sidebar.write(f"**Buffer size:** {MAX_BUFFER_MESSAGES} messages")
st.sidebar.write(f"**Total messages:** {len(st.session_state.messages)}")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()