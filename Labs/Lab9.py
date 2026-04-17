import streamlit as st
import json
import os
from anthropic import Anthropic

# Page config
st.set_page_config(
    page_title="Chatbot with Long-Term Memory",
    page_icon="🧠",
    layout="wide",
)

#  Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Page background */
.stApp {
    background: #0f0f13;
    color: #e8e4dc;
}

/* Title styling */
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    color: #e8e4dc !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem !important;
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b6b7a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16161d !important;
    border-right: 1px solid #2a2a35;
}

[data-testid="stSidebar"] h2 {
    font-family: 'DM Serif Display', serif !important;
    color: #e8e4dc !important;
    font-size: 1.2rem !important;
}

.memory-pill {
    background: #1e1e28;
    border: 1px solid #2e2e3d;
    border-left: 3px solid #7c6af7;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    font-family: 'DM Sans', sans-serif;
    color: #c5c0d8;
    line-height: 1.4;
}

.memory-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #7c6af7;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

.no-memory {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #45454f;
    text-align: center;
    padding: 1.5rem 0;
    border: 1px dashed #2a2a35;
    border-radius: 8px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e28;
    padding: 1rem 0 !important;
}

/* Input */
[data-testid="stChatInput"] textarea {
    background: #16161d !important;
    border: 1px solid #2e2e3d !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124, 106, 247, 0.15) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #2e2e3d !important;
    color: #6b6b7a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}

.stButton > button:hover {
    border-color: #ff5f5f !important;
    color: #ff5f5f !important;
    background: rgba(255, 95, 95, 0.05) !important;
}

/* Divider */
hr {
    border-color: #2a2a35 !important;
}

/* Status badges */
.status-badge {
    display: inline-block;
    background: rgba(124, 106, 247, 0.12);
    border: 1px solid rgba(124, 106, 247, 0.3);
    color: #9d8ff5;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 1.5rem;
}

/* New memory toast */
.new-memory-indicator {
    background: rgba(124, 106, 247, 0.08);
    border: 1px solid rgba(124, 106, 247, 0.2);
    border-radius: 6px;
    padding: 6px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #7c6af7;
    letter-spacing: 0.06em;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Constants
MEMORY_FILE = "memories.json"
CHAT_MODEL = "claude-opus-4-5"
EXTRACT_MODEL = "claude-haiku-4-5-20251001"

# API Client 
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Memory System 

def load_memories() -> list[str]:
    """Load memories from JSON file. Returns empty list if file doesn't exist."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_memories(memories: list[str]) -> None:
    """Save memories list to JSON file."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(memories, f, indent=2)
    except IOError as e:
        st.error(f"Could not save memories: {e}")


def extract_new_memories(user_msg: str, assistant_msg: str, existing_memories: list[str]) -> list[str]:
    """
    Use a cheap model to extract new memorable facts from the exchange.
    Returns a list of new memory strings (may be empty).
    """
    existing_str = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "None"

    extraction_prompt = f"""You are a memory extraction assistant. Your job is to identify NEW facts worth remembering about the user from a conversation exchange.

EXISTING MEMORIES (do NOT repeat these):
{existing_str}

CONVERSATION EXCHANGE:
User: {user_msg}
Assistant: {assistant_msg}

Extract ONLY new, distinct facts about the user that are NOT already captured in the existing memories above. Focus on:
- Name, nickname, age, location
- Job, major, school, career goals
- Hobbies, interests, favorite things (food, music, movies, etc.)
- Personal preferences or opinions they've expressed
- Important life context (family, relationships, health, etc.)

Rules:
- Only extract facts the user explicitly stated or strongly implied
- Do NOT duplicate anything already in existing memories
- Do NOT extract generic statements or things about the assistant
- If there are no new facts worth remembering, return an empty list

Respond ONLY with a valid JSON array of strings. No markdown, no explanation, no backticks.
Example: ["User's name is Alex", "User studies computer science at MIT", "User loves hiking"]
If nothing new: []"""

    try:
        response = client.messages.create(
            model=EXTRACT_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": extraction_prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        new_facts = json.loads(raw)
        if isinstance(new_facts, list):
            return [str(f) for f in new_facts if f]
        return []
    except (json.JSONDecodeError, Exception):
        return []


def build_system_prompt(memories: list[str]) -> str:
    """Build the system prompt, injecting long-term memories."""
    base = (
        "You are a thoughtful, personable AI assistant with long-term memory. "
        "You have a warm but concise communication style. "
        "When you know things about the user, reference them naturally — don't announce that you're using your memory, just use it."
    )
    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        return (
            f"{base}\n\n"
            f"Here are things you remember about this user from past conversations:\n"
            f"{memory_block}\n\n"
            f"Use this context to personalize your responses where relevant."
        )
    return base


# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_new_memories" not in st.session_state:
    st.session_state.last_new_memories = []

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 Long-Term Memory")
    st.markdown("---")

    memories = load_memories()

    if memories:
        st.markdown(f'<div class="memory-count">📌 {len(memories)} memor{"y" if len(memories) == 1 else "ies"} stored</div>', unsafe_allow_html=True)
        for mem in memories:
            st.markdown(f'<div class="memory-pill">{mem}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="no-memory">No memories yet.<br>Start chatting!</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑 Clear All Memories"):
        save_memories([])
        st.session_state.last_new_memories = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#45454f;line-height:1.6;">'
        'Memories persist across sessions via <code style="color:#7c6af7">memories.json</code>.<br><br>'
        'Chat history lives in session state and clears on refresh.'
        '</div>',
        unsafe_allow_html=True
    )

# Main Page
st.markdown('<h1>Chatbot with Long-Term Memory</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">IST 488/688 · Lab 9 · Persistent Memory via JSON</div>', unsafe_allow_html=True)

mem_count = len(load_memories())
badge_label = f"{mem_count} memor{'y' if mem_count == 1 else 'ies'} loaded" if mem_count else "no memories yet"
st.markdown(f'<div class="status-badge">🧠 {badge_label}</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Say something — I'll remember what matters…"):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Load memories and build system prompt
    current_memories = load_memories()
    system_prompt = build_system_prompt(current_memories)

    # Call the main LLM
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = client.messages.create(
                model=CHAT_MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=st.session_state.messages,
            )
            reply = response.content[0].text

        st.markdown(reply)

        # Extract new memories from this exchange
        new_facts = extract_new_memories(prompt, reply, current_memories)

        if new_facts:
            updated_memories = current_memories + new_facts
            save_memories(updated_memories)
            st.session_state.last_new_memories = new_facts

            facts_preview = " · ".join(f[:40] + ("…" if len(f) > 40 else "") for f in new_facts)
            st.markdown(
                f'<div class="new-memory-indicator">✦ remembered: {facts_preview}</div>',
                unsafe_allow_html=True
            )
        else:
            st.session_state.last_new_memories = []

    # Append assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()