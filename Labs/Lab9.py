import streamlit as st
import json
import os
from anthropic import Anthropic

# Page config
st.title("🧠 Chatbot with Long-Term Memory")
st.caption("Memories persist across sessions via memories.json")

# Constants
MEMORY_FILE = "memories.json"
CHAT_MODEL = "claude-opus-4-5"
EXTRACT_MODEL = "claude-haiku-4-5-20251001"

# API client
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Memory functions

def load_memories():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memories(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

def extract_new_memories(user_msg, assistant_msg, existing_memories):
    existing_str = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "None"

    extraction_prompt = f"""You are a memory extraction assistant. Identify any NEW facts worth remembering about the user from this conversation exchange.

EXISTING MEMORIES (do NOT repeat these):
{existing_str}

CONVERSATION:
User: {user_msg}
Assistant: {assistant_msg}

Extract only new facts about the user (name, location, school, major, interests, preferences, etc.).
Do NOT duplicate anything already in existing memories.
If there is nothing new, return an empty list.

Respond ONLY with a valid JSON array of strings. No markdown, no explanation.
Example: ["User's name is Alex", "User studies at MIT"]
If nothing new: []"""

    response = client.messages.create(
        model=EXTRACT_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": extraction_prompt}]
    )

    try:
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        new_facts = json.loads(raw)
        return [str(f) for f in new_facts if f] if isinstance(new_facts, list) else []
    except (json.JSONDecodeError, Exception):
        return []

def build_system_prompt(memories):
    base = "You are a helpful, friendly assistant with long-term memory."
    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        return (
            f"{base}\n\n"
            f"Here are things you remember about this user from past conversations:\n"
            f"{memory_block}"
        )
    return base

# Sidebar
st.sidebar.header("🧠 Long-Term Memory")

memories = load_memories()

if memories:
    st.sidebar.write(f"**{len(memories)} memor{'y' if len(memories) == 1 else 'ies'} stored:**")
    for mem in memories:
        st.sidebar.write(f"- {mem}")
else:
    st.sidebar.write("No memories yet. Start chatting!")

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear All Memories"):
    save_memories([])
    st.rerun()

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.write(f"**Model:** {CHAT_MODEL}")
st.sidebar.write(f"**Messages in session:** {len(st.session_state.messages)}")

# Chat display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Say something — I'll remember what matters..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    current_memories = load_memories()
    system_prompt = build_system_prompt(current_memories)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.messages.create(
                model=CHAT_MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=st.session_state.messages,
            )
            reply = response.content[0].text
        st.markdown(reply)

        # Extract and save new memories
        new_facts = extract_new_memories(prompt, reply, current_memories)
        if new_facts:
            updated = current_memories + new_facts
            save_memories(updated)
            st.info(f"✦ New memor{'y' if len(new_facts) == 1 else 'ies'} saved: {', '.join(new_facts)}")

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()