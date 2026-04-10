import streamlit as st
from openai import OpenAI
import requests
import base64

# Page config 
st.set_page_config(page_title="Lab 8 – Image Captioning Bot", page_icon="🖼️")

# OpenAI client 
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Session state
if "url_response" not in st.session_state:
    st.session_state.url_response = None

if "upload_response" not in st.session_state:
    st.session_state.upload_response = None

# Shared prompt
PROMPT = (
    "Describe the image in at least 3 sentences. "
    "Write five different captions for this image. "
    "Captions must vary in length, minimum one word but be no longer than 2 sentences. "
    "Captions should vary in tone, such as, but not limited to funny, intellectual, and aesthetic."
)


# Header
st.title("🖼️ Image Captioning Bot")
st.write(
    "Provide the bot with either an image URL or file upload "
    "and let it write your captions for you!"
)

st.divider()

# Part A – Image URL Input
st.header("📎 Image URL Input")
st.write("Input your image URL here")

url = st.text_input(
    "Image URL",
    placeholder="https://example.com/image.jpg",
    help="Ensure the link leads directly to the image and not a webpage with images on it.",
)
st.caption("⚠️ Ensure the link leads directly to the image and not a webpage with images on it.")

if st.button("Generate Caption for Inputted URL"):
    if url:
        with st.spinner("Generating captions from URL…"):
            url_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": url, "detail": "auto"},
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            st.session_state.url_response = url_response
    else:
        st.warning("Please enter an image URL first.")

if st.session_state.url_response:
    st.image(url, use_container_width=True)
    st.write(st.session_state.url_response.choices[0].message.content)

st.divider()

# Part B – File Upload

st.header("📤 Image Upload Input")
st.write("Upload your image here")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "gif"],
)

if st.button("Generate Caption for Uploaded Image"):
    if uploaded:
        with st.spinner("Generating captions from uploaded image…"):
            # Encode image to base64
            b64 = base64.b64encode(uploaded.read()).decode("utf-8")
            mime = uploaded.type   # e.g. "image/png"
            data_uri = f"data:{mime};base64,{b64}"

            upload_response = client.chat.completions.create(
                model="gpt-4.1-mini",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri, "detail": "low"},
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            st.session_state.upload_response = upload_response
    else:
        st.warning("Please upload an image first.")

if st.session_state.upload_response:
    # Reset file pointer so st.image can read it
    uploaded.seek(0)
    st.image(uploaded, use_container_width=True)
    st.write(st.session_state.upload_response.choices[0].message.content)