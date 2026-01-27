import streamlit as st
from openai import OpenAI
import PyPDF2

# Read PDFs
def read_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please configure it in `.streamlit/secrets.toml` or as an environment variable.", icon="🔑")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Sidebar 
st.sidebar.header("Summary Options")

# Language selection dropdown
language = st.sidebar.selectbox(
    "Select output language:",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Portuguese", "Italian"]
)

# Summary type dropdown
summary_type = st.sidebar.selectbox(
    "Select summary type:",
    ["100 words", "2 connecting paragraphs", "5 bullet points"]
)

# Model selection checkbox
use_advanced = st.sidebar.checkbox("Use advanced model (GPT-4o)", value=False)
model = "gpt-4o" if use_advanced else "gpt-4o-mini"

st.sidebar.write(f"**Current model:** {model}")

def create_summary_prompt(document_text, summary_type, language):
    base_prompt = f"Please summarize the following document in {language}.\n\n"
    
    if summary_type == "100 words":
        instruction = "Provide a concise summary in approximately 100 words."
    elif summary_type == "2 connecting paragraphs":
        instruction = "Provide a summary in exactly 2 well-connected paragraphs that flow logically from one to the other."
    else:  # 5 bullet points
        instruction = "Provide a summary as exactly 5 clear and informative bullet points."
    
    return f"{base_prompt}{instruction}\n\nDocument:\n{document_text}"

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        document_text = read_pdf(uploaded_file)
    
    if not document_text.strip():
        st.error("Could not extract text from the PDF. Please try a different file.")
    else:
        st.success(f"Successfully extracted {len(document_text)} characters from the PDF.")
        
        # Show preview of extracted text
        with st.expander("Preview extracted text"):
            st.write(document_text[:1000] + "..." if len(document_text) > 1000 else document_text)
        
        # Generate summary button
        if st.button("Generate Summary", type="primary"):
            with st.spinner(f"Generating summary using {model}..."):
                try:
                    prompt = create_summary_prompt(document_text, summary_type, language)
                    
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that creates clear, accurate summaries of documents."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                    )
                    
                    st.subheader(f"Summary ({summary_type} in {language}):")
                    st.write_stream(stream)
                    
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
else:
    st.info("Please upload a PDF file to get started.")