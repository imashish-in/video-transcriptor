# Apply PyTorch path fix BEFORE importing Streamlit
import os
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], "classes")]

# Now import other modules
import streamlit as st
import whisper
import requests

# Updated Ollama endpoint (remove '/api' from URL)
OLLAMA_URL = "http://localhost:11434/generate"  # Fixed endpoint
OLLAMA_MODEL = "llama3"  # Ensure this model is installed via 'ollama pull llama3'

st.title("🎬 Local Video Summarizer (Ollama)")

def summarize_with_ollama(transcript):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Summarize this video transcript concisely:\n{transcript}",
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        return response.json().get("response", "No summary generated")
    except requests.ConnectionError:
        st.error("Ollama server not running! Start it with 'ollama serve'")
    except Exception as e:
        st.error(f"Summarization failed: {str(e)}")
        return "Error"

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    with st.spinner("Transcribing..."):
        # Save and transcribe
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
            
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe("temp_video.mp4")
        transcript = result["text"]
        
        st.subheader("Transcript")
        st.write(transcript)
    
    with st.spinner("Summarizing..."):
        summary = summarize_with_ollama(transcript)
        st.subheader("Summary")
        st.write(summary)
