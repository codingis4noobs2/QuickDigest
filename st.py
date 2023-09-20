import os
import streamlit as st
import assemblyai as aai

CACHE_DIR = "./uploads"
aai.settings.api_key = st.secrets['assembly_api_key']

def render_message(message):
    with st.chat_message(message["role"]):
        st.write(message["text"])

def get_key():
    if "openai_api_key" not in st.session_state:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.session_state["openai_api_key"] = openai_api_key
    return st.session_state["openai_api_key"]

def transcribe_audio_video(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    transcript_path = file_path + ".txt"
    with open(transcript_path, "w") as f:
        f.write(transcript.text)
    return transcript_path

def upload_files(types=["txt", "mp3", "mp4", "mpeg"], **kwargs):
    files = st.sidebar.file_uploader(
        label=f"Upload files", type=types, **kwargs
    )
    # st.write("Uploaded files:", files)  # Debug statement
    if not files:
        st.info(f"Please add documents")
        st.stop()
    return cache_files(files, types=types)

def cache_files(files, types=["txt", "mp3", "mp4", 'mpeg']) -> list[str]:
    filepaths = []
    for file in files:
        # Determine the file extension from the mime type
        ext = file.type.split("/")[-1]
        if ext == "plain":  # Handle text/plain mime type
            ext = "txt"
        # st.write("Mime type:", file.type, "Determined extension:", ext)  # Debug statement
        if ext not in types:
            continue
        filepath = f"{CACHE_DIR}/{file.name}"
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        if ext in ["mp3", "mp4"]:
            filepath = transcribe_audio_video(filepath)
        filepaths.append(filepath)
    # st.write("Processed file paths:", filepaths)  # Debug statement
    return filepaths
