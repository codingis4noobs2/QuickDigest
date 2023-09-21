import streamlit as st
from llama_index import (
    OpenAIEmbedding,
    ServiceContext,
    set_global_service_context,
)
from llama_index.llms import OpenAI
import llamaindex  # Updated import
import st as st_utils  # Updated import to avoid naming conflict
import assemblyai as aai

st.set_page_config(page_title="QuickDigest AI", page_icon=":brain:")
st.title("QuickDigest AI - Powered by OpenAI, AssemblyAI & Llama Index")

openai_api_key = st_utils.get_key()

# Define service-context
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", api_key=openai_api_key)
embed_model = OpenAIEmbedding(api_key=openai_api_key)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# Upload PDFs, DOCs, TXTs, MP3s, and MP4s
documents = st_utils.upload_files(types=["txt", "mp3", "mp4", 'mpeg'], accept_multiple_files=True)

# Transcribe audio/video files and save the transcription as a .txt file
def transcribe_and_save(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    transcript_path = file_path + ".txt"
    with open(transcript_path, "w") as f:
        f.write(transcript.text)
    return transcript_path

# Check if any of the uploaded files are audio or video and transcribe them
for doc in documents:
    if doc.endswith(('.mp3', '.mp4', '.mpeg')):
        transcribed_path = transcribe_and_save(doc)
        documents.remove(doc)  # Remove the original audio/video file from the list
        documents.append(transcribed_path)  # Add the path of the transcribed .txt file to the list

# Debug statement to print uploaded files
# st.write(documents)

if not documents:
    st.warning("No documents uploaded!")
    st.stop()

index = llamaindex.build_index(documents)
query_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

messages = st.session_state.get("messages", [])

if not messages:
    messages.append({"role": "assistant", "text": "Hi, How can I assist you today?"})

for message in messages:
    st_utils.render_message(message)

if user_query := st.chat_input():
    message = {"role": "user", "text": user_query}
    messages.append(message)
    st_utils.render_message(message)

    with st.chat_message("assistant"):
        stream = query_engine.stream_chat(user_query)
        text = llamaindex.handle_stream(st.empty(), stream)
        message = {"role": "assistant", "text": text}
        messages.append(message)
        st.session_state.messages = messages
