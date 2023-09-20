import streamlit as st
from llama_index import OpenAIEmbedding, ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
import llamaindex
import st as st_utils

st.title("QuickDigest")

openai_api_key = st_utils.get_key()

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", api_key=openai_api_key)
embed_model = OpenAIEmbedding(api_key=openai_api_key)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# Upload PDFs, DOCs, and TXTs
documents = st_utils.upload_files(types=["pdf", "txt"], accept_multiple_files=True)

# Debug statement to print uploaded files
st.write(documents)

if not documents:
    st.warning("No documents uploaded!")
    st.stop()

index = llamaindex.build_index(documents)
query_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

messages = st.session_state.get("messages", [])

if not messages:
    messages.append({"role": "assistant", "text": "Hi!"})

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
