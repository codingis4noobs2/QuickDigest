# Importing necessary library
import streamlit as st


# Setting up the page configuration
st.set_page_config(
    page_title="QuickDigest AI",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Defining the function to display the home page
def home():
    import streamlit as st
    from streamlit_extras.badges import badge
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.let_it_rain import rain

    # Displaying a rain animation with specified parameters
    rain(
        emoji="🎈",
        font_size=54,
        falling_speed=5,
        animation_length="1",
    )
    
    # Displaying a colored header with specified parameters
    colored_header(
        label="QuickDigest AI🧠, Your Intelligent Data Companion",
        description="~ Powered by OpenAI, Llamaindex, AssemblyAI, Langchain, Replicate, Clipdrop",
        color_name="violet-70",
    )

    # Displaying information and warnings in the sidebar
    st.sidebar.info(
        "Visit [OpenAI Pricing](https://openai.com/pricing#language-models) to get an overview of costs incurring depending upon the model chosen."
    )
    st.sidebar.info(
        "For key & data privacy concerns, We do not store your Key, it will be removed after your session ends. Also OpenAI will not use data submitted by customers via our API to train or improve our models, unless you explicitly decide to share your data with us for this purpose, For more info please visit [OpenAI FAQs](https://help.openai.com/en/articles/7039943-data-usage-for-consumer-services-faq)."
    )
    st.sidebar.warning(
        "LLMs may produce inaccurate information about people, places, or facts. Don't entirely trust them."
    )
    
    # Displaying markdown text on the page
    st.markdown(
        "<h6>Discover a new horizon of data interaction with QuickDigest AI, your intelligent companion in navigating through diverse data formats. QuickDigest AI is meticulously crafted to simplify and enrich your engagement with data, ensuring a seamless flow of insights right at your fingertips.</h6>",
        unsafe_allow_html=True
    )
    st.markdown(
        "**Effortless Data Extraction and Interaction:** QuickDigest AI stands as a beacon of innovation, allowing users to upload and interact with a variety of file formats including PDFs, Word documents, text files, and even audio/video files. The platform's cutting-edge technology ensures a smooth extraction of data, paving the way for meaningful conversations with the information gleaned from these files."
    )
    st.markdown(
        "**Engage with your Datasets:** Dive into datasets like never before. QuickDigest AI invites you to upload your dataset and engage in a dialogue with it. Our advanced AI algorithms facilitate a conversational interaction with your dataset, making the extraction of insights an intuitive and enriching experience."
    )
    st.markdown(
        "**Real-Time Web Search:** One of the limitations of large language models is there limited knowledge. QuickDigest AI's real-time web search feature ensures you're always ahead with the latest information. Be it market trends, news updates, or the newest research findings, QuickDigest AI brings the world to you in real-time."
    )
    st.markdown(
        "**Ignite Your Creative Spark:** For product creators, QuickDigest AI unveils a realm of possibilities. Bored of simple product images, The Product Advertising Image Creator is your tool to craft captivating advertising images that resonate with your audience. Additionally, the Image Generator feature is your canvas to bring your creative visions to life, creating visually appealing images that speak volumes."
    )

    st.markdown("---")
    
    # Displaying a support section with badges and link button
    st.markdown("<h5>Support Us</h5>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Star this repository on Github")
        badge(type="github", name="codingis4noobs2/QuickDigest")
    with col2:
        st.write("Follow me on twitter")
        badge(type="twitter", name="4gameparth")
    with col3:
        st.write("Buy me a coffee")
        badge(type="buymeacoffee", name="codingis4noobs2")
    with col4:
        st.link_button("Upvote on Replit", "https://replit.com/@ParthShah38/QuickDigestAI?v=1")


# Function to display chat with files page
def chat_with_files():
    import os
    import streamlit as st
    from streamlit_extras.badges import badge
    from streamlit_extras.colored_header import colored_header
    from llama_index import (
        OpenAIEmbedding,
        ServiceContext,
        set_global_service_context,
    )
    from llama_index.llms import OpenAI
    from llama_index.chat_engine.types import StreamingAgentChatResponse
    from llama_index import SimpleDirectoryReader, VectorStoreIndex
    import assemblyai as aai
    from PyPDF2 import PdfReader
    from docx import Document

    # Cache the result to avoid recomputation
    @st.cache_resource(show_spinner="Indexing documents...Please have patience")
    def build_index(files):
        documents = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

    # Handle streaming responses
    def handle_stream(root, stream: StreamingAgentChatResponse):
        text = ""
        root.markdown("Thinking...")
        for token in stream.response_gen:
            text += token
            root.markdown(text)
        return text

    # Define constants and settings
    CACHE_DIR = "./uploads"
    aai.settings.api_key = st.secrets['assembly_api_key']

    # Render chat messages
    def render_message(message):
        with st.chat_message(message["role"]):
            st.write(message["text"])

    # Transcribe audio and video files
    def transcribe_audio_video(file_path):
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        transcript_path = file_path + ".txt"
        with open(transcript_path, "w") as f:
            f.write(transcript.text)
        return transcript_path

    # Upload files and cache them
    def upload_files(types=["pdf", "txt", "mp3", "mp4", 'mpeg', 'doc', 'docx'], **kwargs):
        files = st.file_uploader(
            label=f"Upload files", type=types, **kwargs
        )
        if not files:
            st.info(f"Please add documents, Note: Scanned documents are not supported yet!")
            st.stop()
        return cache_files(files, types=types)

    # Cache uploaded files
    def cache_files(files, types=["pdf", "txt", "mp3", "mp4", 'mpeg', 'doc', 'docx']) -> list[str]:
        filepaths = []
        for file in files:
            # Determine the file extension from the mime type
            ext = file.type.split("/")[-1]
            if ext == "plain":  # Handle text/plain mime type
                ext = "txt"
            elif ext in ["vnd.openxmlformats-officedocument.wordprocessingml.document", "vnd.ms-word"]:
                ext = "docx"  # or "doc" depending on your needs
            if ext not in types:
                continue
            filepath = f"{CACHE_DIR}/{file.name}"
            with open(filepath, "wb") as f:
                f.write(file.getvalue())
            if ext in ["mp3", "mp4"]:
                filepath = transcribe_audio_video(filepath)
            filepaths.append(filepath)
        # st.sidebar.write("Uploaded files", filepaths)  # Debug statement
        with st.sidebar:
            with st.expander("Uploaded Files"):
                filepaths_pretty = "\n".join(f"- {filepath}" for filepath in filepaths)
                st.markdown(f"{filepaths_pretty}")
        return filepaths

    def transcribe_and_save(file_path):
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        transcript_path = file_path + ".txt"
        with open(transcript_path, "w") as f:
            f.write(transcript.text)
        return transcript_path

    # Save extracted text to a txt file
    def save_extracted_text_to_txt(text, filename):
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join('uploads', txt_filename)
        with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
        return txt_filepath

    # Get OpenAI API key from session state
    def get_key():
        return st.session_state["openai_api_key"]

    # Read text from Word document
    def read_word_file(file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    # Process uploaded documents
    def process_documents(documents):
        processed_docs = []
        for doc in documents:
            if doc.endswith('.pdf'):
                processed_docs.append(process_pdf(doc))
            elif doc.endswith(('.doc', '.docx')):
                text = read_word_file(doc)
                txt_filepath = save_extracted_text_to_txt(text, os.path.basename(doc))
                processed_docs.append(txt_filepath)
            elif doc.endswith(('.mp3', '.mp4', '.mpeg')):
                processed_docs.append(transcribe_and_save(doc))
            else:
                processed_docs.append(doc)
        return processed_docs

    # Process PDF files
    def process_pdf(pdf_path):
        reader = PdfReader(pdf_path)
        all_text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                processed_text = ' '.join(extracted_text.split('\n'))
                all_text += processed_text + "\n\n"
        txt_filepath = save_extracted_text_to_txt(all_text, os.path.basename(pdf_path))
        os.remove(pdf_path)  # Delete the original PDF file
        return txt_filepath

    # Main logic for handling OpenAI API key and document processing
    if "openai_api_key" not in st.session_state:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.sidebar.warning("Please add your OpenAI API key to continue!!")
            st.sidebar.info("To obtain your OpenAI API key, please visit [OpenAI](https://platform.openai.com/account/api-keys). They provide a $5 credit to allow you to experiment with their models. If you're unsure about how to get the API key, you can follow this [Tutorial](https://www.maisieai.com/help/how-to-get-an-openai-api-key-for-chatgpt). While obtaining the API key doesn't require a compulsory payment, once your allotted credit is exhausted, a payment will be necessary to continue using their services.")
            st.stop()
        st.session_state["openai_api_key"] = openai_api_key

    st.sidebar.text_input("Enter Youtube Video ID(Coming soon)", disabled=True)
    st.sidebar.text_input("Enter Spotify Podast link(Coming soon)", disabled=True)
    
    openai_api_key = get_key()

    if openai_api_key:
        st.toast('OpenAI API Key Added ✅')
        # Define service-context
        with st.sidebar:
            with st.expander("Advanced Settings"):
                st.session_state['temperature'] = st.number_input("Enter Temperature", help="It determines how creative the model should be", min_value=0.0,max_value=1.0, value=0.1)
        llm = OpenAI(temperature=st.session_state['temperature'], model='gpt-3.5-turbo', api_key=openai_api_key)
        embed_model = OpenAIEmbedding(api_key=openai_api_key)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        set_global_service_context(service_context)

        # Upload PDFs, DOCs, TXTs, MP3s, and MP4s
        documents = upload_files(types=["pdf", "txt", "mp3", "mp4", 'mpeg', 'doc', 'docx'], accept_multiple_files=True)

        # Process the uploaded documents
        processed_documents = process_documents(documents)

        if not processed_documents:
            st.warning("No documents uploaded!")
            st.stop()

        index = build_index(processed_documents)
        query_engine = index.as_chat_engine(chat_mode="condense_question", streaming=True)

        messages = st.session_state.get("messages", [])

        if not messages:
            messages.append({"role": "assistant", "text": "Hi!"})

        for message in messages:
            render_message(message)

        if user_query := st.chat_input():
            message = {"role": "user", "text": user_query}
            messages.append(message)
            render_message(message)

            with st.chat_message("assistant"):
                stream = query_engine.stream_chat(user_query)
                text = handle_stream(st.empty(), stream)
                message = {"role": "assistant", "text": text}
                messages.append(message)
                st.session_state.messages = messages


# Function to use LLMs with web search
def use_llms_with_web():
    from langchain.agents import ConversationalChatAgent, AgentExecutor
    from langchain.callbacks import StreamlitCallbackHandler
    from langchain.chat_models import ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
    from langchain.tools import DuckDuckGoSearchRun
    import streamlit as st


    st.title("Use web search with LLMs")
    # Taking OpenAI API key input from the user
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    # Initializing message history and memory
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
    )
    # Resetting chat history logic
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}

    # Defining avatars for chat messages
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    # Taking new input from the user
    if prompt := st.chat_input(placeholder="Who won the 2022 Cricket World Cup?"):
        st.chat_message("user").write(prompt)
        # Checking if OpenAI API key is provided
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        # Initializing LLM and tools for web search
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
        tools = [DuckDuckGoSearchRun(name="Search")]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)

        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]


# Function to display chat with dataset page
def chat_with_dataset():
    from langchain.agents import AgentType
    from langchain.agents import create_pandas_dataframe_agent
    from langchain.callbacks import StreamlitCallbackHandler
    from langchain.chat_models import ChatOpenAI
    import streamlit as st
    import pandas as pd
    import os


    file_formats = {
        "csv": pd.read_csv,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "xlsm": pd.read_excel,
        "xlsb": pd.read_excel,
    }

    def clear_submit():
        """
        Clear the Submit Button State
        Returns:

        """
        st.session_state["submit"] = False

    @st.cache_data()
    def load_data(uploaded_file):
        """
        Load data from the uploaded file based on its extension.
        """
        try:
            ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        except:
            ext = uploaded_file.split(".")[-1]
        if ext in file_formats:
            return file_formats[ext](uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
            return None

    st.title("Chat with your dataset")
    st.info("Asking one question at a time will result in a better output")

    uploaded_file = st.file_uploader(
        "Upload a Data file",
        type=list(file_formats.keys()),
        help="Various File formats are Support",
        on_change=clear_submit,
    )

    df = None  # Initialize df to None outside the if block

    if uploaded_file:
        df = load_data(uploaded_file)  # df will be assigned a value if uploaded_file is truthy

    if df is None:  # Check if df is still None before proceeding
        st.warning("No data file uploaded or there was an error in loading the data.")
        return  # Exit the function early if df is None

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    st.sidebar.info("If you face a KeyError: 'content' error, Press the clear conversation histroy button")
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    # Display previous chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="What is this data about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Check if OpenAI API key is provided
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
        )

        pandas_df_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = pandas_df_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

# Function to display transform products page
def transform_products():
    import streamlit as st
    import requests
    import os
    import replicate
    import io
    from PIL import Image

    
    st.session_state['replicate_api_token'] = st.sidebar.text_input("Replicate API Token", type='password')
    os.environ['REPLICATE_API_TOKEN'] = st.session_state['replicate_api_token']

    if not st.session_state['replicate_api_token']:
        st.sidebar.warning('Please enter your Replicate API Token to continue!!')
        st.sidebar.info("You can get your Replicate API Token form here: [Replicate](https://replicate.com/account/api-tokens)")
        st.stop()
        
    if st.session_state['replicate_api_token']:
        st.info("This model works best with product images having transparent or plain backgrounds")
        # Prompt user to upload an image file
        img = st.file_uploader("Upload your product image", type=['png', 'jpg', 'jpeg'])

        if img is not None:
            has_plain_background = st.toggle("Does your product image have a plain or transparent background? If not, let us do the hard work for you!")
            prompt = st.text_input("Enter Prompt", help="Enter something you imagine...")
            negative_prompt = st.text_input("Enter Negative Prompt", help="Write what you don't want in the generated images")
            submit = st.button("Submit")

            if submit:
                if has_plain_background:
                    # If image already has a plain background, prepare it for Replicate
                    image = Image.open(img)
                    bytes_obj = io.BytesIO()
                    image.save(bytes_obj, format='PNG')
                    bytes_obj.seek(0)
                else:
                    # If image does not have a plain background, send it to ClipDrop to remove background
                    image_file_object = img.read()
                    r = requests.post('https://clipdrop-api.co/remove-background/v1',
                        files={
                            'image_file': ('uploaded_image.jpg', image_file_object, 'image/jpeg')
                        },
                        headers={'x-api-key': st.secrets['clipdrop_api_key']}
                    )

                    if r.ok:
                        # If background removal is successful, prepare image for Replicate
                        image = Image.open(io.BytesIO(r.content))
                        bytes_obj = io.BytesIO()
                        image.save(bytes_obj, format='PNG')
                        bytes_obj.seek(0)
                    else:
                        r.raise_for_status()
                        st.error('Failed to remove background. Try again.')
                        st.stop()

                # Send image to Replicate for transformation
                output = replicate.run(
                    "logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df",
                    input={"image_path": bytes_obj, "prompt": prompt, "image_num": 4}
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.image(output[1])
                    st.image(output[2])
                with col2:
                    st.image(output[3])
                    st.image(output[4])

# Function to generate images based on user input
def generate_images():
    import streamlit as st
    import replicate
    import os


    st.session_state['replicate_api_token'] = st.sidebar.text_input("Replicate API Token", type='password')
    os.environ['REPLICATE_API_TOKEN'] = st.session_state['replicate_api_token']

    if not st.session_state['replicate_api_token']:
        st.sidebar.warning('Please enter your Replicate API Token to continue!!')
        st.sidebar.info("You can get your Replicate API Token form here: [Replicate](https://replicate.com/account/api-tokens)")
        st.stop()

    if st.session_state['replicate_api_token']:
        prompt = st.text_input(
            "Enter prompt", 
            help="Write something you can imagine..."
        )
        negative_prompt = st.text_input(
            "Enter Negative prompt", 
            help="Write what you don't want to see in the generated images"
        )
        submit = st.button("Submit")

        if submit:
            output = replicate.run(
                "stability-ai/sdxl:8beff3369e81422112d93b89ca01426147de542cd4684c244b673b105188fe5f",
                input={
                    "prompt": prompt, 
                    "negative_prompt": negative_prompt, 
                    "num_outputs": 4
                },
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(output[0])
                st.image(output[2])
            with col2:
                st.image(output[1])
                st.image(output[3])

# Dictonary to store all functions as pages
page_names_to_funcs = {
    "Home 🏠": home,
    "Chat with files 📁": chat_with_files,
    "Chat with dataset 📖": chat_with_dataset,
    "Use web search with LLMs 🌐": use_llms_with_web,
    "Generate Images 🖌️": generate_images,
    "Transform your products 🎨": transform_products,
}

# display page by dictionary
demo_name = st.sidebar.selectbox("Choose a page to navigate to", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
