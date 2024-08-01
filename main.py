import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Infant Nutrition Chatbot", page_icon="🍼", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets["OpenAi_Api_key"]
st.title("Infant Nutrtion Chatbot 👶💬")
st.info("Use this RAG extended LLM to get information about infant nutrition recommendations.", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about baby feeding (Example: When can babies start eating solid food?)",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        system_prompt="""You are an expert on infant nutrition and your job is to answer questions. 
        Assume that all questions are related to the domain of infant and toddler feeding and nutrition. 
        Keep your answers concise and based on facts. If the answer is unknown then reply; I don't know.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()
# explore different chat_modes 
if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input("Ask a question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
