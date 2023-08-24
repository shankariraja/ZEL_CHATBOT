import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from PIL import Image
from gtts import gTTS
from simpleplayer import simpleplayer
import streamlit as st
import os
import openai
from langchain.embeddings import SelfHostedHuggingFaceInstructEmbeddings


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

def handle_userinput(user_question): 
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    res_answer= response['answer']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    return res_answer

def play_audio(message):
    ta_tts = gTTS(message)
    ta_tts.save('trans1.mp3')
    player = simpleplayer('trans1.mp3')
    player.play()
    

def main():

    load_dotenv()
    st.set_page_config(page_title="ZELBOT")
    st.write(css, unsafe_allow_html=True)


    image = Image.open("zelbot-removebg-preview.png")

    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image(image, caption='Zelbot', use_column_width=True)
    st.header("WELCOME TO ZEL YOUR CHATBOT")
    #user_question = st.text_input("Ask a question:")
    st.write(bot_template.replace(
                "{{MSG}}", "Hello, How can I assist you ?"), unsafe_allow_html=True)
    play_audio("Hello, How can I assist you ?") 
    user_question = st.text_input("Ask a question:")

    if user_question:
        res = handle_userinput(user_question)
        play_audio(res)

    #with st.sidebar:
    directory_path = "data\pdf's"
        # Get all PDF files in the directory
    pdf_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]
    if st.button("Process"):
        with st.spinner("Processing"):
            #get pdf text
            raw_text = get_pdf_text(pdf_files)

                # get the text chunks
            text_chunks = get_text_chunks(raw_text)

                # create vector store
            vectorstore = get_vectorstore(text_chunks)


                # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                    vectorstore) 
                
                
                
                
if __name__ == '__main__':
    main()
