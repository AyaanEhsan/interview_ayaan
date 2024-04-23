import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()
OPEN_AI_API_KEY = "sk-proj-XzQOsCWpeTEuh2S4Q4s7T3BlbkFJiXRY5Fn1JgF0GIyJ1iAr"
# OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print(OPEN_AI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME)

import streamlit as st  
from openai import OpenAI
import openai
from streamlit_chat import message
# from utils import *
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.embeddings.openai import OpenAIEmbeddings
OPENAI_API_KEY ="sk-proj-XzQOsCWpeTEuh2S4Q4s7T3BlbkFJiXRY5Fn1JgF0GIyJ1iAr"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


def find_match(query,k=1,score=False):
    input_em = embeddings.embed_query(query)
    index = PineconeVectorStore(index_name =PINECONE_INDEX_NAME,embedding= embeddings)
    similar_docs = index.similarity_search(query, k=1)
    return similar_docs


st.markdown("<h1 style='text-align: center; color: grey;'>🤖 Interview-Ayaan 👨</h1>", unsafe_allow_html=True)


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am Ayaan Ehsan Mohammed's personal assistant. How can I help you?"),
    ]



def get_response(user_query, chat_history):
    # We recieve the context "string"
    context_from_db = find_match(user_query)

    for i in range(len(context_from_db)):
        context_from_db[i].page_content = f"""
     You are Ayaan Ehsan Mohammed personal assistant. Answer the following questions considering the history of the conversation and Context retrieved from Ayaan Ehsan's Database:
     Chat history: {chat_history}
     User question: {user_query}
     Context to respond to user question: {context_from_db[0].page_content}. Always give context preference over conversation history. Do not mention source.
    """
        print(f"contexttt After appending---> {context_from_db[i].page_content}")

    print(f"contexttt After appending---> {len(context_from_db)}")
    
        
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key = OPEN_AI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    chains_response = chain.run(input_documents = context_from_db, question = user_query)
    
    return chains_response

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# Append questions and responses
user_query = st.chat_input("Type question that you want to ask Ayaan !")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))


# # Show Chat Hsitory
# with st.expander("Show Chat Hsitory"):
#     st.session_state


