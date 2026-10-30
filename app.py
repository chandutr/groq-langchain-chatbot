import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Groq LLM", page_icon=":robot:", layout="wide")
st.title("ITK Groq LLM")

#sidebar
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password", placeholder="gsk-.......")
    model = st.selectbox("Select a model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=100)

#User question input
user_question = st.text_input("Enter your question", placeholder="Type your question here...")

#Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions and help with various tasks."),
    ("user", "Question: {question}")
])

if groq_api_key:
    # Initialize the ChatGroq model
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    if st.button("Generate"):
        response = chain.invoke({"question": user_question})
        st.write(f"Response: {response}")
else:
    st.error("Please enter your Groq API Key in the sidebar to use the application.")
    

