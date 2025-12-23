import streamlit as st
from langchain_helper import get_few_shot_db_chain

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    try:
        chain = get_few_shot_db_chain()
        response = chain.run(question)
        st.header("Answer")
        st.write(response)
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
        st.info(
            "Verify your database credentials and GOOGLE_API_KEY "
            "environment variable, then try again."
        )