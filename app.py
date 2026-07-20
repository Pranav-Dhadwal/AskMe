import streamlit as st
import pandas as pd
from core import run_query

# -- Basic Config 
st.set_page_config(page_title="AskMe", page_icon="💬")
st.title("💬 AskMe")
st.caption("Upload a CSV and ask questions about it in plain English.")

#  -- Upload handling ---------------------------------
uploaded_file = st.file_uploader("Upload a CSV", type="csv")
if uploaded_file is not None:
    if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.filename = uploaded_file.name
        st.session_state.messages = []

    st.subheader("Preview")
    st.dataframe(st.session_state.df.head())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# -- user question handling ---------------------------------
    question = st.chat_input("Ask a question about your data")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # -- calling core functionality --------------------------
                answer, result, code, error = run_query(st.session_state.df, question)

            with st.expander("Generated code"):
                st.code(code, language="python")

            if error:
                st.error(error)
                answer = f"Couldn't answer that: {error}"
            else:
                with st.expander('Result'):
                    st.write(result)
                st.write(f'Summary : {answer}')

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a CSV to get started.")