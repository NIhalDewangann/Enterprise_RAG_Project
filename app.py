import streamlit as st
import requests

st.title("Enterprise RAG Document Assistant")

st.header("1. Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Upload and Process"):
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post("http://localhost:8000/upload", files=files)
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

st.header("2. Ask a Question")
question = st.text_input("Enter your question about the document:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Generating answer..."):
            response = requests.post("http://localhost:8000/query", json={"question": question})
            if response.status_code == 200:
                st.subheader("Answer:")
                st.write(response.json()["answer"])
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    else:
        st.warning("Please enter a question.")