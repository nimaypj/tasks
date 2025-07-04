import databutton as db
import streamlit as st
from pdf_handler import (
    get_index_for_pdf, 
    get_index_for_csv, 
    get_index_for_mixed_files,
    summarize_pdf, 
    parse_pdf, 
    parse_csv,
    parse_csv_by_path
)
from io import BytesIO
from huggingface_hub import InferenceClient
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv('.env', override=True)

st.set_page_config(page_title="DocuAssist", page_icon="🤖", layout="wide")
st.title("DocuAssist")
st.markdown("Upload PDF or CSV files and ask questions about their content!")

# Hugging Face API setup
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    st.error("Please set your HUGGINGFACE_API_TOKEN in your .env file")
    st.info("Get your free token from: https://huggingface.co/settings/tokens")
    st.stop()

# Model selection
model_options = [
    "microsoft/phi-4",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

selected_model = st.sidebar.selectbox("Select Language Model", model_options, index=0)

# Embedding model selection
embedding_options = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

selected_embedding = st.sidebar.selectbox("Select Embedding Model", embedding_options, index=0)

# Clean result.page_content before use
def sanitize_content(content):
    if content:
        cleaned = content.replace("</|im_end|>", "")
        cleaned = cleaned.replace("Question:", "").replace("Answer:", "")
        return cleaned.strip()
    return ""

# Hugging Face API
def query_huggingface_api(prompt, model_id=None, provider="auto"):
    if model_id is None:
        model_id = selected_model

    try:
        client = InferenceClient(provider=provider, api_key=HUGGINGFACE_API_TOKEN)
        completion = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_resource
def create_vectordb(files, filenames, file_types):
    with st.spinner("Creating vector database..."):
        if len(set(file_types)) == 1:
            if file_types[0].lower() == 'pdf':
                return get_index_for_pdf([file.getvalue() for file in files], filenames)
            elif file_types[0].lower() == 'csv':
                return get_index_for_csv([file.getvalue() for file in files], filenames)
        return get_index_for_mixed_files(
            [file.getvalue() for file in files], filenames, file_types
        )

def generate_response(prompt_text):
    try:
        return query_huggingface_api(prompt_text, selected_model)
    except Exception as e:
        return f"Error generating response: {str(e)}"

def summarize_with_hf_api(text, filename, file_type="PDF"):
    summarize_prompt = f"""
    Please provide a concise summary of the following {file_type} document titled "{filename}":

    {text[:2000]}

    Summary:
    """
    return generate_response(summarize_prompt)

def compare_documents_with_hf_api(combined_text, filenames):
    comparison_prompt = f"""
    Please compare the following documents ({', '.join(filenames)}) and provide key insights, similarities, and differences:

    {combined_text[:3000]}

    Comparison Analysis:
    """
    return generate_response(comparison_prompt)

uploaded_files = st.file_uploader(
    "Upload PDF or CSV files", 
    type=["pdf", "csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    file_details = []
    files, filenames, file_types = [], [], []

    for file in uploaded_files:
        file_type = file.name.split('.')[-1].upper()
        file_details.append({
            "Name": file.name,
            "Type": file_type,
            "Size": f"{file.size / 1024:.1f} KB"
        })
        files.append(file)
        filenames.append(file.name)
        file_types.append(file_type)

    st.subheader("📁 Uploaded Files")
    st.dataframe(pd.DataFrame(file_details), use_container_width=True)

    st.session_state["files"] = files
    st.session_state["filenames"] = filenames
    st.session_state["file_types"] = file_types

    try:
        vectordb = create_vectordb(files, filenames, file_types)
        st.session_state["vectordb"] = vectordb
        st.success("✅ Vector database created successfully!")
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")

with st.sidebar:
    st.header("📊 Document Operations")

    if "files" in st.session_state:
        files = st.session_state["files"]
        filenames = st.session_state["filenames"]
        file_types = st.session_state["file_types"]
        file_paths = st.session_state.get("file_paths", filenames)

        st.subheader("📝 Summarize Documents")
        for idx, file in enumerate(files):
            file_name = filenames[idx]
            file_type = file_types[idx]

            if st.button(f"Summarize {file_name}", key=f"summarize_{idx}"):
                with st.spinner(f"Summarizing {file_name}..."):
                    try:
                        file.seek(0)
                        if file_type.lower() == 'pdf':
                            text, _ = parse_pdf(BytesIO(file.read()), file_name)
                        elif file_type.lower() == 'csv':
                            text, _ = parse_csv(BytesIO(file.read()), file_name)
                        combined_text = " ".join(text)
                        summary = summarize_with_hf_api(combined_text, file_name, file_type)
                        st.sidebar.success(f"✅ Summary for {file_name}:")
                        st.sidebar.write(summary)
                    except Exception as e:
                        st.sidebar.error(f"Error summarizing {file_name}: {str(e)}")

        if len(files) > 1:
            st.subheader("🔄 Compare Documents")
            if st.button("Compare All Documents"):
                with st.spinner("Comparing documents..."):
                    try:
                        combined_text = ""
                        for idx, file in enumerate(files):
                            file.seek(0)
                            file_type = file_types[idx]
                            if file_type.lower() == 'pdf':
                                text, _ = parse_pdf(BytesIO(file.read()), file.name)
                            elif file_type.lower() == 'csv':
                                text, _ = parse_csv_by_path(file_paths[idx], file.name)
                            combined_text += f"\n--- {file.name} ---\n" + " ".join(text) + "\n"
                        comparison_result = compare_documents_with_hf_api(combined_text, filenames)
                        st.write("### 📊 Comparison Result:")
                        st.write(comparison_result)
                    except Exception as e:
                        st.error(f"Error comparing documents: {str(e)}")

        st.subheader("📈 Statistics")
        st.info(f"Total files: {len(files)}")
        st.info(f"PDFs: {sum(1 for ft in file_types if ft.lower() == 'pdf')} | CSVs: {sum(1 for ft in file_types if ft.lower() == 'csv')}")

# Chatbot Prompt Template
prompt_template = """
You are a helpful AI assistant that answers questions based on the provided document contexts.

Instructions:
- Keep your answers clear and concise
- Always cite the source filename and page/row number when referencing information
- If the information is not in the provided context, say "I don't have that information in the uploaded documents"
- Focus on providing accurate information from the documents

Context from documents:
{pdf_extract}

Question: {question}

Answer:
"""

st.subheader("💬 Ask Questions About Your Documents")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask anything about your uploaded documents...")

if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("⚠️ Please upload some documents first to create the vector database.")
        st.stop()

    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating response..."):
            try:
                search_results = vectordb.similarity_search(question, k=5)
                pdf_extract = "\n---\n".join([
                    f"Source: {res.metadata.get('filename', 'Unknown')} (Page/Row: {res.metadata.get('page', 'Unknown')})\n{sanitize_content(res.page_content)}"
                    for res in search_results
                ])
                full_prompt = prompt_template.format(pdf_extract=pdf_extract, question=question.strip())
                result = generate_response(full_prompt)
                st.write(result)
                st.session_state["chat_history"].append({"role": "assistant", "content": result})

                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(search_results):
                        st.write(f"**Source {i+1}:** {doc.metadata.get('filename', 'Unknown')} (Page/Row: {doc.metadata.get('page', 'Unknown')})")
                        st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        st.write("---")

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state["chat_history"].append({"role": "assistant", "content": error_msg})

st.markdown("---")
