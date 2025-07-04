import databutton as db
import re
from io import BytesIO
from typing import Tuple, List
import pickle
import pandas as pd

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

import os
from dotenv import load_dotenv

load_dotenv()

# PARSING PDF
def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output, filename

    
def parse_csv(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """Parse CSV file and convert to structured text chunks with enhanced context."""
    try:
        df = pd.read_csv(file)
        output = []
        
        # Create comprehensive CSV metadata and summary
        summary = f"=== CSV ANALYSIS: {filename} ===\n"
        summary += f"Dataset Overview:\n"
        summary += f"- Total Records: {len(df)}\n"
        summary += f"- Total Columns: {len(df.columns)}\n"
        summary += f"- File: {filename}\n\n"
        
        # Column analysis with data types and sample values
        summary += "Column Details:\n"
        for col in df.columns:
            col_info = f"- {col}:\n"
            col_info += f"  * Type: {df[col].dtype}\n"
            col_info += f"  * Non-null count: {df[col].count()}/{len(df)}\n"
            
            # Add sample values (first few unique values)
            unique_vals = df[col].dropna().unique()[:3]
            if len(unique_vals) > 0:
                sample_vals = [str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in unique_vals]
                col_info += f"  * Sample values: {', '.join(sample_vals)}\n"
            
            # Add statistical info for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info += f"  * Range: {df[col].min()} to {df[col].max()}\n"
                col_info += f"  * Mean: {df[col].mean():.2f}\n"
            
            summary += col_info + "\n"
        
        output.append(summary)
        
        # Convert each row to structured JSON-like format with context
        for idx, row in df.iterrows():
            # Create structured row representation
            row_data = {}
            for col, value in row.items():
                # Handle different data types appropriately
                if pd.isna(value):
                    row_data[col] = "NULL"
                elif isinstance(value, (int, float)):
                    row_data[col] = value
                else:
                    row_data[col] = str(value)
            
            # Format as structured text with context
            row_text = f"RECORD {idx + 1} of {len(df)}:\n"
            row_text += "{\n"
            
            for col, value in row_data.items():
                # Add type hints and context
                if isinstance(value, str) and value != "NULL":
                    row_text += f'  "{col}": "{value}",\n'
                elif value == "NULL":
                    row_text += f'  "{col}": null,\n'
                else:
                    row_text += f'  "{col}": {value},\n'
            
            # Remove trailing comma and close
            row_text = row_text.rstrip(',\n') + '\n}\n'
            
            # Add contextual information for this record
            row_text += f"Context: This is record {idx + 1} out of {len(df)} total records in {filename}\n"
            
            # Add relationships or patterns if detectable
            if idx > 0:
                row_text += f"Previous record index: {idx}\n"
            if idx < len(df) - 1:
                row_text += f"Next record index: {idx + 2}\n"
            
            row_text += "---\n"  # Separator between records
            
            output.append(row_text)
        
        # Add a final summary chunk with insights
        insights = f"=== DATA INSIGHTS: {filename} ===\n"
        insights += f"Total records processed: {len(df)}\n"
        insights += f"Data completeness: {((df.count().sum()) / (len(df) * len(df.columns)) * 100):.1f}% of cells contain data\n"
        
        # Identify potential key columns
        potential_keys = []
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].count() == len(df):
                potential_keys.append(col)
        
        if potential_keys:
            insights += f"Potential unique identifier columns: {', '.join(potential_keys)}\n"
        
        # Identify categorical columns
        categorical_cols = []
        for col in df.columns:
            if df[col].nunique() < len(df) * 0.5 and df[col].nunique() > 1:
                categorical_cols.append(f"{col} ({df[col].nunique()} unique values)")
        
        if categorical_cols:
            insights += f"Categorical columns: {', '.join(categorical_cols)}\n"
        
        insights += "\nThis structured format allows for:\n"
        insights += "- Easy identification of individual records\n"
        insights += "- Clear column-value relationships\n"
        insights += "- Data type recognition\n"
        insights += "- Pattern detection across records\n"
        insights += "- Contextual understanding of dataset structure\n"
        
        output.append(insights)
        
        return output, filename
        
    except Exception as e:
        error_msg = f"Error parsing CSV {filename}: {str(e)}\n"
        error_msg += "This could be due to:\n"
        error_msg += "- Encoding issues (try UTF-8, latin-1, or cp1252)\n"
        error_msg += "- Malformed CSV structure\n"
        error_msg += "- Empty or corrupted file\n"
        error_msg += "- Unsupported file format\n"
        return [error_msg], filename



def parse_csv_by_path(file_path: str, filename: str) -> Tuple[List[str], str]:
    """Parse CSV file from file path and convert to structured text chunks with enhanced context."""
    try:
        # Read CSV directly from file path
        df = pd.read_csv(file_path)
        output = []
        
        # Create comprehensive CSV metadata and summary
        summary = f"=== CSV ANALYSIS: {filename} ===\n"
        summary += f"Dataset Overview:\n"
        summary += f"- Total Records: {len(df)}\n"
        summary += f"- Total Columns: {len(df.columns)}\n"
        summary += f"- File: {filename}\n"
        summary += f"- Source Path: {file_path}\n\n"
        
        # Column analysis with data types and sample values
        summary += "Column Details:\n"
        for col in df.columns:
            col_info = f"- {col}:\n"
            col_info += f"  * Type: {df[col].dtype}\n"
            col_info += f"  * Non-null count: {df[col].count()}/{len(df)}\n"
            
            # Add sample values (first few unique values)
            unique_vals = df[col].dropna().unique()[:3]
            if len(unique_vals) > 0:
                sample_vals = [str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in unique_vals]
                col_info += f"  * Sample values: {', '.join(sample_vals)}\n"
            
            # Add statistical info for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info += f"  * Range: {df[col].min()} to {df[col].max()}\n"
                col_info += f"  * Mean: {df[col].mean():.2f}\n"
            
            summary += col_info + "\n"
        
        output.append(summary)
        
        # Convert each row to structured JSON-like format with context
        for idx, row in df.iterrows():
            # Create structured row representation
            row_data = {}
            for col, value in row.items():
                # Handle different data types appropriately
                if pd.isna(value):
                    row_data[col] = "NULL"
                elif isinstance(value, (int, float)):
                    row_data[col] = value
                else:
                    row_data[col] = str(value)
            
            # Format as structured text with context
            row_text = f"RECORD {idx + 1} of {len(df)}:\n"
            row_text += "{\n"
            
            for col, value in row_data.items():
                # Add type hints and context
                if isinstance(value, str) and value != "NULL":
                    row_text += f'  "{col}": "{value}",\n'
                elif value == "NULL":
                    row_text += f'  "{col}": null,\n'
                else:
                    row_text += f'  "{col}": {value},\n'
            
            # Remove trailing comma and close
            row_text = row_text.rstrip(',\n') + '\n}\n'
            
            # Add contextual information for this record
            row_text += f"Context: This is record {idx + 1} out of {len(df)} total records in {filename}\n"
            
            # Add relationships or patterns if detectable
            if idx > 0:
                row_text += f"Previous record index: {idx}\n"
            if idx < len(df) - 1:
                row_text += f"Next record index: {idx + 2}\n"
            
            row_text += "---\n"  # Separator between records
            
            output.append(row_text)
        
        # Add a final summary chunk with insights
        insights = f"=== DATA INSIGHTS: {filename} ===\n"
        insights += f"Total records processed: {len(df)}\n"
        insights += f"Data completeness: {((df.count().sum()) / (len(df) * len(df.columns)) * 100):.1f}% of cells contain data\n"
        
        # Identify potential key columns
        potential_keys = []
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].count() == len(df):
                potential_keys.append(col)
        
        if potential_keys:
            insights += f"Potential unique identifier columns: {', '.join(potential_keys)}\n"
        
        # Identify categorical columns
        categorical_cols = []
        for col in df.columns:
            if df[col].nunique() < len(df) * 0.5 and df[col].nunique() > 1:
                categorical_cols.append(f"{col} ({df[col].nunique()} unique values)")
        
        if categorical_cols:
            insights += f"Categorical columns: {', '.join(categorical_cols)}\n"
        
        insights += "\nThis structured format allows for:\n"
        insights += "- Easy identification of individual records\n"
        insights += "- Clear column-value relationships\n"
        insights += "- Data type recognition\n"
        insights += "- Pattern detection across records\n"
        insights += "- Contextual understanding of dataset structure\n"
        
        output.append(insights)
        
        return output, filename
        
    except Exception as e:
        error_msg = f"Error parsing CSV {filename} from path {file_path}: {str(e)}\n"
        error_msg += "This could be due to:\n"
        error_msg += "- File not found or inaccessible\n"
        error_msg += "- Encoding issues (try UTF-8, latin-1, or cp1252)\n"
        error_msg += "- Malformed CSV structure\n"
        error_msg += "- Empty or corrupted file\n"
        error_msg += "- Insufficient permissions\n"
        return [error_msg], filename

# CHUNKING BY LANGCHAIN TEXT-SPLITTER
def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,  # Added some overlap for better context
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, 
                metadata={
                    "page": doc.metadata["page"], 
                    "chunk": i,
                    "filename": filename
                }
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

# FETCH FOR SIMILARITY RETRIEVAL USING HUGGING FACE EMBEDDINGS
def docs_to_index(docs):
    """Create FAISS index using Hugging Face embeddings."""
    # Initialize Hugging Face embeddings
    # You can choose different models based on your needs:
    # - "sentence-transformers/all-MiniLM-L6-v2" (fast, good performance)
    # - "sentence-transformers/all-mpnet-base-v2" (better performance, slower)
    # - "BAAI/bge-small-en-v1.5" (good multilingual support)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    index = FAISS.from_documents(docs, embeddings)
    return index

# INDEX VECTORS PROPERLY
def get_index_for_pdf(pdf_files, pdf_names, openai_api_key=None):
    """Create index for PDF files. openai_api_key parameter kept for compatibility but not used."""
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents)
    return index

# NEW: INDEX FOR CSV FILES
def get_index_for_csv(csv_files, csv_names):
    """Create index for CSV files."""
    documents = []
    for csv_file, csv_name in zip(csv_files, csv_names):
        text, filename = parse_csv(BytesIO(csv_file), csv_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents)
    return index

# NEW: INDEX FOR MIXED FILES (PDF + CSV)
def get_index_for_mixed_files(files, filenames, file_types):
    """Create index for mixed file types (PDF and CSV)."""
    documents = []
    for file, filename, file_type in zip(files, filenames, file_types):
        if file_type.lower() == 'pdf':
            text, fname = parse_pdf(BytesIO(file), filename)
        elif file_type.lower() == 'csv':
            text, fname = parse_csv(BytesIO(file), filename)
        else:
            continue  # Skip unsupported file types
        
        documents = documents + text_to_docs(text, fname)
    index = docs_to_index(documents)
    return index

# ALTERNATIVE EMBEDDING MODELS
def get_index_with_custom_embeddings(documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create index with custom embedding model."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    index = FAISS.from_documents(documents, embeddings)
    return index

# SUMMARIZATION FUNCTION (keeping for compatibility, but now uses HF API)
def summarize_pdf(file, file_name):
    """Generate a summary for the given PDF file using Hugging Face API."""
    from huggingface_hub import InferenceClient
    
    # Read the PDF and extract the text
    text, _ = parse_pdf(file, file_name)
    full_text = " ".join(text)
    
    # Truncate text if too long (adjust based on model limits)
    max_length = 2000
    if len(full_text) > max_length:
        full_text = full_text[:max_length] + "..."
    
    # Create a prompt for summarization
    summary_prompt = f"Please provide a concise summary of the following document:\n\n{full_text}"
    
    try:
        # Initialize Hugging Face Inference Client
        client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_TOKEN"))
        
        # Generate summary
        response = client.chat.completions.create(
            model="microsoft/phi-4",  # or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"