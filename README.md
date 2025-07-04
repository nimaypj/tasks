# 🤖 DocuGenie

A powerful Retrieval-Augmented Generation (RAG) application that allows you to upload documents and ask questions about their content using open-source language models and embeddings.

## ✨ Features

* **Multi-format Support** : Upload and analyze both PDF and CSV files
* **Intelligent Q&A** : Ask natural language questions about your documents
* **Document Summarization** : Generate concise summaries of uploaded files
* **Document Comparison** : Compare multiple documents to find similarities and differences
* **Source Attribution** : Responses include references to specific pages/rows and filenames
* **Open Source** : Uses open-source language models and embedding models
* **Privacy-Focused** : Document processing happens locally for embeddings
* **Interactive Chat** : Streamlit-based chat interface with conversation history

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* API token for language model inference (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   https://github.com/nimaypj/tasks.git
   cd ai-rag-assistant
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACE_API_TOKEN=your_api_token_here
   ```
4. **Run the application**
   ```bash
   streamlit run chat_with_docs.py
   ```

##  🔧 Configuration
   
###  Language Models

The application supports multiple open-source language models:

* Microsoft Phi-4
* Meta Llama models
* Mistral models
* And more...

### Embedding Models

Choose from various embedding models for document similarity:

* Sentence Transformers (multiple variants)
* Multilingual models
* Optimized models for different use cases

## 📖 Usage

### 1. Upload Documents

* Click "Upload PDF or CSV files"
* Select one or multiple files
* Supported formats: PDF, CSV
* Files are automatically processed and indexed

### 2. Ask Questions

* Use the chat interface to ask questions about your documents
* Questions can be about specific content, summaries, or comparisons
* Responses include source references (filename and page/row)

### 3. Document Operations

Use the sidebar to:

* **Summarize** : Generate summaries for individual documents
* **Compare** : Analyze similarities and differences between multiple documents
* **View Statistics** : See file counts and types

### 4. Example Questions

* "What are the main points discussed in document X?"
* "Summarize the findings from the uploaded research paper"
* "What differences exist between these two reports?"
* "Find information about [specific topic] in the documents"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Upload Files  │───▶│  Text Extraction │───▶│   Chunking      │
│   (PDF/CSV)     │    │  & Preprocessing │    │   & Metadata    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat Interface│◄───│  Response Gen.   │◄───│ Vector Database │
│   & History     │    │  (Open LLMs)     │    │ (Local Embed.)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘

```
## 📁 Project Structure
```

ai-rag-assistant/
├── chat_with_docs.py          # Main Streamlit application
├── pdf_handler.py       # Document processing and indexing
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── .env.example         # Environment variables template
└── README.md           # This file
```
## ⚙️ Advanced Configuration

### Custom Models

You can modify the model lists in `chat_with_docs.py` to use different language models or embedding models based on your needs.

 ### Performance Tuning

* **Chunk Size** : Adjust `chunk_size` in `pdf_handler.py` for different document types
* **Search Results** : Modify the `k` parameter in similarity search for more/fewer results

### Memory Management

For large documents:
* Increase `chunk_overlap` for better context
* Adjust `max_tokens` in API calls
* Consider using more efficient embedding models

## 🔐 Privacy & Security

* **Local Processing** : Document embeddings are generated locally
* **No Data Storage** : Documents are processed in memory only
* **API Communication** : Only processed queries are sent to language model APIs
* **Source Control** : Sensitive files should be added to `.gitignore`

## 🐛 Troubleshooting

### Common Issues

1. **API Token Error**
   * Ensure your API token is correctly set in the `.env` file
   * Verify the token has necessary permissions
2. **Memory Issues**
   * Reduce chunk size for large documents
   * Process fewer documents simultaneously
   * Use CPU-optimized embedding models
3. **Slow Performance**
   * Enable GPU acceleration if available
   * Use smaller, faster embedding models
   * Reduce the number of search results (k parameter)
4. **File Upload Issues**
   * Check file format (PDF/CSV only)
   * Ensure files are not corrupted
   * Verify file size limits

## 📈 Performance Tips

* **Batch Processing** : Upload related documents together for better context
* **Clear Questions** : Be specific in your questions for better results
* **Document Quality** : Ensure PDFs have extractable text (not just images)
* **CSV Structure** : Use well-formatted CSVs with clear headers



Made by Nimay PJ

# 🚀 Updated README on July 2
