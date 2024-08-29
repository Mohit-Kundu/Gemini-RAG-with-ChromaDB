# Gemini-RAG-with-ChromaDB
A modular Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Google Generative AI, and Streamlit for advanced document-based Q&amp;A.

## Features

- **Google Generative AI Integration:** Leverages the Gemini model for generating human-like responses.
- **Retrieval-Augmented Generation (RAG):** Combines retrieval-based models with generative AI for contextual responses.
- **Supports Multiple Document Types:** Loads and processes PDFs, TXT files, and more using LangChain document loaders.
- **User-Friendly Interface:** Streamlit-based UI for easy interaction with the chatbot and document management.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Required Python libraries as mentioned in `requirements.txt`

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/gemini-rag-chatbot.git
    cd gemini-rag-chatbot
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

    Insert your gemin api key in the `.env` file:

    ```env
    GOOGLE_API_KEY=your_google_api_key
    ```

### Usage

1. **Prepare Your Documents:**
Place your PDF or TXT files in a seperate directory (e.g., `docs/`) and directory path in gemini-rag.py
2. **Run the Application:**

   Start the Streamlit application by running:

   ```bash
   streamlit run gemini-rag.py
