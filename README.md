
# RAG with Gemini
This repository implements a Retrieval-Augmented Generation (RAG) question-answering system using Gemini, Google Generative AI, and LangChain. The application processes PDF documents, stores their embeddings in a vector database, retrieves relevant contexts, and uses an LLM to generate concise answers to user queries.

## Setup

Prerequisites:

    - Python 3.8+

    - Streamlit

    - Required Python libraries:

        langchain

        langchain-community

        langchain-google-genai

        dotenv

        chromadb
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Sharptooth14/QA_tool.git
    cd rag-with-gemini
    ```
    
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a .env file to store your API keys:
    ```bash
    GOOGLE_API_KEY=your_google_api_key
    ```
## Run the application

1. Add your PDF files to the working directory or specify their paths in the code.

2. Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```

3. Open the app in your browser: http://localhost:8501
  

## How it works

Document Processing:

    - PDFs are loaded using PyPDFLoader from langchain_community.document_loaders.

    - Documents are split into smaller chunks using the RecursiveCharacterTextSplitter.

Embedding Creation:

    - Embeddings are generated using GoogleGenerativeAIEmbeddings and stored in a vector   database (Chroma).

Context Retrieval:

    - The vector database retrieves the top 10 most similar document chunks based on the user query.

Response Generation:

    - A Language Model (ChatGoogleGenerativeAI) generates concise answers using the retrieved contexts.

User Interface:

    - Interactive chat interface provided by Streamlit.

    - Chat history is displayed along with retrieved contexts.


## Key components

PDF Loading
```
loader = PyPDFLoader(pdf_file)
data = loader.load()
```

Document Splitting
```
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(all_docs)
```

Embedding Creation
```
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=docs, embedding=gemini_embeddings, persist_directory="chroma_db")
```

Retrieval and Generation Chain
```
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```
Chat Interface
```
query = st.chat_input("Ask your question here...")
response = rag_chain.invoke({"input": query, "context": context_str})
```

## Screenshots

![App Screenshot](https://github.com/Sharptooth14/QA_tool/blob/main/Screenshot%202024-12-25%20220447.png)

![App Screenshot](https://github.com/Sharptooth14/QA_tool/blob/main/Screenshot%202024-12-25%20220328.png)

![App Screenshot](https://github.com/Sharptooth14/QA_tool/blob/main/Screenshot%202024-12-25%20220344.png)


## Future Improvements

    - Add support for uploading PDF files dynamically.

    - Optimize document chunking for better retrieval performance.

    - Implement advanced context summarization techniques.
