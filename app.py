import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG with Gemini", layout="wide")

st.title("🔍 Question-answering system powered by a Language Model and RAG")
st.markdown("This app answers your queries based on the provided documents. Type your query below!")

# Load and process multiple PDFs
pdf_files = ["yolov9_paper.pdf", "Python Machine Learning.pdf"]  # List of PDF file paths
all_docs = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    data = loader.load()
    all_docs.extend(data)  # Combine all documents into one list

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(all_docs)

# Create embeddings and vector store
gemini_embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=gemini_embeddings, persist_directory="chroma_db")
vectorstore_disk = Chroma(
    persist_directory="chroma_db", embedding_function=gemini_embeddings
)
retriever = vectorstore_disk.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Retrieve top 10 contexts

llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'),model="llama-3.3-70b-versatile",temperature=0.0,max_retries=2,)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask your question here...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Retrieve relevant contexts
    retrieved_docs = retriever.get_relevant_documents(query)
    contexts = [doc.page_content for doc in retrieved_docs]

    # Set up the chain
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Prepare the context string for the LLM
    context_str = "\n\n".join(contexts)
    response = rag_chain.invoke({"input": query, "context": context_str})
    answer = response["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer, "contexts": contexts})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").markdown(message["content"])
        
        # Display the contexts
        if "contexts" in message:
            with st.expander("View related contexts"):
                for i, context in enumerate(message["contexts"], 1):
                    st.markdown(f"**Context {i}:** {context}")
