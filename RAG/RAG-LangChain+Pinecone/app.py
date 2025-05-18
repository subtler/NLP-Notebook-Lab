import os
import streamlit as st
import tempfile
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- ENV SETUP ---
os.environ["PINECONE_API_KEY"] = "Your API key"  # Replace this

# --- PINECONE ---
pc = Pinecone()
index_name = "rag-index"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_model)

# --- HUGGINGFACE LLM ---
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# --- STREAMLIT UI ---
st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant with Pinecone + HuggingFace")

# --- STEP 3: File Upload & Insert ---
st.markdown("### 📤 Upload New Documents")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    # Choose loader and assign category
    if uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path)
        category = uploaded_file.name.replace(".txt", "")
    else:
        loader = PyPDFLoader(file_path)
        category = uploaded_file.name.replace(".pdf", "")

    docs = loader.load()
    for doc in docs:
        doc.metadata["category"] = category

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Insert to Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name
    )

    st.success(f"✅ `{uploaded_file.name}` added to knowledge base!")

# --- STEP 2: Category Filter ---
category_filter = st.selectbox(
    "📂 Filter by document category (optional):",
    ["All", "billing", "account", "support"]
)

# --- STEP 1: Ask Question ---
query = st.text_input("Ask me a question from your documents:")

# Create retriever + chain (with filter if needed)
if category_filter == "All":
    retriever = vectorstore.as_retriever()
else:
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"category": category_filter}}
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# Run query
if query:
    response = qa_chain.invoke({"query": query})

    st.markdown("### 🤖 Answer:")
    st.write(response["result"])

    st.markdown("### 📂 Sources:")
    for i, doc in enumerate(response["source_documents"]):
        st.markdown(f"**🔹 Source {i+1} — Category:** `{doc.metadata.get('category')}`")
        with st.expander("🔍 View Matched Text"):
            st.write(doc.page_content.strip())
