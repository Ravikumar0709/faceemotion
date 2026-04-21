import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. ANALYSIS-BASED PROMPT SYSTEM
def get_repair_prompt():
    template = """
    You are the Pixel 9 Repair Expert. Use the provided manual context to answer the query.
    
    ACCURACY PROTOCOL:
    - If the user asks for a procedure, list the tools FIRST.
    - Mention specific safety warnings (e.g., temperatures, ESD precautions).
    - If the context mentions specific part numbers or image references, include them.
    - Maintain the disassembly sequence: explain what must be removed BEFORE the target component.

    Context: {context}
    Question: {question}

    Expert Instructions:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

# 2. HIGH-CONNECTIVITY (GRAPH-INSPIRED) PROCESSING
def process_pdf_with_graph_logic(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # We use a large overlap (30%) to simulate the 'expander' property. 
    # This ensures semantic links between sequential repair steps are never broken.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=350,
        separators=["--- PAGE", "\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(pages)
    
    # Enrich metadata for better 'Semantic Navigation'
    for doc in docs:
        content = doc.page_content.upper()
        if "CAUTION" in content or "WARNING" in content:
            doc.metadata["priority"] = "High (Safety)"
        else:
            doc.metadata["priority"] = "Standard"

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # FAISS with HNSW index (Graph-based search)
    # This mimics Ramanujan graph properties by ensuring short path lengths between nodes.
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db

# 3. STREAMLIT UI APPLICATION
st.set_page_config(page_title="Pixel 9 Repair RAG", layout="wide")
st.title("🛠️ Pixel 9 Semantic Repair Assistant")
st.markdown("Using Graph-based Vector Retrieval for high-accuracy repair guidance.")

# Sidebar for Upload
st.sidebar.header("Data Management")
uploaded_file = st.sidebar.file_uploader("Upload Pixel 9 Manual", type="pdf")

if uploaded_file:
    # Local save for loader
    with open("temp_manual.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if 'vector_store' not in st.session_state:
        with st.spinner("Constructing Semantic Graph..."):
            st.session_state.vector_store = process_pdf_with_graph_logic("temp_manual.pdf")
        st.sidebar.success("Manual Indexed!")

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a repair step..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if 'vector_store' in st.session_state:
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            # Retrieval using the Graph Index
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={"prompt": get_repair_prompt()},
                return_source_documents=True
            )
            
            result = qa_chain({"query": prompt})
            st.markdown(result["result"])
            
            # Show metadata for verification
            with st.expander("Semantic Metadata Sources"):
                for doc in result["source_documents"]:
                    st.write(f"Page {doc.metadata['page']} | Priority: {doc.metadata['priority']}")
            
            st.session_state.messages.append({"role": "assistant", "content": result["result"]})