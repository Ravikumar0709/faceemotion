import os
import asyncio
import faiss
from flask import Flask, render_template, request, redirect
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# SharePoint configuration
SHAREPOINT_SITE_URL = os.getenv("SHAREPOINT_SITE_URL")
SHAREPOINT_FOLDER = os.getenv("SHAREPOINT_FOLDER")
SHAREPOINT_USERNAME = os.getenv("SHAREPOINT_USERNAME")
SHAREPOINT_PASSWORD = os.getenv("SHAREPOINT_PASSWORD")

vectorstore = None
conversation_chain = None
chat_history = []

FAISS_INDEX_PATH = 'faiss_index.index'

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n', ' ', ''],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_vectorstore(vectorstore):
    faiss.write_index(vectorstore.index, FAISS_INDEX_PATH)

def load_vectorstore():
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        return FAISS(index=index, embedding_function=None, docstore=None, index_to_docstore_id=None)
    return None

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

@app.route('/')
def index():
    return render_template('index_sp.html')

async def process_documents_async(pdf_docs):
    global vectorstore, conversation_chain
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    save_vectorstore(vectorstore)
    conversation_chain = get_conversation_chain(vectorstore)

def fetch_pdfs_from_sharepoint():
    pdf_docs = []
    ctx = ClientContext(SHAREPOINT_SITE_URL).with_credentials(UserCredential(SHAREPOINT_USERNAME, SHAREPOINT_PASSWORD))
    folder = ctx.web.lists.get_by_title(SHAREPOINT_FOLDER).root_folder
    files = folder.files
    ctx.load(files)
    ctx.execute_query()

    for file in files:
        if file.name.endswith('.pdf'):
            pdf_stream = file.open_binary()
            pdf_docs.append(pdf_stream)
    
    return pdf_docs

@app.route('/process', methods=['POST'])
def process_documents():
    pdf_docs = fetch_pdfs_from_sharepoint()  # Fetch PDFs from SharePoint
    asyncio.run(process_documents_async(pdf_docs))
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history

    if request.method == 'POST':
        if conversation_chain is None:
            return redirect('/')  # Redirect to process documents first
        user_question = request.form['user_question']
        
        response = conversation_chain({'question': user_question})
        print("Response from conversation chain:", response)
        bot_response = response.get('response') or response.get('answer') or 'No response found'
        chat_history.append({'user': user_question, 'bot': bot_response})

    return render_template('chat.html', chat_history=chat_history)

# Load the vectorstore when the app starts
vectorstore = load_vectorstore()
if vectorstore is not None:
    conversation_chain = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    app.run(debug=True)
