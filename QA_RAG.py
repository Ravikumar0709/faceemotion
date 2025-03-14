import os
import logging
import faiss
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, session
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session

# Set up the upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Set up logging
logging.basicConfig(level=logging.INFO)

# FAISS Paths (for storing the vector index)
VECTORSTORE_PATH = 'faiss_index.index'

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

def generate_embeddings(text_chunks):
    """
    Create embeddings using Google's API (or any other pre-trained model).
    """
    embeddings = genai.embeddings(
        model="gemini-1.5-flash",
        texts=text_chunks
    )
    return embeddings

def get_vectorstore(text_chunks):
    """
    Creates a FAISS vector store and saves it as a persistent index file.
    """
    embeddings = generate_embeddings(text_chunks)
    
    # Create a FAISS vector store from embeddings
    faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
    
    # Add embeddings to FAISS index
    faiss_index.add(np.array(embeddings))
    
    # Save FAISS index to file
    faiss.write_index(faiss_index, VECTORSTORE_PATH)
    
    return FAISS(index=faiss_index)

def load_vectorstore():
    """
    Load an existing vector store from file if available.
    """
    if os.path.exists(VECTORSTORE_PATH):
        index = faiss.read_index(VECTORSTORE_PATH)
        return FAISS(index=index)
    return None

def get_conversation_chain(vectorstore):
    """
    Create the conversational chain for answering questions based on vector store.
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=genai.ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def generate_test_cases_with_llm(requirements_text):
    """
    Use a large language model to generate test cases from the business requirement text.
    """
    try:
        prompt = f"Generate a set of detailed test cases based on the following business requirements:\n{requirements_text}\n"
        response = genai.chat(model="gemini-1.5-flash", messages=[{"role": "user", "content": prompt}])
        test_case_text = response['choices'][0]['message']['content']
        test_cases = test_case_text.split('\n')
        
        structured_test_cases = []
        for i, test_case in enumerate(test_cases):
            if test_case.strip():
                structured_test_cases.append({
                    "test_case": f"Test case {i + 1}",
                    "description": test_case.strip(),
                    "expected_result": "Pass"
                })
        
        return structured_test_cases
    except Exception as e:
        logging.error(f"Error generating test cases: {str(e)}")
        return []

def save_test_cases_as_html(test_cases):
    """
    Save the test cases to an HTML file using a Jinja2 template.
    """
    template_str = """
    <html>
    <head>
        <title>Generated Test Cases</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Generated Test Cases</h1>
        <table>
            <tr><th>Test Case</th><th>Description</th><th>Expected Result</th></tr>
            {% for test_case in test_cases %}
                <tr>
                    <td>{{ test_case.test_case }}</td>
                    <td>{{ test_case.description }}</td>
                    <td>{{ test_case.expected_result }}</td>
                </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """
    template = Template(template_str)
    html_output = template.render(test_cases=test_cases)

    output_file = 'generated_test_cases.html'
    with open(output_file, 'w') as file:
        file.write(html_output)

    return output_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'business_req_doc' not in request.files:
        return redirect('/')
    
    file = request.files['business_req_doc']
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract text from the uploaded PDF
        raw_text = get_pdf_text([file])

        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)

        # Create or load the vector store
        vectorstore = load_vectorstore()
        if vectorstore is None:
            vectorstore = get_vectorstore(text_chunks)

        # Generate test cases using the LLM
        test_cases = generate_test_cases_with_llm(raw_text)

        # Save test cases as HTML
        html_file_path = save_test_cases_as_html(test_cases)

        # Send the generated HTML file back as a download
        return send_file(html_file_path, as_attachment=True)

    return redirect('/')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    # Retrieve chat history from session
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        # Check if vectorstore exists
        vectorstore = load_vectorstore()
        if vectorstore is None:
            return redirect('/')  # Redirect to upload page if no documents are processed

        # Retrieve the question asked by the user
        user_question = request.form['user_question']

        # Get the conversation chain for the chat
        conversation_chain = get_conversation_chain(vectorstore)

        # Get the response from the conversation chain
        response = conversation_chain({'question': user_question})

        # Extract the response text
        bot_response = response.get('response') or response.get('answer') or 'No response found'

        # Add to chat history
        chat_history.append({'user': user_question, 'bot': bot_response})
        
        # Save the chat history in the session
        session['chat_history'] = chat_history

    return render_template('chat.html', chat_history=chat_history)

@app.route('/clear_chat_history', methods=['POST'])
def clear_chat_history():
    # Clear the chat history stored in the session
    session['chat_history'] = []
    return redirect('/chat')

if __name__ == '__main__':
    app.run(debug=True)
