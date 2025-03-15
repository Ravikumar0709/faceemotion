import os
import logging
import faiss
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, session
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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


def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def load_vectorstore():
    """
    Load an existing vector store from file if available.
    """
    if os.path.exists(VECTORSTORE_PATH):
        index = faiss.read_index(VECTORSTORE_PATH)
        return FAISS(index=index)
    return None

import json
from jinja2 import Template

def generate_test_cases_with_llm(requirements_text):
    """
    Use a large language model to generate test cases from the business requirement text.
    """
    example_prompts = """
    Example 1: User Authentication System
    Test Case ID: TC_001
    Test Case Title: User Registration with Valid Email and Password
    Description: Ensure that users can successfully register with a valid email address and password.
    Expected Output: User should be registered and redirected to the login page.
    """

    try:
        # Construct the prompt for LLM, including the examples and additional instructions
        prompt = f"""
        {example_prompts}
        
        Instructions:
        1. Only include meaningful descriptions. All extraneous details, values like '|', and irrelevant logic should be omitted.
        2. Ensure the descriptions focus on testing the business requirements as part of the software SDLC cycle. Descriptions should be actionable and meaningful. 
           Make sure the test cases align closely with business rules and requirements with additional information with semantic meaning.
        3. Need to have Test Case, Test Case Title, Description, Expected Outcome in the model response output
        
        Business Requirements:
        {requirements_text}
        
        Now, based on the above business requirements, generate a set of detailed test cases, focusing on the business logic, conditions, and actions that should be tested.
        """

        # Make the API call to generate test cases
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-002')

        response = model.generate_content(prompt)
        print(response.text)
        # Extract test case content from the LLM response
        try:
            test_case_text = response.text
            # Log the raw extracted text for debugging
            logging.info(f"Extracted test case text: {test_case_text}")
        except KeyError as e:
            logging.error(f"Error extracting test case text: {e}")
            return json.dumps([])

        # Check if the extracted text is non-empty
        if not test_case_text.strip():
            logging.error("Generated test case content is empty.")
            return json.dumps([])

        # Clean the response by removing '*' or '**' and making the list readable
        import re
        cleaned_test_cases = [
            re.sub(r"(Test\s*Case)\s*\d+", r"\1", line.strip())  # Removes number after 'Test Case' (keeps 'Test Case')
            .replace("'", '')  # Remove single quotes from the text
            .replace('*', '')  # Remove asterisks
            .replace('**', '')  # Remove double asterisks
            for line in test_case_text.strip().split('\n') if line.strip()
        ]


        # Strip the quotes from the entire list
        cleaned_test_cases1 = [item.strip("'") for item in cleaned_test_cases]

        print(cleaned_test_cases1)


        # Log the cleaned test cases for verification
        logging.info(f"Cleaned test cases: {cleaned_test_cases1}")

        structured_test_cases = []
        for i, test_case in enumerate(cleaned_test_cases1):
            if test_case.strip():
                structured_test_cases.append({
                    "test_case_id": f"TC_00{i + 1}",
                    "test_case_title": f"Test Case {i + 1}",
                    "description": test_case.strip(),
                    "expected_output": "Pass"
                })

        # Log the structured test cases for debugging
        logging.info(f"Structured test cases: {structured_test_cases}")
        return json.dumps(structured_test_cases)

    except Exception as e:
        logging.error(f"Error generating test cases: {str(e)}")
        # If something goes wrong, return a default message
        return json.dumps([{
            "test_case_id": "TC_001",
            "test_case_title": "Default Test Case",
            "description": f"An error occurred while generating test cases: {str(e)}",
            "expected_output": "Pass"
        }])


def save_test_cases_as_html(test_cases_json):
    """
    Save the test cases from JSON string to an HTML file using a Jinja2 template.
    """
    try:
        # Parse the test cases JSON into Python objects
        test_cases = json.loads(test_cases_json)

        if not test_cases:
            logging.error("No test cases to generate HTML.")
            return None

        # Define the HTML template for rendering the test cases
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
                <tr><th>Test Case ID</th><th>Test Case Title</th><th>Description</th><th>Expected Output</th></tr>
                {% for test_case in test_cases %}
                    <tr>
                        <td>{{ test_case.test_case_id }}</td>
                        <td>{{ test_case.test_case_title }}</td>
                        <td>{{ test_case.description }}</td>
                        <td>{{ test_case.expected_output }}</td>
                    </tr>
                {% endfor %}
            </table>
        </body>
        </html>
        """

        # Use Jinja2 template to generate HTML
        template = Template(template_str)
        html_output = template.render(test_cases=test_cases)

        output_file = 'generated_test_cases.html'
        with open(output_file, 'w') as file:
            file.write(html_output)

        logging.info(f"HTML file generated at {output_file}")
        return output_file

    except Exception as e:
        # Log any exceptions while saving the HTML file
        logging.error(f"Error saving test cases as HTML: {str(e)}")
        return None


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

        if not test_cases or test_cases == "[]":
            logging.error("No valid test cases generated.")
            return redirect('/')  # Redirect if no test cases are generated

        # Save test cases as HTML
        html_file_path = save_test_cases_as_html(test_cases)

        if not html_file_path:
            logging.error("Failed to generate HTML file for test cases.")
            return redirect('/')  # Redirect if HTML generation fails

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
