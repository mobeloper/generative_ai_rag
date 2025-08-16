# app.py

# ==============================================================================
# Step 1: Import essential tools and set up the OpenAI API environment
# ==============================================================================
import os
import json # Import the json module to parse the router's output
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain 
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

from langchain_community.document_loaders import UnstructuredURLLoader 

# os.system('pip install langchain_community unstructured "unstructured[html]"')

# ==============================================================================
# Step 2: Set up the OpenAI API Key
# ==============================================================================
# Use dotenv to load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize the LLM and Embeddings model
# Setting temperature to 0 for more consistent responses
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

# ==============================================================================
# Step 3: Prepare Data from URLs
# ==============================================================================
# Define the URLs for each domain
urls = {
    "dining": "https://www.notion.so/eric-michel/dining-251a3168f4d080d9b4a0e626fe9e8d9c",
    "rooms": "https://www.notion.so/eric-michel/rooms-251a3168f4d08090be6cdc607f3b7720",
    "wellness": "https://www.notion.so/eric-michel/wellness-251a3168f4d0800bbc51e57865cd5312"
}

# Ingest data from the three specified URLs
print("Loading and splitting documents from URLs...")
try:
    # # Use WebBaseLoader to load content from the URLs
    # loader = WebBaseLoader(list(urls.values()))
    
    # Use UnstructuredURLLoader to handle dynamic content
    loader = UnstructuredURLLoader(urls=list(urls.values()))
    
    all_docs = loader.load()
    print(f"Loaded {len(all_docs)} raw documents from the URLs.")

    if not all_docs:
        print("No documents were loaded. This may be due to a network or URL issue.")
        exit()


    print("\n--- Successfully Loaded Document Content ---")
    for doc in all_docs:
        # The 'source' metadata field contains the URL from which the text was loaded.
        print(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
        print("Content:")
        # The 'page_content' field holds the actual text of the document.
        print(doc.page_content)
        print("-" * 50)
    
    print("Data loading and printing complete.")

    # Split the documents for each domain
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Filter documents based on URL to assign them to the correct domain
    dining_docs = text_splitter.split_documents([doc for doc in all_docs if "dining" in doc.metadata.get("source", "")])
    rooms_docs = text_splitter.split_documents([doc for doc in all_docs if "rooms" in doc.metadata.get("source", "")])
    wellness_docs = text_splitter.split_documents([doc for doc in all_docs if "wellness" in doc.metadata.get("source", "")])
    
    print(f"Loaded {len(dining_docs)} chunks for dining.")
    print(f"Loaded {len(rooms_docs)} chunks for rooms.")
    print(f"Loaded {len(wellness_docs)} chunks for wellness.")
    
except Exception as e:
    print(f"An error occurred while loading data from the URLs: {e}")
    exit()
    
    
    

# ==============================================================================
# Step 4: Create Embeddings and Vector Stores for each domain
# ==============================================================================
# Create a dictionary of FAISS vector stores directly from the loaded documents.
print("Creating FAISS vector stores for all domains...")
vector_stores = {
    "dining": FAISS.from_documents(dining_docs, embeddings),
    "rooms": FAISS.from_documents(rooms_docs, embeddings),
    "wellness": FAISS.from_documents(wellness_docs, embeddings)
}
print("FAISS vector stores created successfully.")

# ==============================================================================
# Step 5: Build Domain-Specific Retrievers and Prompts
# ==============================================================================
# Set up a retriever for each vector store.
retrievers = {
    "dining": vector_stores["dining"].as_retriever(),
    "rooms": vector_stores["rooms"].as_retriever(),
    "wellness": vector_stores["wellness"].as_retriever()
}

# Define prompt templates for each domain.
dining_template = """
You are a concierge AI assistant for a luxury hotel, specializing in dining.
Answer the user's question based ONLY on the following context. If the answer is not
in the context, state that you cannot provide information on that topic.

Context:
{context}

Question:
{input}
"""

rooms_template = """
You are a concierge AI assistant for a luxury hotel, specializing in rooms and hotel policies.
Answer the user's question based ONLY on the following context. If the answer is not
in the context, state that you cannot provide information on that topic.

Context:
{context}

Question:
{input}
"""

wellness_template = """
You are a concierge AI assistant for a luxury hotel, specializing in wellness and fitness.
Answer the user's question based ONLY on the following context. If the answer is not
in the context, state that you cannot provide information on that topic.

Context:
{context}

Question:
{input}
"""

# Define a prompt for the router chain to help it decide which domain to use.
# The prompt now explicitly asks for 'next_inputs' as a string.
router_template = """
Given a user's question, determine the most relevant domain to route it to.
The available domains are:
1. dining: For questions about restaurants, menus, and dining hours.
2. rooms: For questions about room types, amenities, and hotel policies like check-in/out.
3. wellness: For questions about the spa, gym, pool, and yoga classes.
If the question does not fit any of the domains, categorize it as "default".

Respond with a single JSON object. The JSON object should have two keys: 'destination' and 'next_inputs'. The value of 'destination' should be the name of the most relevant domain (dining, rooms, wellness) or 'default' if none apply. The value of 'next_inputs' should be the original user question as a string.

Example JSON:
{{
  "destination": "rooms",
  "next_inputs": "What time is check-in?"
}}

Question: {input}
Response:
"""

# ==============================================================================
# Step 6: Implement a Router Chain (using modern LCEL approach)
# ==============================================================================
# Define the destination chains using the modern approach.
dining_doc_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(dining_template))
rooms_doc_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(rooms_template))
wellness_doc_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(wellness_template))

# Create the retrieval chains for each domain
dining_chain = create_retrieval_chain(retrievers["dining"], dining_doc_chain)
rooms_chain = create_retrieval_chain(retrievers["rooms"], rooms_doc_chain)
wellness_chain = create_retrieval_chain(retrievers["wellness"], wellness_doc_chain)

# Create the default chain for non-domain questions using LCEL
default_prompt = ChatPromptTemplate.from_template(
    "You are a general concierge AI assistant. You cannot provide information about specific hotel policies, dining, or wellness services. Please state that you can only answer general questions. User's question: {input}"
)
default_chain = default_prompt | llm

# Create the router chain. It's a runnable sequence: prompt -> llm -> parse JSON
router_prompt = PromptTemplate(template=router_template, input_variables=["input"])
router_chain = router_prompt | llm | RunnableLambda(lambda x: json.loads(x.content))

# Use a RunnableBranch to route the requests based on the router's output.
# The router's output is expected to be a JSON object with a 'destination' key.
# We will use this key to route to the correct chain.
full_chain = (
    RunnablePassthrough.assign(
        route=router_chain,
    )
    | RunnableBranch(
        (lambda x: x["route"]["destination"] == "dining", dining_chain),
        (lambda x: x["route"]["destination"] == "rooms", rooms_chain),
        (lambda x: x["route"]["destination"] == "wellness", wellness_chain),
        default_chain,
    )
)

# Initialize a chat history list for the Flask app.
chat_history = []

# ==============================================================================
# Step 7: Use Flask to build a modern and beautiful chatbot interface
# ==============================================================================
app = Flask(__name__)

# HTML template string for a clean, responsive UI with Tailwind CSS
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luxury Hotel Concierge AI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white shadow-xl rounded-2xl w-full max-w-2xl overflow-hidden flex flex-col h-[80vh]">
        <div class="bg-emerald-600 text-white p-4 flex items-center justify-between shadow-md">
            <h1 class="text-xl font-bold">Luxury Hotel Concierge AI</h1>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        </div>
        <div id="chat-history" class="flex-1 p-4 overflow-y-auto space-y-4">
            <div class="flex justify-start">
                <div class="bg-gray-200 text-gray-800 p-3 rounded-xl max-w-sm">
                    <p>Hello! I'm your Concierge AI Assistant. How can I help you today?</p>
                </div>
            </div>
        </div>
        <form id="chat-form" class="bg-gray-200 p-4 flex items-center">
            <input type="text" id="user-input" class="flex-1 p-3 rounded-full border border-gray-300 focus:outline-none focus:ring-2 focus:ring-emerald-500" placeholder="Ask a question about dining, rooms, or wellness...">
            <button type="submit" class="ml-2 bg-emerald-600 text-white p-3 rounded-full hover:bg-emerald-700 transition duration-300">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
            </button>
        </form>
        <!-- Footer with author credit -->
        <div class="bg-gray-200 p-2 text-center text-gray-500 text-sm shadow-inner rounded-b-2xl">
            <p>Created by: Eric Michel</p>
            <a href="https://www.linkedin.com/in/ericmichelcv/" target="_blank" class="text-emerald-600 hover:underline">LinkedIn</a>
        </div>
    </div>
    <script>
        document.getElementById('chat-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const userInput = document.getElementById('user-input');
            const userMessage = userInput.value;
            if (userMessage.trim() === '') return;

            const chatHistory = document.getElementById('chat-history');

            // Display user message
            const userDiv = document.createElement('div');
            userDiv.className = 'flex justify-end';
            userDiv.innerHTML = `
                <div class="bg-emerald-500 text-white p-3 rounded-xl max-w-sm">
                    <p>${userMessage}</p>
                </div>
            `;
            chatHistory.appendChild(userDiv);

            // Display loading indicator
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'flex justify-start';
            loadingDiv.innerHTML = `
                <div class="bg-gray-200 text-gray-800 p-3 rounded-xl max-w-sm">
                    <div class="flex space-x-2 animate-pulse">
                        <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
                        <div class="w-2 h-2 bg-gray-400 rounded-full"></div>
                    </div>
                </div>
            `;
            chatHistory.appendChild(loadingDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;

            userInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: userMessage }),
                });

                const data = await response.json();

                // Remove loading indicator
                chatHistory.removeChild(loadingDiv);

                // Get the answer from the API response
                const assistantDiv = document.createElement('div');
                assistantDiv.className = 'flex justify-start';
                assistantDiv.innerHTML = `
                    <div class="bg-gray-200 text-gray-800 p-3 rounded-xl max-w-sm">
                        <p>${data.response}</p>
                    </div>
                `;
                chatHistory.appendChild(assistantDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            } catch (error) {
                console.error('Error:', error);
                chatHistory.removeChild(loadingDiv);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'flex justify-start';
                errorDiv.innerHTML = `
                    <div class="bg-red-200 text-red-800 p-3 rounded-xl max-w-sm">
                        <p>An error occurred. Please try again.</p>
                    </div>
                `;
                chatHistory.appendChild(errorDiv);
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    """Renders the main chatbot interface."""
    return render_template_string(html_template)

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint to handle user queries and return chatbot responses."""
    global chat_history
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"response": "Please enter a query."}), 400

    try:
        # Get the response from the router chain
        # The new chain takes a dictionary with 'input' and 'chat_history' as a list
        response = full_chain.invoke(
            {"input": user_query, "chat_history": chat_history}
        )
        
        # Add the new messages to the chat history for context in the next turn.
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response["answer"]))
        
        return jsonify({"response": response["answer"]})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"response": "An error occurred while processing your request."}), 500

if __name__ == "__main__":
    app.run(debug=True)
