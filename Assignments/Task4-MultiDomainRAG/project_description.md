
Assignment Prompt: Multi-Domain RAG Assistant with Router Chain


Master Prompt:

Create an AI-Powered Luxury Hotel Assistant chatbot. Please provide step by step instructions for the entire workflow: setting up the programming environment, processing text documents, creating text vector representations, and building a question-answering system. 


Overview:
The project aims to create a conversational chatbot that responds to user inquiries using document information. It requires proficiency in extracting and converting text into numerical vectors, establishing an answer-finding mechanism, and designing a user-friendly chatbot interface with Flask. Additionally, the initiative emphasizes structuring inquiries for clear communication and deploying the chatbot for practical use, guaranteeing the system's accessibility and efficiency in meeting user needs.

Instructions:
• Read the sections on Scenario, task, action, and result carefully to understand the 
assignment. 
• Adhere closely to the provided guidelines, ensuring your output contains all 
necessary analyses and interpretations. 


Context:
As a developer, you have received the critical task of improving the operational efficiency of a luxury hotel. Your toolkit includes cutting-edge conversational AI technology, Python libraries, LangChain, the powerful GPT model from OpenAI, and the user-friendly Flask UI. Your mission is to integrate these advanced tools seamlessly to transform customer service processes, creating a more streamlined and efficient workflow within the hotel organization. 


Scenario:

You are building a Concierge AI Assistant for a luxury hotel. This assistant must answer guest questions about:

Dining: Restaurants, menus, opening hours, cuisine

Rooms: Room types, amenities, check-in/out, housekeeping

Wellness: Spa, gym, pool, yoga classes, wellness packages

Your goal is to create a RAG (Retrieval-Augmented Generation) Application that can route user questions to the correct domain-specific retriever using a router chain (MultiPromptChain) and respond with an accurate, contextually grounded answer.



Tasks:

Your task is to develop a conversational chatbot. This chatbot must answer queries about 
the hotel reports efficiently. Use Python libraries, LangChain, OpenAI's GPT model, and Flask UI. 
These tools will help you create a user-friendly interface. This interface will extract and 
process information from documents. It will provide accurate responses to user queries.


Action:
- Import essential tools and set up OpenAI's API environment. 

- Prepare Data: You will ingest three text files (./dining.txt, ./rooms.txt, ./wellness.txt) each containing information for the three domains above and split it in chunks for easy processing. 

- Load and Split Documents: Load and split each domain's data using TextLoader and RecursiveCharacterTextSplitter.

- Create vector representations for text chunks using FAISS and OpenAI's 
embeddings. Create Embeddings & Vector Stores: Generate a FAISS vector store for each domain.

- Build a question-answering system using the gpt-4o-mini model to retrieve answers 
from text chunks.

- Create a prompt template to guide the chatbot in understanding and responding to 
users. 

- Build Domain-Specific Retrievers: Set up one retriever per domain.

- Implement a Router Chain: Route questions to the right retriever using a router chain (e.g., MultiPromptChain).

- Use Flask to build a modern and beautiful chatbot interface, enabling interaction and 
information retrieval. Ensure the interface is user-friendly to facilitate effective interaction and information retrieval. 

