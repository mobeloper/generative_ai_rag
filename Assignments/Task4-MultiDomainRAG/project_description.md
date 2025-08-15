
Assignment Prompt: Multi-Domain RAG Assistant with Router Chain


Master Prompt:

Create an AI-Powered Luxury Hotel Assistant chatbot. Please provide step by step instructions for the entire workflow: setting up the programming environment, processing text documents, creating text vector representations, and building a question-answering system. 




Scenario

You are building a Concierge AI Assistant for a luxury hotel. This assistant must answer guest questions about:

Dining: Restaurants, menus, opening hours, cuisine

Rooms: Room types, amenities, check-in/out, housekeeping

Wellness: Spa, gym, pool, yoga classes, wellness packages

Your goal is to create a RAG (Retrieval-Augmented Generation) Application that can route user questions to the correct domain-specific retriever using a router chain (MultiPromptChain) and respond with an accurate, contextually grounded answer.



Tasks

Prepare Data: You will create three text files (dining.txt, rooms.txt, wellness.txt) each containing information for the three domains above.

Load and Split Documents: Load and split each domain's data using TextLoader and RecursiveCharacterTextSplitter.

Create Embeddings & Vector Stores: Generate a FAISS vector store for each domain.

Build Domain-Specific Retrievers: Set up one retriever per domain.

Implement a Router Chain: Route questions to the right retriever using a router chain (e.g., MultiPromptChain).

Query the System: Test with at least 2 queries per domain. Print responses.
