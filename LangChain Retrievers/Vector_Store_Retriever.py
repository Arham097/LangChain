from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vector store in memory
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name='sample'
)

# Step 4: Convert vectorstore into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k":2})

# Step 5: Define Query and invoke retriever
query = "What is Chroma used for?"
docs = retriever.invoke(query)

# print retrieved docs thorugh loop
for i,doc in enumerate(docs):
    print(f"\n---Result {i+1} ----")
    print(f"Content: \n {doc.page_content}...")
