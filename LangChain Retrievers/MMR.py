# MMR (Maximum Marginal Relevance)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# Enable MMR in the retriever
retreiver = vector_store.as_retriever(
    search_type = 'mmr', # <-- This enables MMR
    search_kwargs = {"k": 3, "lambda_mult": 0.5} # top k result , lambda_mult = relevance-diversity balance (0-1) (lower the value higher diversity and vice-versa)
)

query = "What is langchain?"
result = retreiver.invoke(query)

# print retrieved docs thorugh loop
for i,res in enumerate(result):
    print(f"\n---Result {i+1} ----")
    print(f"Content: \n {res.page_content}...")

