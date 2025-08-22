from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

# Step 1: Indexing
# Step 1.1: Document Ingestion
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1.2: Text Splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20
)

chunks = splitter.create_documents([transcript])

# Step 1.3 & 1.4: Generate Embeddings and Creating Vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# Step 2 : Retrieval
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

# Step 3: Augmentation
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def generate_final_context(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel(
    {
        'context': retriever | RunnableLambda(generate_final_context),
        'question': RunnablePassthrough()
    }
)
parser = StrOutputParser()

final_chain = parallel_chain | prompt | model | parser

# Step 4: Generation
question = "Can you summarize the video"
result = final_chain.invoke(question)
print(result)