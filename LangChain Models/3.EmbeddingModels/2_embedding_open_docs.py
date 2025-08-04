from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

documents = [
    "Islamabad is the capital of Pakistan",
    "Dhaka is the capital of Bangladesh",
    "New Delhi is the capital of India"
]

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_documents(documents)
print(str(result))