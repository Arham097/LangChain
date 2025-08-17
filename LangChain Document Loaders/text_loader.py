from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt = PromptTemplate(
    template = "Summarize the given poem \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)