from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
template = PromptTemplate(
    template= "What is the capital city of {country}",
    input_variables=['country']
)
parser = StrOutputParser()
chain = template | model | parser
result = chain.invoke({"country": "Pakistan" })
print(result)