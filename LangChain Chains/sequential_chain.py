from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
template1 = PromptTemplate(
    template = "Generate a comprehensive report on the topic \n {topic}",
    input_variables = ['topic']
)
template2 = PromptTemplate(
    template = "Generate a 5 Points Summary on the following text \n {text}",
    input_variables=['text']
)
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'cricket'})
print (result)

chain.get_graph().print_ascii()