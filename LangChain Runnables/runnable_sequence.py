from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Genrate a Joke about on the following {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = "Explain Joke about the following text \n {text}",
    input_variables=['text']
)
runnable = RunnableSequence(prompt1, model , parser, prompt2, model, parser)
topic = input("Enter a topic: ")
result = runnable.invoke({'topic': topic})
print(result)
