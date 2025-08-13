from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Write a  detailed report on the following topic:  {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = "Summarize the following text: \n {text}",
    input_variables=['text']
)
def word_counter(text):
    return len(text.split())
initial_chain = RunnableSequence(prompt1, model, parser)
conditional_chain = RunnableBranch(
    (lambda x: word_counter(x) > 1000, RunnableSequence(prompt2, model, parser)), # if
    RunnablePassthrough() # else
)
final_chain = RunnableSequence(initial_chain, conditional_chain) 
topic = input("Enter A Topic: ")
result = final_chain.invoke({'topic': topic})
print(result)