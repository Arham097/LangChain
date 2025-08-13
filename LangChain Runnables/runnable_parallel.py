from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Generate a Tweet about the topic {topic}",
    input_variables=['topic'] 
)
prompt2 = PromptTemplate(
    template = "Generate a LinkedIn Post about the topic {topic}",
    input_variables=['topic'] 
)
runnable_parallel = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser),
        'linkedIn' : RunnableSequence(prompt2, model, parser)
    }
)
topic = input("Enter a topic: ")  
result = runnable_parallel.invoke({'topic': topic})
print("Tweet: ",result['tweet'])
print("\nLinkedIn: ", result['linkedIn'])