from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
parser = StrOutputParser()
prompt = PromptTemplate(
    template = "Genrate a detailed reporton the following {topic}",
    input_variables=['topic']
)
def word_counter(text):
    return len(text.split())
initial_chain = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel(
    {
        'report': RunnablePassthrough(),
        'words': RunnableLambda(word_counter)
    }
)
topic = input("Enter a topic:")
final_chain = RunnableSequence(initial_chain, parallel_chain)
result = final_chain.invoke({'topic': topic})
print(result)
