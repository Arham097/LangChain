from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Determine the user's feedback is either positive or negative")

parser = PydanticOutputParser(pydantic_object=Feedback) 
parser2 = StrOutputParser()

template1= PromptTemplate(
    template = "Classify the following feedback as positive or negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)
template2 = PromptTemplate(
    template= "Write an appropriate formal response to this positive feedback \n {feedback}",
    input_variables=['feedback']   
)
template3 = PromptTemplate(
    template= "Write an appropriate formal response to this negative feedback \n {feedback}",
    input_variables=['feedback']   
)
classifier_chain = template1 | model | parser
branch_chain = RunnableBranch(
    (lambda x: x.sentiment=="positive", template2| model| parser2),
    (lambda x: x.sentiment=="negative", template3| model| parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)
chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "This is a beutiful smartphone, and battery timing is great"})
print(result)