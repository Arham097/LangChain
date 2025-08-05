from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

chat_template = ChatPromptTemplate(
    [
        ("system", "You are helpful {domain} expert"),
        ("human", "Explain in simpler terms, what is the {topic}")
    ]
)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
prompt = chat_template.invoke({"domain": "cricket", "topic": "Dusra"})
print(prompt, "\n\n")
result = model.invoke(prompt)
print(result.content)

