from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10,
    chunk_overlap = 0
)

text = """
My Name is Nitish
I am 35 years old

I live in Gurgaon
How are you
"""
result = splitter.split_text(text)
print(result)