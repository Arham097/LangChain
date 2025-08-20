from langchain_community.retrievers import WikipediaRetriever

# initialize retriever

retreiver = WikipediaRetriever(top_k_results=2, lang='en')

# define query
query = "The Geopolitical history of India and Pakistan from the perspective of chinese"

# Get relevant wikipedia documents
docs = retreiver.invoke(query)

# print retrieved docs thorugh loop
for i,doc in enumerate(docs):
    print(f"\n---Result {i+1} ----")
    print(f"Content: \n {doc.page_content}...")
