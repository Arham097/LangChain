import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import time

# Load env variables
load_dotenv()

# 🎨 Streamlit page config
st.set_page_config(
    page_title="🎥 YouTube AI Summarizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Sidebar ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)
    st.title("🎥 YouTube AI Summarizer")
    st.markdown("AI-powered summarization of YouTube videos using LangChain + Gemini.")
    st.markdown("---")
    video_id = st.text_input("🔗 Enter YouTube Video ID", "Gfr50f6ZBvo")
    question = st.text_area("❓ Ask a Question about the Video", "Can you summarize the video?")
    run_button = st.button("🚀 Run Summarizer")

# ---- Main App ----
st.markdown(
    """
    <style>
        .stTextInput textarea, .stTextArea textarea {
            border-radius: 12px;
            border: 2px solid #FF4B4B;
        }
        .result-box {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            padding: 20px;
            border-radius: 15px;
            color: black;
            font-size: 18px;
            font-weight: 500;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        }
    </style>
    """,
    unsafe_allow_html=True
)

if run_button:
    with st.spinner("📡 Fetching transcript..."):
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id)
            transcript = " ".join(chunk.text for chunk in transcript_list)
        except TranscriptsDisabled:
            st.error("🚫 No captions available for this video.")
            st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

    # model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    model = ChatGroq(model='meta-llama/llama-4-maverick-17b-128e-instruct')
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def generate_final_context(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel(
        {
            'context': retriever | RunnableLambda(generate_final_context),
            'question': RunnablePassthrough()
        }
    )
    parser = StrOutputParser()
    final_chain = parallel_chain | prompt | model | parser

    with st.spinner("🤖 Generating answer..."):
        result = final_chain.invoke(question)
        time.sleep(1)  # just to show spinner effect

    st.success("✅ Answer Generated!")
    st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)

    # 🎬 Extra section: show transcript preview
    with st.expander("📜 View Transcript Preview"):
        st.write(transcript[:2000] + "...")

    # 🎥 Embedded video
    st.markdown("---")
    st.markdown("### ▶️ Watch the Video")
    st.video(f"https://www.youtube.com/watch?v={video_id}")
