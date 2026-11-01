
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG QA System",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Question Answering System")
st.write("Ask questions based on your uploaded knowledge base.")

# Load embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings_model = load_embeddings()


# Load Chroma database
@st.cache_resource
def load_vectorstore(_embeddings_model):
    return Chroma(
        persist_directory="chroma-db",
        embedding_function=_embeddings_model
    )


vectorstore = load_vectorstore(embeddings_model)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b"
    )


llm = load_llm()


# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the context provided.
    Use only the provided content to answer the questions.
    If answer is not in the content, say "I don't know" """),
    ("human", "{context} {question}")
])


# User input
query = st.text_input("Enter your question:")

if st.button("Get Answer"):

    if query.strip():

        with st.spinner("Searching documents and generating answer..."):

            docs = retriever.invoke(query)

            context = "\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)

        # Show retrieved documents
        with st.expander("View retrieved documents"):

            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i + 1}**")
                st.write(doc.page_content)

    else:
        st.warning("Please enter a question.")
