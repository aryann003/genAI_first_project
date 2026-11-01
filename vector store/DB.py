
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is widely used in artificial intelligence and machine learning applications.",
        metadata={"source": "AI Article"}
    ),
    Document(
        page_content="Python's simplicity and readability make it a popular choice for beginners.",
        metadata={"source": "Programming Blog"}
    ),
    Document(
        page_content="Python has a large standard library that supports many common programming tasks.",
        metadata={"source": "Python Documentation"}
    ),
]

# Load local embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma vector database
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)


result = vectorstore.similarity_search("What is Python used for?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)


retriver = vectorstore.as_retriever()

docs = retriver.invoke("What is Python used for?")
