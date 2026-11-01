
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs = [
    Document(page_content="iPhone 16 has a 48MP camera with excellent image processing.", metadata={"id": "D1"}),
    Document(page_content="Samsung S24 has a 50MP camera and advanced night photography.", metadata={"id": "D2"}),
    Document(page_content="Google Pixel 9 offers excellent computational photography and portrait mode.", metadata={"id": "D3"}),
    Document(page_content="iPhone 15 offers excellent camera quality and image stabilization.", metadata={"id": "D4"}),
    Document(page_content="Samsung phones offer powerful processors for gaming and multitasking.", metadata={"id": "D5"}),
    Document(page_content="OnePlus 12 has a versatile camera system and fast charging.", metadata={"id": "D6"}),
    Document(page_content="Sony headphones provide excellent noise cancellation and sound quality.", metadata={"id": "D7"}),
    Document(page_content="DSLR cameras offer manual controls and interchangeable lenses for professional photography.", metadata={"id": "D8"})
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    docs,
    embeddings
)

query = "I want a good smartphone for photography"

# Similarity Search
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

similarity_docs = similarity_retriever.invoke(query)

print("\n--- Similarity Search ---")

for doc in similarity_docs:
    print(doc.metadata["id"], ":", doc.page_content)


# MMR Search
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5
    }
)

mmr_docs = mmr_retriever.invoke(query)

print("\n--- MMR Search ---")

for doc in mmr_docs:
    print(doc.metadata["id"], ":", doc.page_content)
