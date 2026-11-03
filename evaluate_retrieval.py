
import json

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def evaluate_retrieval(retriever, dataset):
    hits = 0

    for item in dataset:
        question = item["question"]
        expected_keywords = [
            keyword.lower()
            for keyword in item["expected_keywords"]
        ]

        documents = retriever.invoke(question)

        retrieved_text = " ".join(
            doc.page_content.lower()
            for doc in documents
        )

        matched = [
            keyword
            for keyword in expected_keywords
            if keyword in retrieved_text
        ]

        is_hit = len(matched) > 0

        if is_hit:
            hits += 1

        print(f"\nQuestion: {question}")
        print(f"Matched keywords: {matched}")
        print(f"Hit: {is_hit}")

    hit_rate = hits / len(dataset) if dataset else 0

    print("\n--- Retrieval Evaluation ---")
    print(f"Total questions: {len(dataset)}")
    print(f"Successful hits: {hits}")
    print(f"Hit Rate@4: {hit_rate:.2%}")


if __name__ == "__main__":
    with open(
        "evaluation/evaluation_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    retriever = load_retriever()
    evaluate_retrieval(retriever, dataset)
