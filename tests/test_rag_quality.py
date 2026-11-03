
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_pipeline import (
    retrieve_documents,
    build_context,
    generate_answer,
    answer_question
)


def test_retrieve_documents():
    retriever = Mock()
    expected_docs = [
        SimpleNamespace(page_content="Deep learning uses neural networks.")
    ]
    retriever.invoke.return_value = expected_docs

    result = retrieve_documents(retriever, "What is deep learning?")

    assert result == expected_docs
    retriever.invoke.assert_called_once_with("What is deep learning?")


def test_retrieve_documents_rejects_empty_query():
    retriever = Mock()

    with pytest.raises(ValueError):
        retrieve_documents(retriever, "   ")

    retriever.invoke.assert_not_called()


def test_build_context():
    documents = [
        SimpleNamespace(page_content="Document one."),
        SimpleNamespace(page_content="Document two.")
    ]

    result = build_context(documents)

    assert result == "Document one.\nDocument two."


def test_generate_answer():
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="Deep learning uses neural networks."
    )

    prompt = Mock()
    prompt.invoke.return_value = "formatted prompt"

    result = generate_answer(
        llm,
        prompt,
        "Deep learning context",
        "What is deep learning?"
    )

    assert result == "Deep learning uses neural networks."

    prompt.invoke.assert_called_once_with({
        "context": "Deep learning context",
        "question": "What is deep learning?"
    })

    llm.invoke.assert_called_once_with("formatted prompt")


def test_answer_question():
    retriever = Mock()
    retriever.invoke.return_value = [
        SimpleNamespace(page_content="Deep learning uses neural networks.")
    ]

    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(
        content="Deep learning uses neural networks."
    )

    prompt = Mock()
    prompt.invoke.return_value = "formatted prompt"

    result = answer_question(
        retriever,
        llm,
        prompt,
        "What is deep learning?"
    )

    assert result["answer"] == "Deep learning uses neural networks."
    assert result["context"] == "Deep learning uses neural networks."
    assert len(result["retrieved_documents"]) == 1
