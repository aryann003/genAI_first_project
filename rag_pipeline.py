
def retrieve_documents(retriever, query):
    """Retrieve relevant documents for a user query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return retriever.invoke(query)


def build_context(documents):
    """Combine retrieved document content into context."""
    return "\n".join(
        document.page_content for document in documents
    )


def generate_answer(llm, prompt, context, question):
    """Generate an answer using retrieved context."""
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)

    return response.content


def answer_question(retriever, llm, prompt, query):
    """Execute the complete retrieval and answer-generation flow."""
    documents = retrieve_documents(retriever, query)
    context = build_context(documents)

    answer = generate_answer(
        llm,
        prompt,
        context,
        query
    )

    return {
        "answer": answer,
        "retrieved_documents": documents,
        "context": context
    }
