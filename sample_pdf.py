
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "document loaders/deeplearning.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

total_pages = len(documents)

print("Total pages:", total_pages)

# Select 10 pages spread across the PDF
if total_pages <= 10:
    selected_indices = list(range(total_pages))
else:
    selected_indices = [
        round(i * (total_pages - 1) / 9)
        for i in range(10)
    ]

for index in selected_indices:
    document = documents[index]

    print("\n" + "=" * 70)
    print(f"PAGE: {index + 1}")
    print("=" * 70)

    print(document.page_content[:1500])
