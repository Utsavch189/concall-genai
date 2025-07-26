# from docling.document_converter import DocumentConverter

# source = "reports/TCS/annual/AnnualReport2023.pdf"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)
# print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

from langchain_docling import DoclingLoader

FILE_PATH = "reports/TCS/announcements/M6Y2025D20.pdf"

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()
for d in docs[:3]:
    print(f"- {d.page_content=}")