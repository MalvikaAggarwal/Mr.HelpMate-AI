from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_and_group_sections(text):
    # Group by parts like "PART I - DEFINITIONS", "PART IV - BENEFITS", etc.
    section_pattern = r"(PART [IVXL]+ - .+?)(?=PART [IVXL]+ -|\Z)"
    matches = re.findall(section_pattern, text, re.DOTALL)
    return matches if matches else [text]

def load_and_split(pdf_path):
    loader = PyPDFLoader("/home/mobiledairy/Mr.HelpMate/Mr.HelpMate-AI/data/Principal-Sample-Life-Insurance-Policy.pdf")
    docs = loader.load()

    # Merge all pages to capture full structured content
    full_text = "\n".join(doc.page_content for doc in docs)
    sections = clean_and_group_sections(full_text)

    # Wrap each section into a Document with metadata
    section_docs = [
        Document(page_content=section.strip(), metadata={"section": f"Part {i+1}"})
        for i, section in enumerate(sections)
    ]

    # Chunk semantically for RAG using LangChain's splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(section_docs)
