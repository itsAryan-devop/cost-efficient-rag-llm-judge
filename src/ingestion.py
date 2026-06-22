import os
import hashlib
import re
from typing import List, Dict, Any
import pdfplumber
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".txt"}
MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u0080\u0094": " - ",
    "\u00e2\u0080\u0093": " - ",
    "\u00e2\u0080\u00a2": "- ",
    "\u00e2\u0086\u0092": "->",
    "\u00c2\u00b7": "-",
    "\u00e2\u0080\u0098": "'",
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
}

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def normalize_text(text: str) -> str:
    """Normalizes whitespace while preserving readable paragraph boundaries."""
    for bad, replacement in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, replacement)
    text = "".join(ch for ch in text if not (0x80 <= ord(ch) <= 0x9F))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    compact = "\n".join(line for line in lines if line)
    return compact.strip()

def compute_document_id(relative_path: str) -> str:
    return sha256_text(f"doc:v1:{relative_path}")

def compute_document_hash(text: str) -> str:
    return sha256_text(f"content:v1:{text}")

def compute_chunk_id(document_id: str, chunk_index: int, chunk_text: str) -> str:
    """Stable ID used to make re-ingestion idempotent."""
    key = f"chunk:v1:{document_id}:{settings.chunk_size}:{settings.chunk_overlap}:{chunk_index}:{chunk_text}"
    return sha256_text(key)

def extract_text_from_pdf(filepath: str) -> str:
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"\n[page {page_number}]\n{page_text}")
    return "\n".join(text_parts)

def extract_text_from_html(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text(separator='\n')

def extract_text_from_md(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def parse_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext in ['.html', '.htm']:
        return extract_text_from_html(filepath)
    elif ext == '.md':
        return extract_text_from_md(filepath)
    elif ext == ".txt":
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    raise ValueError(f"Unsupported file type: {ext}")

def process_documents(data_dir: str) -> List[Dict[str, Any]]:
    """
    Reads all documents in data_dir, chunks them, and returns a list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    chunks_data: List[Dict[str, Any]] = []
    
    if not os.path.exists(data_dir):
        return chunks_data

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, data_dir).replace("\\", "/")
            
            try:
                text = normalize_text(parse_file(filepath))
                if not text.strip():
                    continue

                document_id = compute_document_id(relative_path)
                document_hash = compute_document_hash(text)
                
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_id = compute_chunk_id(document_id, i, chunk)
                    chunks_data.append({
                        "id": chunk_id,
                        "document_id": document_id,
                        "document_hash": document_hash,
                        "text": chunk,
                        "metadata": {
                            "source_file": relative_path,
                            "chunk_index": i,
                            "doc_type": ext.lstrip("."),
                            "chunk_size": settings.chunk_size,
                            "chunk_overlap": settings.chunk_overlap,
                        }
                    })
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                
    return chunks_data
