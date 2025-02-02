import json
import os

DOCUMENT_TEXT_PATH = "data/document_text.json"
DOCUMENT_NAMES_PATH = "data/document_names.json"

def load_documents():
    if not os.path.exists(DOCUMENT_TEXT_PATH):
        raise FileNotFoundError(f"Documents text file not found at {DOCUMENT_TEXT_PATH}")
    
    with open(DOCUMENT_TEXT_PATH, "r") as f:
        documents = json.load(f)
    return documents

def load_document_names():
    if not os.path.exists(DOCUMENT_NAMES_PATH):
        raise FileNotFoundError(f"Document names file not found at {DOCUMENT_NAMES_PATH}")
    
    with open(DOCUMENT_NAMES_PATH, "r") as f:
        document_names = json.load(f)
    return document_names
