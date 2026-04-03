import os
import yaml
from datasets import load_dataset

class DataManager:
    @staticmethod
    def extract_and_clean_pdf(pdf_path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Part 01 requires 'pypdf'.")
        
        full_text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Normalize whitespace but keep essential structure
                full_text += " ".join(text.split()) + " "
        return full_text

    @staticmethod
    def get_chunks(text: str, config: dict):
        """Replaced with Recursive splitter for better semantic flow."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError("Part 01 requires 'langchain-text-splitters'.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunking']['size'], 
            chunk_overlap=config['chunking']['overlap'],
            separators=["\n\n", "\n", ".", " ", ""] # Priority splitting
        )
        return splitter.split_text(text)