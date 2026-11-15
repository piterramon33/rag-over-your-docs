"""Build the index:  python ingest.py my_docs/"""
import sys

from rag import ingest

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "my_docs"
    print(f"indexed {ingest(folder)} chunks from {folder}")
