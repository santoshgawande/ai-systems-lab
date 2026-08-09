"""Convenience entrypoint: python -m scripts.ingest_samples"""
from app.ingest import ingest_all
from app import vectorstore

if __name__ == "__main__":
    for name, n in ingest_all().items():
        print(f"{name}: {n} chunks")
    print("Total:", vectorstore.stats())
