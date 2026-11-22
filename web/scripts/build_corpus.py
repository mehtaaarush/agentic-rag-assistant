"""Chunk the paper PDFs for the web demo, the same way src/ingest.py does.

Each PDF page becomes a document (pypdf text, metadata source and page), split with
RecursiveCharacterTextSplitter into 1,000-character chunks with 200 characters of overlap.

    python web/scripts/build_corpus.py path/to/papers/

Writes web/lib/corpus.json (chunk text, kept on the server) and web/public/data/papers.json.
Run web/scripts/embed.mjs afterwards to embed the chunks.
"""
import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

WEB = Path(__file__).resolve().parents[1]

# arXiv id and title for each file in the corpus.
PAPERS = {
    "attention_is_all_you_need": ("1706.03762", "Attention Is All You Need", 2017),
    "vision_transformer_vit": ("2010.11929", "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", 2020),
    "swin_transformer": ("2103.14030", "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", 2021),
    "swinir": ("2108.10257", "SwinIR: Image Restoration Using Swin Transformer", 2021),
    "dinov2": ("2304.07193", "DINOv2: Learning Robust Visual Features without Supervision", 2023),
    "dino_self_supervised_vit": ("2104.14294", "Emerging Properties in Self-Supervised Vision Transformers", 2021),
    "masked_autoencoders_mae": ("2111.06377", "Masked Autoencoders Are Scalable Vision Learners", 2021),
    "generative_adversarial_networks": ("1406.2661", "Generative Adversarial Networks", 2014),
    "clip": ("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision", 2021),
    "resnet": ("1512.03385", "Deep Residual Learning for Image Recognition", 2015),
    "loftr": ("2104.00680", "LoFTR: Detector-Free Local Feature Matching with Transformers", 2021),
    "ddpm_diffusion": ("2006.11239", "Denoising Diffusion Probabilistic Models", 2020),
}


def main() -> None:
    src = Path(sys.argv[1])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks, papers = [], []
    for stem, (arxiv, title, year) in PAPERS.items():
        reader = PdfReader(src / f"{stem}.pdf")
        start = len(chunks)
        for page_no, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            for piece in splitter.split_text(text):
                if piece.strip():
                    chunks.append({"paper": len(papers), "page": page_no, "text": piece})
        papers.append({"file": f"{stem}.pdf", "arxiv": arxiv, "title": title, "year": year,
                       "pages": len(reader.pages), "chunks": len(chunks) - start})
        print(f"{stem:34s} {len(reader.pages):3d} pages {len(chunks) - start:4d} chunks")
    (WEB / "lib").mkdir(exist_ok=True)
    (WEB / "public" / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "lib" / "corpus.json").write_text(json.dumps({"papers": papers, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    (WEB / "public" / "data" / "papers.json").write_text(json.dumps(papers, indent=1), encoding="utf-8")
    print(len(chunks), "chunks from", len(papers), "papers")


if __name__ == "__main__":
    main()
