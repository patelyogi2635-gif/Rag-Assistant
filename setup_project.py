# setup_project.py
# ============================================================
# Run this ONCE from the project root to create all missing
# __init__.py files and verify the structure is correct.
#
# Usage:
#   python setup_project.py
# ============================================================

import os
from pathlib import Path

ROOT = Path(__file__).parent

PACKAGE_DIRS = [
    "config",
    "core",
    "core/ingestion",
    "core/vectorstore",
    "core/rag",
    "models",
    "utils",
]

DATA_DIRS = [
    "data/uploads",
    "data/chroma_db",
    "data/sample_docs",
]

def create_init_files():
    print("\n📦 Creating __init__.py files...")
    for d in PACKAGE_DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        init = path / "__init__.py"
        if not init.exists():
            init.touch()
            print(f"  ✅ Created {init.relative_to(ROOT)}")
        else:
            print(f"  ✓  Already exists: {init.relative_to(ROOT)}")

def create_data_dirs():
    print("\n📁 Creating data directories...")
    for d in DATA_DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {path.relative_to(ROOT)}")

def verify_structure():
    print("\n🔍 Verifying project structure...")
    expected_files = [
        "config/__init__.py",
        "config/settings.py",
        "core/__init__.py",
        "core/ingestion/__init__.py",
        "core/ingestion/pdf_loader.py",
        "core/ingestion/chunker.py",
        "core/ingestion/embedder.py",
        "core/ingestion/pipeline.py",
        "core/vectorstore/__init__.py",
        "core/vectorstore/chroma_store.py",
        "core/rag/__init__.py",
        "core/rag/chain.py",
        "core/rag/retriever.py",
        "core/rag/prompts.py",
        "models/__init__.py",
        "models/schemas.py",
        "utils/__init__.py",
        "utils/logger.py",
        "utils/file_utils.py",
        "main.py",
        ".env.example",
        "requirements.txt",
    ]

    all_ok = True
    for f in expected_files:
        exists = (ROOT / f).exists()
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {f}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n🎉 All files present. Run: python main.py")
    else:
        print("\n⚠️  Some files are missing — re-download them from the chat.")

if __name__ == "__main__":
    create_init_files()
    create_data_dirs()
    verify_structure()