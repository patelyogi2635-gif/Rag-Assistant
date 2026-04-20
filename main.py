# main.py
# ============================================================
# Phase 1 — Demo Runner
# Groq (LLaMA 3.3 70B) + HuggingFace BGE embeddings
# ============================================================

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.ingestion.pipeline import IngestionPipeline
from retrieval.chain import RAGChain
from models.schemas import QueryRequest

console = Console()


def run_ingestion_demo(pdf_paths: list[Path]):
    console.rule("[bold blue]Phase 1 — PDF Ingestion[/bold blue]")

    pipeline = IngestionPipeline()
    result = pipeline.ingest(pdf_paths)

    table = Table(title="Ingestion Results", show_header=True, header_style="bold cyan")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")

    for r in result.processed:
        if r.was_duplicate:
            status = "[yellow]⚠️  DUPLICATE[/yellow]"
        elif r.error:
            status = f"[red]❌ ERROR: {r.error[:40]}[/red]"
        else:
            status = "[green]✅ OK[/green]"

        table.add_row(r.filename, status, str(r.total_pages), str(r.total_chunks))

    console.print(table)
    console.print(
        f"\n[bold]Total chunks added:[/bold] {result.total_chunks_added} "
        f"in {result.duration_seconds}s\n"
    )


def run_query_demo(questions: list[str]):
    console.rule("[bold blue]Phase 1 — RAG Queries (Groq LLaMA)[/bold blue]")

    chain = RAGChain()

    for question in questions:
        console.print(f"\n[bold yellow]❓ Q:[/bold yellow] {question}")

        response = chain.query(QueryRequest(question=question))

        console.print(
            Panel(
                response.answer,
                title=f"[bold green]Answer[/bold green] [dim](Groq · {response.model_used})[/dim]",
                border_style="green",
            )
        )

        if response.sources:
            console.print("[bold]📎 Sources:[/bold]")
            for s in response.sources:
                console.print(
                    f"  • [cyan]{s.source_file}[/cyan] | "
                    f"Page {s.page_number} | "
                    f"Score: {s.similarity_score:.3f}"
                )

        console.print(
            f"[dim]⏱  {response.duration_seconds}s | "
            f"{response.retrieval_count} chunks retrieved | "
            f"cache={'HIT' if response.from_cache else 'MISS'}[/dim]\n"
        )


if __name__ == "__main__":
    # ── Add your PDFs here ───────────────────────────────────
    SAMPLE_PDFS = [
        # Path("data/sample_docs/medical_report.pdf"),
        # Path("data/sample_docs/insurance_policy.pdf"),
    ]

    TEST_QUESTIONS = [
        "What is the main subject of the uploaded documents?",
        "Summarize the key points from the documents.",
    ]
    # ─────────────────────────────────────────────────────────

    console.print(
        Panel(
            "[bold]RAG Assistant — Phase 1: Foundation[/bold]\n\n"
            "LLM:        [cyan]Groq · LLaMA 3.3 70B[/cyan] (~500 tok/s)\n"
            "Embeddings: [cyan]BAAI/bge-base-en-v1.5[/cyan] (local, free)\n"
            "Vector DB:  [cyan]ChromaDB[/cyan] (persistent)\n",
            style="blue",
            title="🚀 Fully Open-Source RAG Stack",
        )
    )

    if SAMPLE_PDFS:
        run_ingestion_demo(SAMPLE_PDFS)
        run_query_demo(TEST_QUESTIONS)
    else:
        console.print(
            "[yellow]ℹ️  No PDFs configured in SAMPLE_PDFS. "
            "Add PDF paths to main.py to test ingestion.[/yellow]\n"
        )
        console.print(
            "[bold]Quick start:[/bold]\n\n"
            "  [dim]# 1. Ingest[/dim]\n"
            "  from core.ingestion.pipeline import IngestionPipeline\n"
            "  pipeline = IngestionPipeline()\n"
            "  pipeline.ingest([Path('your_doc.pdf')])\n\n"
            "  [dim]# 2. Query[/dim]\n"
            "  from core.rag.chain import RAGChain\n"
            "  from models.schemas import QueryRequest\n"
            "  chain = RAGChain()\n"
            "  r = chain.query(QueryRequest(question='What does this document say about X?'))\n"
            "  print(r.answer)\n"
        )