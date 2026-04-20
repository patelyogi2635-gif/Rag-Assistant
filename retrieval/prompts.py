# core/rag/prompts.py
# ============================================================
# Prompt templates — centralized so they're easy to iterate on.
# In Phase 2 we'll add a query-expansion prompt here too.
# ============================================================

from langchain_core.prompts import ChatPromptTemplate

# ── System Instruction ───────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, helpful document assistant.
Your job is to answer questions **strictly** based on the provided context.

Rules:
1. Answer ONLY from the context provided. Do not use external knowledge.
2. If the context doesn't contain enough information, say clearly:
   "I couldn't find a clear answer in the uploaded documents."
3. Always cite your sources by mentioning the document name and page number.
4. Be concise but complete. Use bullet points for lists.
5. Never fabricate or assume information not in the context."""

# ── RAG Answer Prompt ────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Context from uploaded documents:
{context}

---
Question: {question}

Answer (cite document name + page number for every claim):""",
        ),
    ]
)


def format_context(chunks) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM.
    Groups by source file for readability.
    """
    if not chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.source_file} | Page {chunk.page_number}\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(parts)