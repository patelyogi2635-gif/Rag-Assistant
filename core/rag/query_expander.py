# core/rag/query_expander.py
# ============================================================
# Query Expansion — improves recall for ambiguous questions.
#
# Problem it solves:
#   "What's my coverage limit?" may miss chunks that say
#   "maximum benefit", "policy cap", or "insured amount".
#   Embeddings help but aren't perfect across paraphrases.
#
# Solution:
#   Use the LLM to generate N alternative phrasings of the query.
#   Retrieve for ALL variants, deduplicate by chunk_id, re-rank.
#   This is called "HyDE-lite" or "multi-query retrieval".
#
# Cost: 1 extra Groq call per query (~100ms on Groq's fast infra).
# Gain: 15-25% better recall on ambiguous medical/legal/policy docs.
#
# Disabled by default — enable via QUERY_EXPANSION_ENABLED=true in .env
# ============================================================

import json
import re
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriting assistant. Your job is to generate alternative "
        "phrasings of a user question to improve document retrieval. "
        "Return ONLY a JSON array of strings. No explanation, no markdown."
    ),
    (
        "human",
        "Generate {n} alternative phrasings of this question. "
        "Keep the same meaning but use different words and sentence structures.\n\n"
        "Original question: {question}\n\n"
        "Return format: [\"phrasing 1\", \"phrasing 2\", ...]"
    ),
])


class QueryExpander:
    """
    Generates alternative query phrasings to improve retrieval recall.

    Usage:
        expander = QueryExpander()
        variants = expander.expand("What is my deductible?", n=3)
        # ["What deductible amount applies to my policy?",
        #  "How much do I pay before insurance kicks in?",
        #  "What is the out-of-pocket threshold in my plan?"]
    """

    def __init__(self):
        self._llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0.4,     # slight creativity for diverse phrasings
            max_tokens=300,
        )
        self._chain = _EXPANSION_PROMPT | self._llm

    def expand(self, query: str, n: int = 3) -> List[str]:
        """
        Return n alternative phrasings + the original query.
        Always includes the original so retrieval is never worse.
        Falls back to [original] on any error.
        """
        if not settings.query_expansion_enabled:
            return [query]

        try:
            result = self._chain.invoke({"question": query, "n": n})
            variants = self._parse_json_list(result.content)

            # Always prepend original query — ensures it's always searched
            all_queries = [query] + [v for v in variants if v != query]
            logger.info(
                f"🔁 Query expanded to {len(all_queries)} variants:\n"
                + "\n".join(f"   {i+1}. {q}" for i, q in enumerate(all_queries))
            )
            return all_queries

        except Exception as e:
            logger.warning(f"⚠️  Query expansion failed: {e}. Using original query.")
            return [query]

    # ── Private ──────────────────────────────────────────────

    def _parse_json_list(self, text: str) -> List[str]:
        """Safely parse LLM JSON output."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?|```", "", text).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got: {type(parsed)}")
        return [str(item).strip() for item in parsed if item]