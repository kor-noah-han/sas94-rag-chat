from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a SAS documentation RAG assistant. "
    "Answer only from the supplied context. "
    "If the context contains syntax, examples, or procedural steps, answer directly from them. "
    "Only say the context is insufficient when the context truly does not support an answer. "
    "Always answer in Korean unless the user explicitly asks in another language. "
    "Cite supporting evidence inline with short labels such as [lepg p.12-13]. "
    "When answering a how-to question, start with the direct method first, then add short details."
)

SEARCH_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into SAS 9.4 documentation search queries. "
    "Do not answer the question. "
    "Return only a concise search query with SAS-specific English terms such as procedure names, "
    "statement names, graphics terms, or feature names. "
    "Prefer terms that would appear in SAS documentation headings. "
    "No markdown, no bullets, no explanation, no surrounding quotes."
)


def build_user_prompt(query: str, context: str) -> str:
    return (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Write a concise answer grounded in the context. "
        "When the context includes SAS syntax, include a short code example if helpful. "
        "Do not invent SAS behavior that is not present in the sources."
    )


def build_search_rewrite_prompt(
    query: str,
    *,
    expanded_terms: list[str] | None = None,
    top_sections: list[str] | None = None,
    family_hints: list[str] | None = None,
) -> str:
    lines = [
        f"Original question:\n{query}",
        (
            "Task:\nRewrite this as a short SAS 9.4 documentation search query. "
            "Include likely SAS procedure names, statements, options, or graphics terms when helpful."
        ),
    ]
    if expanded_terms:
        lines.append("Current expanded terms:\n" + ", ".join(expanded_terms[:12]))
    if top_sections:
        lines.append("Current weak search hits:\n" + "\n".join(top_sections[:5]))
    if family_hints:
        lines.append("Likely SAS terms to prefer:\n" + ", ".join(family_hints[:12]))
    lines.append(
        "Output requirements:\n"
        "- single line only\n"
        "- 4 to 14 keywords or short phrases\n"
        "- focused on search terms, not prose"
    )
    return "\n\n".join(lines)
