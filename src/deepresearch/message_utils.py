"""Shared message/text normalization helpers."""

from __future__ import annotations

from typing import Any


def extract_text_content(content: Any) -> str:
    """Extract plain text from LangChain/OpenAI-style content payloads."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if text is None:
                continue

            item_type = str(item.get("type") or "").strip().lower()
            if item_type and item_type != "text":
                continue

            chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return str(content)
