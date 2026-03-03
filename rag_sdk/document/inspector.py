"""Chunk inspection utilities.

Lets you see exactly what a splitter produces from a set of documents — no
vector store, no embeddings, no LLM calls.  Pure text analysis.

Usage::

    from rag_sdk.document import TextSplitter, inspect_chunks

    splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
    report = inspect_chunks(docs, splitter)

    report.summary()       # aggregate stats + char-count histogram
    report.table()         # every chunk as a fixed-width table row
    report.detail(3)       # full content of chunk #3

    sub = report.for_source("Scott Derrickson")  # filter to one source doc
    sub.table()
"""

import math
import sys
from dataclasses import dataclass
from typing import Any, List

from .base import BaseTextSplitter
from .models import Document


def _out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


@dataclass
class ChunkInfo:
    """Details for a single chunk produced by a splitter."""

    index: int  # global index across all chunks (0-based)
    doc_index: int  # index of the source document in the input list
    chunk_index: int  # position of this chunk within its source document
    source: str  # metadata["source"] or "doc_<doc_index>"
    char_count: int
    word_count: int
    content: str  # full content

    def preview(self, width: int = 60) -> str:
        """Return the first ``width`` characters on a single line."""
        text = self.content.replace("\n", " ")
        if len(text) <= width:
            return text
        return text[: width - 1] + "\u2026"


@dataclass
class ChunkReport:
    """Aggregate view of all chunks produced from a document set."""

    source_documents: int
    chunks: List[ChunkInfo]

    # ── Aggregate properties ──────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def avg_chars(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.char_count for c in self.chunks) / len(self.chunks)

    @property
    def min_chars(self) -> int:
        return min((c.char_count for c in self.chunks), default=0)

    @property
    def max_chars(self) -> int:
        return max((c.char_count for c in self.chunks), default=0)

    # ── Display methods ───────────────────────────────────────────────────

    def table(self, preview_width: int = 55) -> None:
        """Print all chunks as a fixed-width table.

        Columns: # | Doc | Source | Chars | Words | Preview

        Args:
            preview_width: Number of characters to show in the Preview column.
        """
        if not self.chunks:
            _out("(no chunks)")
            return

        src_w = min(30, max(6, max(len(c.source) for c in self.chunks)))
        pw = preview_width

        def _row(
            idx: str, doc: str, src: str, chars: str, words: str, prev: str
        ) -> str:
            return (
                f"\u2502 {idx:>4} \u2502 {doc:>3} \u2502 {src:<{src_w}} \u2502"
                f" {chars:>5} \u2502 {words:>5} \u2502 {prev:<{pw}} \u2502"
            )

        def _hline(lc: str, mc: str, rc: str) -> str:
            segs = [
                "\u2500" * 6,
                "\u2500" * 5,
                "\u2500" * (src_w + 2),
                "\u2500" * 7,
                "\u2500" * 7,
                "\u2500" * (pw + 2),
            ]
            return lc + mc.join(segs) + rc

        _out(_hline("\u250c", "\u252c", "\u2510"))
        _out(_row("#", "Doc", "Source", "Chars", "Words", "Preview"))
        _out(_hline("\u251c", "\u253c", "\u2524"))
        for c in self.chunks:
            _out(
                _row(
                    str(c.index),
                    str(c.doc_index),
                    c.source[:src_w],
                    str(c.char_count),
                    str(c.word_count),
                    c.preview(pw),
                )
            )
        _out(_hline("\u2514", "\u2534", "\u2518"))
        _out(
            f"  {self.total_chunks} chunks \u2502 {self.source_documents} source docs"
            f" \u2502 avg {self.avg_chars:.0f} chars"
            f" \u2502 min {self.min_chars} \u2502 max {self.max_chars}"
        )

    def detail(self, index: int) -> None:
        """Print the full content of a single chunk by its global index.

        Args:
            index: The global chunk index shown in the ``#`` column of :meth:`table`.
        """
        matches = [c for c in self.chunks if c.index == index]
        if not matches:
            _out(f"No chunk with index {index}.")
            return
        c = matches[0]
        bar = "\u2500" * 64
        _out(bar)
        _out(f"  Chunk #{c.index}")
        _out(f"  Source    : {c.source}")
        _out(f"  Doc index : {c.doc_index}  (chunk {c.chunk_index} within this doc)")
        _out(f"  Chars     : {c.char_count}  \u2502  Words: {c.word_count}")
        _out(bar)
        _out(c.content)
        _out(bar)

    def summary(self) -> None:
        """Print aggregate statistics and a character-count histogram."""
        _out("ChunkReport Summary")
        _out(f"  Source documents : {self.source_documents}")
        _out(f"  Total chunks     : {self.total_chunks}")
        if not self.chunks:
            return
        _out(f"  Avg char count   : {self.avg_chars:.1f}")
        _out(f"  Min char count   : {self.min_chars}")
        _out(f"  Max char count   : {self.max_chars}")
        _out()

        buckets: List[tuple[str, int]] = [
            ("<100", sum(1 for c in self.chunks if c.char_count < 100)),
            ("100\u2013200", sum(1 for c in self.chunks if 100 <= c.char_count < 200)),
            ("200\u2013300", sum(1 for c in self.chunks if 200 <= c.char_count < 300)),
            ("300\u2013500", sum(1 for c in self.chunks if 300 <= c.char_count < 500)),
            (
                "500\u20131000",
                sum(1 for c in self.chunks if 500 <= c.char_count < 1000),
            ),
            ("1000+", sum(1 for c in self.chunks if c.char_count >= 1000)),
        ]
        active = [(lbl, cnt) for lbl, cnt in buckets if cnt > 0]
        if not active:
            return

        max_cnt = max(cnt for _, cnt in active)
        bar_w = 28
        label_w = max(len(lbl) for lbl, _ in active)
        _out("  Size distribution (chars):")
        for lbl, cnt in active:
            pct = cnt / self.total_chunks * 100
            filled = math.ceil(cnt / max_cnt * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            _out(f"    {lbl:<{label_w}}  {bar}  {cnt:>3} ({pct:5.1f}%)")

    def for_source(self, source: str) -> "ChunkReport":
        """Return a new :class:`ChunkReport` containing only chunks from one source.

        Args:
            source: The value of ``metadata["source"]`` to filter on.

        Example::

            sub = report.for_source("Scott Derrickson")
            sub.table()
        """
        filtered = [c for c in self.chunks if c.source == source]
        return ChunkReport(source_documents=1 if filtered else 0, chunks=filtered)

    def to_dataframe(self) -> "Any":
        """Return all chunks as a pandas DataFrame for programmatic analysis.

        Requires ``pandas`` (``pip install pandas``).

        Columns: index, doc_index, chunk_index, source, char_count, word_count, content

        Example::

            df = report.to_dataframe()
            df[df.char_count < 100]                          # tiny chunks
            df.groupby("source").char_count.mean()           # avg size per doc
            df.sort_values("char_count", ascending=False)    # largest first
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            ) from exc

        return pd.DataFrame(
            [
                {
                    "index": c.index,
                    "doc_index": c.doc_index,
                    "chunk_index": c.chunk_index,
                    "source": c.source,
                    "char_count": c.char_count,
                    "word_count": c.word_count,
                    "content": c.content,
                }
                for c in self.chunks
            ]
        )


def inspect_chunks(
    documents: List[Document],
    splitter: BaseTextSplitter,
) -> ChunkReport:
    """Split documents and return a :class:`ChunkReport` for visual inspection.

    This is a pure dry-run — it calls ``splitter.split_text`` directly and
    never touches any vector store or embedding model.

    Args:
        documents: Source documents to inspect.
        splitter: Any :class:`~rag_sdk.document.base.BaseTextSplitter`
                  (``TextSplitter``, ``SemanticSplitter``, etc.).

    Returns:
        :class:`ChunkReport` with per-chunk details and aggregate statistics.

    Example::

        from rag_sdk.document import TextSplitter, inspect_chunks

        splitter = TextSplitter(chunk_size=512, chunk_overlap=50)
        report = inspect_chunks(docs, splitter)
        report.summary()
        report.table()
        report.detail(3)
    """
    chunks: List[ChunkInfo] = []
    global_idx = 0

    for doc_idx, doc in enumerate(documents):
        source = str(doc.metadata.get("source", f"doc_{doc_idx}"))
        texts = splitter.split_text(doc.content)
        for chunk_idx, text in enumerate(texts):
            chunks.append(
                ChunkInfo(
                    index=global_idx,
                    doc_index=doc_idx,
                    chunk_index=chunk_idx,
                    source=source,
                    char_count=len(text),
                    word_count=len(text.split()),
                    content=text,
                )
            )
            global_idx += 1

    return ChunkReport(source_documents=len(documents), chunks=chunks)
