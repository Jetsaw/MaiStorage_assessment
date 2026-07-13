from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import psycopg
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb


ROOT = Path(__file__).parents[1]
NAMESPACE = uuid.UUID("0f50f5d1-c438-4f8a-b0ca-091baaa651b8")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://maistorage:maistorage@127.0.0.1:5432/maistorage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR")


def stable_id(*parts: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, "\x1f".join(parts))


def chunks(text: str, size: int = 220, overlap: int = 35):
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def family_for(code: str) -> tuple[str, str]:
    if code.startswith("X"):
        return "X-Series", "performance"
    if code.startswith("D"):
        return "D-Series", "data-center"
    if code.startswith(("B", "BA")):
        return "B-Series", "boot"
    return "S-Series", "sata"


def source_priority(url: str) -> int:
    return 1 if "enterprise-" in url or url.lower().endswith(".pdf") else 2


def corpus_version(sources: list[dict]) -> uuid.UUID:
    digest = hashlib.sha256(
        "\n".join(
            f'{source["url"]}:{source["content_hash"]}:{source.get("processed_hash", "")}'
            for source in sources
        ).encode()
    ).hexdigest()
    return stable_id("corpus", digest)


def load_processed(source: dict) -> dict:
    return json.loads((ROOT / source["processed_path"]).read_text(encoding="utf-8"))


def build() -> dict:
    registry = json.loads((ROOT / "data" / "source_registry.json").read_text(encoding="utf-8"))
    sources = registry["sources"]
    version = corpus_version(sources)
    schema = (ROOT / "schema.sql").read_text(encoding="utf-8")

    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(schema, prepare=False)
        register_vector(connection)
        existing = connection.execute(
            "SELECT status FROM corpus WHERE version = %s", (version,)
        ).fetchone()
        if existing:
            connection.execute("UPDATE corpus SET status = 'superseded' WHERE status = 'active' AND version <> %s", (version,))
            connection.execute("UPDATE corpus SET status = 'active', published_at = now() WHERE version = %s", (version,))
            connection.execute(
                "INSERT INTO active_corpus(singleton, version) VALUES (true, %s) "
                "ON CONFLICT (singleton) DO UPDATE SET version = excluded.version",
                (version,),
            )
            counts = connection.execute(
                "SELECT (SELECT count(*) FROM source WHERE corpus_version = %s), "
                "(SELECT count(*) FROM product WHERE corpus_version = %s), "
                "(SELECT count(*) FROM document_chunk dc JOIN source_section ss ON ss.id=dc.source_section_id "
                "JOIN source s ON s.id=ss.source_id WHERE s.corpus_version = %s)",
                (version, version, version),
            ).fetchone()
            return {"status": "already_indexed", "corpus_version": str(version), "sources": counts[0], "products": counts[1], "chunks": counts[2]}

        connection.execute("INSERT INTO corpus(version, status) VALUES (%s, 'staging')", (version,))
        sections: list[dict] = []

        for source in sources:
            source_id = stable_id(str(version), "source", source["url"])
            connection.execute(
                "INSERT INTO source(id, corpus_version, canonical_url, kind, title, retrieved_at, content_hash, "
                "raw_object_uri, authority_rank, status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'active')",
                (
                    source_id,
                    version,
                    source["url"],
                    source["kind"],
                    source["title"],
                    source["retrieved_at"],
                    source["content_hash"],
                    source["raw_path"],
                    source_priority(source["url"]),
                ),
            )
            document = load_processed(source)
            if source["kind"] == "pdf":
                candidates = [
                    (f'Page {page["page"]}', page["page"], page["text"], None)
                    for page in document["pages"]
                    if page["text"].strip()
                ]
            elif document.get("products"):
                candidates = []
                for product in document["products"]:
                    text = "\n".join(f"{key}: {value}" for key, value in product.items() if value)
                    candidates.append((product["name"], None, text, product))
            else:
                candidates = [("Page", None, document["text"], None)]

            for heading, page, text, product in candidates:
                section_hash = hashlib.sha256(text.encode()).hexdigest()
                section_id = stable_id(str(source_id), heading, section_hash)
                connection.execute(
                    "INSERT INTO source_section(id, source_id, heading_path, page, raw_text, normalized_text, section_hash) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (section_id, source_id, heading, page, text, " ".join(text.split()), section_hash),
                )
                sections.append({"id": section_id, "source_id": source_id, "heading": heading, "page": page, "text": text, "url": source["url"]})
                if product:
                    product_id = stable_id(str(version), "product", product["name"])
                    family, category = family_for(product["name"])
                    connection.execute(
                        "INSERT INTO product(id, corpus_version, name, family, category, positioning, source_section_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (product_id, version, product["name"], family, category, product.get("description"), section_id),
                    )
                    for field in ("interface", "capacity", "power", "performance"):
                        value = product.get(field)
                        if value:
                            connection.execute(
                                "INSERT INTO product_spec(id, product_id, field, text_value, source_section_id) VALUES (%s,%s,%s,%s,%s)",
                                (stable_id(str(product_id), field), product_id, field, value, section_id),
                            )

        guide_section = next(
            section for section in sections
            if "Install-guide" in section["url"] and section["page"] == 6
        )
        rules = [
            ("operating_system", "eq", {"name": "Ubuntu", "version": "22.04 LTS Desktop"}),
            ("nvidia_driver", "gte", 550),
            ("ports", "contains_all", [8899, 8799, 3019, 5432, 9400, 7017, 8000, 3090, 9100]),
        ]
        for field, operator, expected in rules:
            connection.execute(
                "INSERT INTO compatibility_rule(id, corpus_version, product_version, field, operator, expected_value, source_section_id) "
                "VALUES (%s,%s,'Pro Suite 2.0.5',%s,%s,%s,%s)",
                (stable_id(str(version), "rule", field), version, field, operator, Jsonb(expected), guide_section["id"]),
            )

        chunk_rows = []
        for section in sections:
            for index, content in enumerate(chunks(section["text"])):
                chunk_rows.append((
                    stable_id(str(section["id"]), "chunk", str(index)),
                    section["id"],
                    index,
                    content,
                    len(content.split()),
                ))
        model = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=EMBEDDING_CACHE_DIR)
        embeddings = list(model.embed([row[3] for row in chunk_rows]))
        for row, embedding in zip(chunk_rows, embeddings, strict=True):
            connection.execute(
                "INSERT INTO document_chunk(id, source_section_id, chunk_index, content, token_count, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (*row, embedding),
            )

        connection.execute("UPDATE corpus SET status = 'superseded' WHERE status = 'active'")
        connection.execute("UPDATE corpus SET status = 'active', published_at = now() WHERE version = %s", (version,))
        connection.execute(
            "INSERT INTO active_corpus(singleton, version) VALUES (true, %s) "
            "ON CONFLICT (singleton) DO UPDATE SET version = excluded.version",
            (version,),
        )
        return {"status": "indexed", "corpus_version": str(version), "sources": len(sources), "products": 10, "chunks": len(chunk_rows)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
