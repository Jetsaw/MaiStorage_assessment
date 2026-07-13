from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import scrapy
from lxml import html as lxml_html
from pypdf import PdfReader
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import SitemapSpider


BASE_URL = "https://www.maistorage.com"
ALLOWED_HOSTS = {"maistorage.com", "www.maistorage.com"}
ALLOWED_HTML_PATHS = {
    "/",
    "/about/",
    "/aidaptiv-plus-support/",
    "/aidaptiv/",
    "/automotive/",
    "/careers/",
    "/enterprise-b-series/",
    "/enterprise-d-series/",
    "/enterprise-s-series/",
    "/enterprise-x-series/",
    "/enterprise/",
}
REQUIRED_HTML_PATHS = {
    "/aidaptiv-plus-support/",
    "/enterprise-b-series/",
    "/enterprise-d-series/",
    "/enterprise-s-series/",
    "/enterprise-x-series/",
}
REQUIRED_PDF_NAMES = {
    "phison-aidaptiv-prosuite-2.0_install-guide_v1.5.pdf",
    "phison-aidaptiv-prosuite-2.0_user-guide_-v1.9-.pdf",
}
PRODUCT_RE = re.compile(r"^[A-Z]{1,3}\d{2,3}[A-Z]?$")
FIELD_LABELS = {
    "interface specifications": "interface",
    "capacity": "capacity",
    "ssd power consumption": "power",
    "performance": "performance",
}
SKIP_TOKENS = {"contact us", "download brochure", "press release"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_path(url: str) -> str:
    path = unquote(urlparse(url).path or "/")
    return path if path.endswith("/") or path.lower().endswith(".pdf") else f"{path}/"


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
        return False
    path = normalized_path(url)
    return path in ALLOWED_HTML_PATHS or (
        path.lower().startswith("/wp-content/uploads/") and path.lower().endswith(".pdf")
    )


def clean_token(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def parse_products(tokens: list[str]) -> list[dict[str, str]]:
    products: list[dict[str, list[str] | str]] = []
    current: dict[str, list[str] | str] | None = None
    field = "description"

    for raw in tokens:
        token = clean_token(raw)
        if not token:
            continue
        if PRODUCT_RE.fullmatch(token):
            if current:
                products.append(current)
            current = {"name": token, "description": []}
            field = "description"
            continue
        if current is None:
            continue
        label = FIELD_LABELS.get(token.casefold())
        if label:
            field = label
            current.setdefault(field, [])
            continue
        if token.casefold() in SKIP_TOKENS:
            continue
        values = current.setdefault(field, [])
        assert isinstance(values, list)
        if not values or values[-1] != token:
            values.append(token)

    if current:
        products.append(current)

    result: list[dict[str, str]] = []
    for product in products:
        result.append(
            {
                key: "\n".join(value) if isinstance(value, list) else value
                for key, value in product.items()
            }
        )
    return result


def extract_html(body: bytes, url: str) -> dict:
    document = lxml_html.fromstring(body.decode("utf-8", errors="replace"), base_url=url)
    for element in document.xpath("//script|//style|//noscript|//header|//nav|//footer|//form|//aside"):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
    content = document.xpath('//*[@id="content"]') or document.xpath("//main") or [document]
    root = content[0]
    tokens = [clean_token(text) for text in root.itertext() if clean_token(text)]
    headings = [
        clean_token(element.text_content())
        for element in root.xpath(".//h1|.//h2|.//h3")
        if clean_token(element.text_content())
    ]
    title_nodes = document.xpath("//title/text()")
    title = clean_token(title_nodes[0]) if title_nodes else headings[0] if headings else url
    return {
        "kind": "html",
        "url": url,
        "title": title,
        "text": "\n".join(tokens),
        "headings": headings,
        "products": parse_products(tokens),
    }


def extract_pdf(body: bytes, url: str) -> dict:
    reader = PdfReader(io.BytesIO(body))
    pages = [
        {"page": number, "text": (page.extract_text() or "").strip()}
        for number, page in enumerate(reader.pages, start=1)
    ]
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title") or Path(unquote(urlparse(url).path)).name)
    return {"kind": "pdf", "url": url, "title": title, "pages": pages}


def source_diff(old_sources: list[dict], new_sources: list[dict]) -> dict[str, list[str]]:
    def fingerprint(entry: dict) -> tuple[str, str | None]:
        return entry["content_hash"], entry.get("processed_hash")

    old = {entry["url"]: fingerprint(entry) for entry in old_sources}
    new = {entry["url"]: fingerprint(entry) for entry in new_sources}
    return {
        "added": sorted(url for url in new.keys() - old.keys()),
        "changed": sorted(url for url in new.keys() & old.keys() if new[url] != old[url]),
        "unchanged": sorted(url for url in new.keys() & old.keys() if new[url] == old[url]),
        "removed": sorted(old.keys() - new.keys()),
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_name(url: str, digest: str) -> str:
    stem = Path(unquote(urlparse(url).path).rstrip("/")).stem or "home"
    safe = re.sub(r"[^a-z0-9]+", "-", stem.casefold()).strip("-") or "source"
    return f"{safe}-{digest[:12]}"


class ArtifactPipeline:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.entries: dict[str, dict] = {}

    @classmethod
    def from_crawler(cls, crawler):
        instance = cls(crawler.settings["OUTPUT_DIR"])
        instance.crawler = crawler
        return instance

    def open_spider(self):
        (self.output_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(parents=True, exist_ok=True)

    def process_item(self, item):
        raw: bytes = item["raw"]
        processed: dict = item["processed"]
        digest = hashlib.sha256(raw).hexdigest()
        name = artifact_name(processed["url"], digest)
        extension = ".pdf" if processed["kind"] == "pdf" else ".html"
        raw_path = self.output_dir / "raw" / f"{name}{extension}"
        processed_path = self.output_dir / "processed" / f"{name}.json"
        raw_path.write_bytes(raw)
        atomic_json(processed_path, processed)
        processed_hash = hashlib.sha256(
            json.dumps(processed, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        products = [product["name"] for product in processed.get("products", [])]
        self.entries[processed["url"]] = {
            "url": processed["url"],
            "kind": processed["kind"],
            "title": processed["title"],
            "content_hash": digest,
            "processed_hash": processed_hash,
            "retrieved_at": item["retrieved_at"],
            "raw_path": raw_path.relative_to(self.output_dir.parent).as_posix(),
            "processed_path": processed_path.relative_to(self.output_dir.parent).as_posix(),
            "page_count": len(processed.get("pages", [])),
            "product_codes": products,
        }
        return item

    def close_spider(self):
        sources = sorted(self.entries.values(), key=lambda entry: entry["url"])
        html_paths = {normalized_path(entry["url"]) for entry in sources if entry["kind"] == "html"}
        pdf_names = {
            Path(unquote(urlparse(entry["url"]).path)).name.casefold()
            for entry in sources
            if entry["kind"] == "pdf"
        }
        missing = sorted(REQUIRED_HTML_PATHS - html_paths) + sorted(REQUIRED_PDF_NAMES - pdf_names)
        registry_path = self.output_dir / "source_registry.json"
        old_registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"sources": []}
        diff = source_diff(old_registry.get("sources", []), sources)
        status = "published" if not missing else "rejected"
        run = {
            "finished_at": utc_now(),
            "status": status,
            "source_count": len(sources),
            "html_count": sum(entry["kind"] == "html" for entry in sources),
            "pdf_count": sum(entry["kind"] == "pdf" for entry in sources),
            "missing_required_sources": missing,
            "diff": diff,
        }
        atomic_json(self.output_dir / "last_run.json", run)
        if status == "published":
            atomic_json(registry_path, {"generated_at": run["finished_at"], "sources": sources})
        else:
            self.crawler.spider.logger.error("Publication rejected; missing required sources: %s", missing)


class MaiStorageSpider(SitemapSpider):
    name = "maistorage_sources"
    allowed_domains = sorted(ALLOWED_HOSTS)
    sitemap_urls = [f"{BASE_URL}/robots.txt"]
    sitemap_rules = [(r".*", "parse_page")]

    def sitemap_filter(self, entries):
        for entry in entries:
            parsed = urlparse(entry["loc"])
            is_child_sitemap = (
                parsed.scheme == "https"
                and (parsed.hostname or "").lower() in ALLOWED_HOSTS
                and parsed.path.lower().endswith(".xml")
            )
            if is_child_sitemap or (
                is_allowed_url(entry["loc"]) and normalized_path(entry["loc"]) in ALLOWED_HTML_PATHS
            ):
                yield entry

    def parse_page(self, response):
        if not is_allowed_url(response.url):
            self.logger.warning("Rejected out-of-scope response: %s", response.url)
            return
        yield {"raw": response.body, "processed": extract_html(response.body, response.url), "retrieved_at": utc_now()}
        for href in response.xpath('//a[contains(translate(@href,"PDF","pdf"),".pdf")]/@href').getall():
            pdf_url = response.urljoin(href)
            if is_allowed_url(pdf_url):
                yield scrapy.Request(pdf_url, callback=self.parse_pdf)

    def parse_pdf(self, response):
        if not is_allowed_url(response.url):
            self.logger.warning("Rejected out-of-scope PDF: %s", response.url)
            return
        yield {"raw": response.body, "processed": extract_pdf(response.body, response.url), "retrieved_at": utc_now()}


def run(output_dir: Path, log_level: str = "INFO") -> dict:
    process = CrawlerProcess(
        settings={
            "AUTOTHROTTLE_ENABLED": True,
            "COOKIES_ENABLED": False,
            "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
            "DOWNLOAD_DELAY": 0.5,
            "DOWNLOAD_MAXSIZE": 25 * 1024 * 1024,
            "ITEM_PIPELINES": {ArtifactPipeline: 300},
            "LOG_LEVEL": log_level,
            "OUTPUT_DIR": str(output_dir),
            "RETRY_TIMES": 2,
            "ROBOTSTXT_OBEY": True,
            "TELNETCONSOLE_ENABLED": False,
            "USER_AGENT": "MaiStorageTechnicalCopilot/0.1 (+public-source research prototype)",
        }
    )
    process.crawl(MaiStorageSpider)
    process.start()
    return json.loads((output_dir / "last_run.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Crawl approved public MaiStorage sources.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.log_level)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "published" else 1


if __name__ == "__main__":
    sys.exit(main())
