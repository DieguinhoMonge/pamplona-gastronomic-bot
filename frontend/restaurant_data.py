"""Parse the restaurant PDFs into structured data.

Used by the mock backend (as a small searchable corpus) and, later, by the
Explorer and Insights pages. It is independent from the real RAG ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf

from frontend.config import DOCS_DIR

# Manual labels (not stated in the PDFs) to allow filtering in the Explorer.
CATEGORIES: dict[int, str] = {
    1: "Burgers", 2: "Burgers", 3: "Burgers", 4: "Burgers",
    5: "Pintxos & miniature", 8: "Pintxos & miniature", 20: "Pintxos & miniature",
    6: "International fusion", 7: "International fusion",
    9: "Traditional Navarrese & Basque", 10: "Traditional Navarrese & Basque",
    12: "Traditional Navarrese & Basque", 16: "Traditional Navarrese & Basque",
    13: "Asador & Sidrería", 14: "Asador & Sidrería", 15: "Asador & Sidrería",
    11: "Haute cuisine", 17: "Haute cuisine", 18: "Haute cuisine", 19: "Haute cuisine",
}

# Files whose PDF has no "Name (Town)" title line.
TITLE_FALLBACK: dict[int, tuple[str, str]] = {
    3: ("More Than Burgers", "Pamplona"),
}

_TITLE_RE = re.compile(r"^(?:\d+\.\s*)?(?P<name>.+?)\s*\((?P<town>[^)]+)\)\s*$")
_DISH_RE = re.compile(
    r"^(?P<name>.+)\((?P<price>\d+[.,]\d{2})\s*€(?P<kg>\s*/\s*kg)?\)\s*:\s*(?P<ing>.+)$", re.S
)
_BULLET_RE = re.compile(r"^\s*(?:•|\*|\d{1,2}\.)\s*")


@dataclass
class Dish:
    name: str
    price: float
    per_kg: bool
    ingredients: str
    page: int

    @property
    def price_label(self) -> str:
        return f"{self.price:.2f} €".replace(".", ",") + ("/kg" if self.per_kg else "")


@dataclass
class Restaurant:
    id: int
    name: str
    town: str
    category: str
    file: str
    history: str
    cuisine: str
    dishes: list[Dish] = field(default_factory=list)

    @property
    def avg_price(self) -> float | None:
        prices = [d.price for d in self.dishes if not d.per_kg]
        return sum(prices) / len(prices) if prices else None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(path: Path) -> Restaurant:
    rid = int(path.name.split(".")[0])
    doc = pymupdf.open(path)
    blocks: list[tuple[str, int]] = []
    for page_no, page in enumerate(doc, start=1):
        for b in page.get_text("blocks"):
            text = _clean(b[4])
            if text:
                blocks.append((text, page_no))

    name, town = TITLE_FALLBACK.get(rid, ("", ""))
    paragraphs: list[str] = []
    dishes: list[Dish] = []
    in_dishes = False

    for text, page_no in blocks:
        if text.startswith("Lista de Platos"):
            in_dishes = True
            continue
        if not in_dishes:
            m = _TITLE_RE.match(text)
            if not name and m and len(text) < 80:
                name, town = m["name"].strip(), m["town"].strip()
            elif text not in {"•", "*"}:
                paragraphs.append(text)
            continue

        item = _BULLET_RE.sub("", text).strip()
        if not item:
            continue
        m = _DISH_RE.match(item)
        if m:
            dishes.append(Dish(
                name=_clean(m["name"]),
                price=float(m["price"].replace(",", ".")),
                per_kg=bool(m["kg"]),
                ingredients=_clean(m["ing"]),
                page=page_no,
            ))
        elif dishes:  # continuation of the previous dish split across blocks
            dishes[-1].ingredients = _clean(dishes[-1].ingredients + " " + item)

    return Restaurant(
        id=rid,
        name=name or path.stem,
        town=town or "Navarra",
        category=CATEGORIES.get(rid, "Other"),
        file=path.name,
        history=paragraphs[0] if paragraphs else "",
        cuisine=" ".join(paragraphs[1:]),
        dishes=dishes,
    )


@lru_cache(maxsize=1)
def load_restaurants(docs_dir: Path = DOCS_DIR) -> tuple[Restaurant, ...]:
    files = sorted(docs_dir.glob("*.pdf"), key=lambda p: int(p.name.split(".")[0]))
    return tuple(parse_pdf(f) for f in files)
