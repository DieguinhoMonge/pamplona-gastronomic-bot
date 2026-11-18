"""Mock RAG backend used while the real LangChain + Gemini pipeline is built.

It behaves like a RAG system, but without an LLM or embeddings:
  * it searches the REAL restaurant PDFs with keyword BM25 (plus a small
    English->Spanish synonym map), so the sources it returns are real chunks;
  * it writes the answer from templates, never inventing information;
  * it simulates the edge cases the demo has to show: follow-ups, out-of-scope
    questions, nonsense, greetings and connection failures.

Implements the same `ask()` contract as the real backend (see INTERFACE.md).
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

from frontend.backend_interface import BackendConnectionError, RAGResponse, Source
from frontend.restaurant_data import Dish, Restaurant, load_restaurants

# --------------------------------------------------------------------------- text

_STOPWORDS = set("""
a al algo and any are as at be best but by can con cual cuales de del do does el en
es esta este for from give good has have how i in is it its la las like los me my
of on one or para por que recommend some tell the their them there these they this
to un una what where which who with y you your want would should could get find know
show list about eat food dish dishes restaurant restaurants place places serve serves
lo le se su sus mas muy did there's whats what's contain contains include includes
ingredient ingredients made offer offers much many is was were are
""".split())

_SYNONYMS: dict[str, list[str]] = {
    "burger": ["hamburguesa", "burger"], "cheese": ["queso"], "cheesecake": ["tarta", "queso"],
    "dessert": ["postre", "tarta", "helado", "torrija", "chocolate"], "sweet": ["postre", "dulce"],
    "cod": ["bacalao"], "lamb": ["cordero"], "steak": ["chuleton", "chuleta", "entrecot"], "chuleton": ["chuleton", "chuleta"],
    "mushroom": ["hongo", "seta", "boletu"], "asparagu": ["esparrago"],
    "artichoke": ["alcachofa"], "vegetarian": ["veggie", "verdura", "vegetal"],
    "vegan": ["veggie", "verdura"], "vegetable": ["verdura", "huerta"],
    "octopu": ["pulpo"], "hake": ["merluza"], "cider": ["sidra", "sidreria", "txotx"],
    "truffle": ["trufa"], "ham": ["jamon"], "egg": ["huevo"], "chicken": ["pollo"],
    "pork": ["cerdo", "cochinillo"], "beef": ["vacuno", "buey", "ternera"],
    "fish": ["pescado", "bacalao", "merluza", "lubina", "rodaballo", "besugo"],
    "seafood": ["almeja", "marisco", "kokotxa", "gamba", "carabinero", "bogavante"],
    "wine": ["vino"], "sausage": ["chistorra", "txistorra", "chorizo", "morcilla"],
    "potato": ["patata"], "fry": ["patata", "frita"], "frie": ["patata", "frita"],
    "tuna": ["atun", "bonito"], "rice": ["arroz"], "bean": ["alubia", "pocha"],
    "venison": ["ciervo"], "deer": ["ciervo"], "boar": ["jabali"],
    "pigeon": ["paloma", "pichon"], "clam": ["almeja"], "spicy": ["picante", "guindilla"],
    "grill": ["parrilla", "brasa"], "grilled": ["parrilla", "brasa"],
    "barbecue": ["barbacoa", "parrilla"], "bbq": ["barbacoa", "parrilla"],
    "history": ["historia", "fundado"], "founded": ["fundado", "historia"],
    "japanese": ["japones", "sushi", "maki"], "argentinian": ["argentina", "criolla"],
    "argentine": ["argentina", "criolla"], "croquette": ["croqueta"],
    "pepper": ["pimiento", "piquillo"], "ice": ["helado"], "cream": ["crema", "nata"],
    "chocolate": ["chocolate", "cacao"], "duck": ["pato", "foie"], "lobster": ["bogavante"],
    "squid": ["calamar", "chipiron"], "breakfast": ["desayuno", "churro"],
    "pintxo": ["pincho", "pintxo"], "pincho": ["pincho", "pintxo"], "tapa": ["pincho", "tapeo"],
}

_PRICE_WORDS = {"price", "cost", "costs", "much", "cheap", "cheapest", "expensive",
                "precio", "cuanto", "cuesta", "euros", "euro"}
_ABOUT_WORDS = {"history", "founded", "story", "atmosphere", "ambience", "about",
                "historia", "ambiente", "style", "decor", "who", "chef"}
_FOLLOWUP_WORDS = {"it", "its", "that", "this", "there", "they", "them", "those", "ones",
                   "he", "she", "his", "her", "else", "another", "also"}
_FOLLOWUP_STARTS = (("and",), ("y",), ("what", "about"), ("how", "about"), ("and", "what"))
_GREETINGS = {"hi", "hello", "hey", "hola", "buenas", "thanks", "thank", "gracias",
              "morning", "evening", "kaixo"}
_OUTSIDE_PLACES = {"madrid", "barcelona", "bilbao", "valencia", "sevilla", "seville",
                   "paris", "london", "york", "zaragoza", "donostia", "sebastian",
                   "logrono", "malaga", "rome", "berlin", "tokyo", "lisboa", "lisbon"}
_GENERIC_NAME_TOKENS = {"restaurante", "asador", "la", "de", "el", "cocina", "burger",
                        "sidreria", "cafe", "bbq", "gastrobar", "gastroteka", "meson",
                        "las", "than", "more", "burgers", "2"}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith("ses"):
        return tok[:-2] if tok[:-2].endswith(("on", "or", "al")) else tok[:-1]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _raw_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in _raw_tokens(text) if t not in _STOPWORDS]


def _expand(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out.append(t)
        out.extend(_SYNONYMS.get(t, []))
    return out


def _looks_like_nonsense(raw: list[str]) -> bool:
    words = [w for w in raw if w.isalpha()]
    if not words:
        return True
    bad = 0
    for w in words:
        vowels = sum(c in "aeiou" for c in w)
        if (len(w) > 3 and vowels / len(w) < 0.25) or re.search(r"[^aeiou]{5,}", w):
            bad += 1
    return bad / len(words) >= 0.5


# --------------------------------------------------------------------------- index

@dataclass
class _Chunk:
    restaurant: Restaurant
    kind: str  # "history" | "cuisine" | "dish"
    text: str
    page: int
    dish: Dish | None
    tokens: Counter


class _Index:
    k1, b = 1.4, 0.75

    def __init__(self, restaurants: tuple[Restaurant, ...]):
        self.restaurants = restaurants
        self.chunks: list[_Chunk] = []
        for r in restaurants:
            header = f"{r.name} ({r.town})"
            for kind, body in (("history", r.history), ("cuisine", r.cuisine)):
                text = f"{header}. {body}"
                self.chunks.append(_Chunk(r, kind, text, 1, None, Counter(_tokens(text))))
            for d in r.dishes:
                text = f"{d.name} ({d.price_label}): {d.ingredients}"
                self.chunks.append(_Chunk(r, "dish", text, d.page, d, Counter(_tokens(text))))
        n = len(self.chunks)
        df: Counter = Counter()
        for c in self.chunks:
            df.update(c.tokens.keys())
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.avgdl = sum(sum(c.tokens.values()) for c in self.chunks) / n
        self.name_tokens = {
            r.id: {t for t in _raw_tokens(r.name) if t not in _GENERIC_NAME_TOKENS and len(t) > 2}
            for r in restaurants
        }
        self.name_tokens[3].add("mtb")

    def bm25(self, query: list[str], chunk: _Chunk) -> float:
        dl = sum(chunk.tokens.values())
        score = 0.0
        for t in set(query):
            f = chunk.tokens.get(t, 0)
            if f:
                score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def mentioned_restaurants(self, raw: list[str]) -> set[int]:
        words = set(raw)
        return {rid for rid, toks in self.name_tokens.items() if toks & words}


@lru_cache(maxsize=1)
def _index() -> _Index:
    return _Index(load_restaurants())


# --------------------------------------------------------------------------- answer helpers

def _first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n])


def _stream(text: str, delay: float) -> Iterator[str]:
    for word in re.findall(r"\S+\s*", text):
        yield word
        if delay:
            time.sleep(delay)


def _build_prompt(question: str, history: list[dict], context: list[_Chunk]) -> str:
    ctx = "\n\n".join(f"[{i}] ({c.restaurant.name} · {c.restaurant.file} p.{c.page})\n{c.text}"
                      for i, c in enumerate(context, 1)) or "(no relevant context found)"
    hist = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in history[-6:]) or "(empty)"
    return (
        "You are a gastronomic assistant for Pamplona and Navarra.\n"
        "Answer ONLY with information from the CONTEXT. If the answer is not in the "
        "CONTEXT, say clearly that you don't have that information. Cite sources as [n].\n\n"
        f"CONTEXT:\n{ctx}\n\nCHAT HISTORY:\n{hist}\n\nQUESTION: {question}\nANSWER:"
    )


def _response(answer: str, sources: list[Source], grounded: bool, settings: dict,
              debug: dict) -> RAGResponse:
    debug.setdefault("model", f"mock (UI selection: {settings.get('model')})")
    return RAGResponse(
        answer=_stream(answer, float(settings.get("stream_delay", 0.02))),
        sources=sources, grounded=grounded, debug=debug,
    )


# --------------------------------------------------------------------------- ask()

def ask(question: str, history: list[dict], settings: dict) -> RAGResponse:
    t0 = time.perf_counter()
    time.sleep(float(settings.get("latency", 0.6)))  # simulate retrieval latency

    if settings.get("simulate_failure"):
        raise BackendConnectionError("Simulated failure: the Gemini API did not respond (timeout).")

    idx = _index()
    raw = _raw_tokens(question)
    content = _tokens(question)
    top_k = int(settings.get("top_k", 4))

    # Greetings / small talk
    if raw and set(raw) & _GREETINGS and len(content) <= 3:
        answer = ("¡Hola! I'm your gastronomic guide to Pamplona and Navarra. "
                  f"I know **{len(idx.restaurants)} restaurants** and their signature dishes, "
                  "prices and ingredients. Ask me things like *“Where can I eat good pintxos?”* "
                  "or *“How much is the chuletón at Asador Bidea 2?”*")
        return _response(answer, [], True, settings, {"intent": "greeting"})

    # Follow-up resolution: merge with the previous user question
    rewritten = question
    prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    named = idx.mentioned_restaurants(raw)
    new_terms = [t for t in content if t not in _FOLLOWUP_WORDS and t not in _PRICE_WORDS]
    is_followup = (any(tuple(raw[:len(s)]) == s for s in _FOLLOWUP_STARTS)
                   or (set(raw) & _FOLLOWUP_WORDS and len(new_terms) <= 2))
    if prev_user and not named and is_followup and not _looks_like_nonsense(raw):
        rewritten = f"{prev_user} {question}"
        raw = _raw_tokens(rewritten)
        content = _tokens(rewritten)
        named = idx.mentioned_restaurants(raw)

    query = _expand(content)
    price_intent = bool(set(raw) & _PRICE_WORDS)
    name_toks = set().union(*(idx.name_tokens[r] for r in named)) if named else set()
    non_name = [t for t in content if t not in name_toks and t not in _GENERIC_NAME_TOKENS]
    about_intent = bool(set(raw) & _ABOUT_WORDS) or (bool(named) and not non_name)

    scored: list[tuple[float, _Chunk]] = []
    for c in idx.chunks:
        s = idx.bm25(query, c)
        if c.restaurant.id in named:
            s = s * 1.5 + 2.0
            if about_intent and c.kind != "dish":
                s += 3.0
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = scored[:top_k]
    retrieval_ms = round((time.perf_counter() - t0) * 1000)

    sources = [Source(restaurant=c.restaurant.name, file=c.restaurant.file, page=c.page,
                      chunk=c.text, score=round(s / (s + 4.0), 3)) for s, c in hits]
    debug = {"rewritten_query": rewritten if rewritten != question else None,
             "expanded_terms": sorted(set(query)), "retrieval_ms": retrieval_ms,
             "prompt": _build_prompt(question, history, [c for _, c in hits])}

    outside = set(raw) & _OUTSIDE_PLACES
    top = hits[0][0] if hits else 0.0

    # Out of scope: a place outside the knowledge base
    if outside:
        place = sorted(outside)[0].title()
        answer = (f"I'm sorry, my knowledge base only covers **{len(idx.restaurants)} restaurants "
                  f"in Pamplona and Navarra**, so I have no information about {place}.")
        if hits and top > 3:
            r = hits[0][1].restaurant
            answer += (f"\n\nIf you're coming to Navarra, the closest match in my documents is "
                       f"**{r.name}** ({r.town}) [1].")
        else:
            sources = []
        return _response(answer, sources, False, settings, debug)

    # Coverage: share of the question's terms (or their synonyms) found in the top hits
    hit_vocab = set().union(*(c.tokens.keys() for _, c in hits)) if hits else set()
    covered = [t for t in content if t in hit_vocab or set(_SYNONYMS.get(t, [])) & hit_vocab
               or t in name_toks]
    coverage = len(covered) / len(content) if content else 0.0
    debug["coverage"] = round(coverage, 2)

    # Nothing relevant retrieved: nonsense or off-topic
    if top < 2.5 or coverage <= 0.5:
        if _looks_like_nonsense(_raw_tokens(question)):
            answer = ("I couldn't understand that question. Could you rephrase it? "
                      "For example: *“Which restaurants serve cod (bacalao)?”*")
        else:
            answer = ("I don't have that information in my documents. I can only answer "
                      "questions about the **restaurants, dishes, prices and ingredients** "
                      "described in my knowledge base of Pamplona and Navarra.")
        return _response(answer, [], False, settings, debug)

    # Grounded answer built only from retrieved chunks
    relevant = [(i, c) for i, (s, c) in enumerate(hits, 1) if s >= 0.45 * top]
    lines: list[str] = []
    dish_hits = [(i, c) for i, c in relevant if c.kind == "dish"]
    info_hits = [(i, c) for i, c in relevant if c.kind != "dish"]

    top_is_dish = hits[0][1].kind == "dish"
    if price_intent:  # for prices, any retrieved dish is better than a description
        dish_hits = [(i, c) for i, (_, c) in enumerate(hits, 1) if c.kind == "dish"] or dish_hits
    if price_intent and dish_hits:
        i, c = dish_hits[0]
        lines.append(f"**{c.dish.name}** at **{c.restaurant.name}** ({c.restaurant.town}) "
                     f"costs **{c.dish.price_label}** [{i}].")
        if len(dish_hits) > 1:
            lines.append("\nOther related dishes in my documents:")
            lines += [f"- **{c.dish.name}** at {c.restaurant.name}: {c.dish.price_label} [{i}]"
                      for i, c in dish_hits[1:]]
    else:
        info_lines = []
        for i, c in info_hits:
            quote = _first_sentences(c.text.split(". ", 1)[-1])
            info_lines.append(f"**{c.restaurant.name}** ({c.restaurant.town})\n> {quote} [{i}]\n")
        dish_lines = [f"- **{c.dish.name}** at **{c.restaurant.name}** ({c.restaurant.town}), "
                      f"{c.dish.price_label}. *{c.dish.ingredients}* [{i}]" for i, c in dish_hits]
        if top_is_dish and not about_intent:
            lines += ["Here's what I found in my documents:", *dish_lines]
            if info_lines:
                lines += ["\nAbout the restaurant:", info_lines[0]]
        else:
            lines += info_lines
            if dish_lines:
                lines += ["Related dishes:", *dish_lines]

    return _response("\n".join(lines), sources, True, settings, debug)
