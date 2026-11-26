"""Поиск фрагментов судебных актов по нескольким описаниям одного вопроса."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from math import fsum, log
import re
import time

import httpx


WORDS = re.compile(r"[а-яёa-z]+|\d+(?:\.\d+)*", re.I)
STOP = set("""
а без бы был быть в вам весь во вот вы где да дали для до его ее если есть еще же за
и из или им их к как какие каким какой когда которые ли мне может можно мой мы на над нам
наш не него нее нет ни но ну нужно о об один он она они от по под после почему при про
с сам со так такой там те то того только том у уже чем через что чтобы это этот я
""".split())
SUFFIX = re.compile(
    r"(иями|ями|ами|ией|иях|иям|ений|ение|ения|ению|ением|"
    r"ого|его|ому|ему|ыми|ими|ую|юю|ая|яя|ое|ее|ые|ие|ый|ий|ой|"
    r"ах|ях|ов|ев|ей|ам|ям|ом|ем|ы|и|у|ю|а|я|е|о)$"
)
SHORT_TERMS = {"ип", "рф", "гк", "ук", "тк", "ск", "ко", "вс", "цб", "ау", "су"}
SEARCH_BUDGET = 45
TOTAL_IO_BUDGET = 105

DECISION = re.compile(
    r"суд.{0,35}(установил|пришел|признал|отказал|удовлетворил)|"
    r"как следует из материалов|судами установлено|доказательств|в рассматриваемом случае",
    re.I,
)
QUOTED_RULE = re.compile(r"^(в силу|согласно|в соответствии|на основании пункта)", re.I)


def terms(text):
    words = WORDS.findall(text.lower().replace("ё", "е"))
    return {SUFFIX.sub("", w) if len(w) >= 6 else w for w in words
            if w not in STOP and (len(w) >= 3 or w.isdigit() or w in SHORT_TERMS)}


def formulations(question, supplied):
    if not question.strip():
        raise ValueError("Нужен непустой вопрос.")
    if supplied is not None:
        queries = list(dict.fromkeys(" ".join(q.lower().split()) for q in supplied if q.strip()))
        if not 2 <= len(queries) <= 3:
            raise ValueError("Передайте две или три различные поисковые формулировки.")
        return queries
    normalized = re.sub(r"\bст\.\s*(?=\d)", "статья ", question.lower())
    clauses = re.split(r"[!?;,:]+|\.(?!\d)", normalized)
    chunks = []
    for clause in clauses:
        words = [w for w in WORDS.findall(clause) if w not in STOP]
        if words:
            chunks.append(words)
    words = [w for chunk in chunks for w in chunk]
    if len(set(words)) < 2:
        raise ValueError("Для короткого вопроса передайте две разные поисковые формулировки.")
    # Для короткой фразы дополнительный запрос вращается вокруг последнего термина.
    queries = [" ".join(words), *(" ".join(chunk) for chunk in chunks if len(chunk) >= 2),
               " ".join(words[-max(1, len(words) // 2):]), " ".join(words[:max(1, len(words) // 2)])]
    return list(dict.fromkeys(queries))[:3]


def passages(text):
    # В конце страницы СудАкт добавляет навигацию и ссылки на другие документы.
    body = re.split(r"Печать документа|Показать все документы по этому делу|Судебная практика по:", text)[0]
    for paragraph in body.splitlines():
        paragraph = paragraph.strip()
        if len(paragraph) < 55:
            continue
        # У длинных абзацев используем перекрывающиеся окна, сохраняя точную цитату.
        start = 0
        while start < len(paragraph):
            end = min(start + 950, len(paragraph))
            if end < len(paragraph):
                end = max(paragraph.rfind(" ", start + 650, end), start + 650)
            yield paragraph[start:end]
            if end == len(paragraph):
                break
            start = paragraph.rfind(" ", max(start + 1, end - 140), end) + 1 or end


def select(question, queries, candidates, limit):
    profiles = [terms(q) for q in [question, *queries]]
    vocabulary = set.union(*profiles)
    indexed = []
    frequency = Counter()
    for card, text, ranks in candidates:
        spans = [(line, matches) for line in passages(text)
                 if len(matches := terms(line) & vocabulary) >= 2]
        frequency.update(set().union(*(m for _, m in spans)))
        indexed.append((card, spans, ranks))
    weights = {t: 1 + log((1 + len(indexed)) / (1 + frequency[t])) for t in vocabulary}
    totals = [fsum(weights[t] for t in profile) or 1 for profile in profiles]
    results = []
    for card, spans, ranks in indexed:
        scored = []
        for line, matches in spans:
            coverage = [fsum(weights[t] for t in matches & profile) / total
                        for profile, total in zip(profiles, totals)]
            best_angle = max(coverage[1:])
            if best_angle < 0.30:
                continue
            score = (0.50 * best_angle + 0.24 * sum(coverage[1:]) / len(queries)
                     + 0.26 * coverage[0]) * (1 + 0.12 * bool(DECISION.search(line)))
            if QUOTED_RULE.match(line):
                score *= 0.88
            scored.append((score, line))
        if not scored:
            continue
        scored.sort(key=lambda item: -item[0])
        excerpts = [scored[0][1]]
        for _, line in scored[1:]:
            if len(terms(line) & terms(excerpts[0])) / max(1, len(terms(line) | terms(excerpts[0]))) < 0.65:
                excerpts.append(line)
                break
        support = sum(1 / (2 + rank) for rank in ranks)
        score = scored[0][0] + 0.10 * support
        results.append((score, {**card, "excerpts": excerpts}))
    results.sort(key=lambda pair: -pair[0])
    return [item for _, item in results[:limit]]


def within_budget(function, items, deadline):
    """Вернуть готовые ответы в исходном порядке по общему сроку ожидания."""
    if not items:
        return []
    workers = ThreadPoolExecutor(max_workers=3)
    try:
        jobs = [workers.submit(function, item) for item in items]
        finished, _ = wait(jobs, timeout=max(0, deadline - time.monotonic()))
        return [job.result() if job in finished else None for job in jobs]
    finally:
        # Уже выполняющийся HTTP-запрос завершится по таймауту клиента.
        workers.shutdown(wait=False, cancel_futures=True)


def search(question, queries, scope, limit, *, lookup, fetch):
    queries = formulations(question, queries)
    if not 1 <= limit <= 5:
        raise ValueError("Лимит должен быть от 1 до 5.")
    pool = {}
    warnings = []
    started = time.monotonic()

    def find(query):
        return lookup(query=query, scope=scope, article="", case_number="", date_from="",
                      date_to="", page=1, max_per_section=5)

    found_batches = within_budget(find, queries, started + SEARCH_BUDGET)
    for query, found in zip(queries, found_batches):
        if found is None:
            warnings.append(f"Истекло время поиска по запросу «{query}».")
            continue
        if found.get("error"):
            warnings.append(found["error"])
            continue
        if found.get("unconfirmed_sections"):
            warnings.append(f"Источник не подтвердил выдачу по запросу «{query}».")
        for rank, card in enumerate(found["results"], 1):
            key = card["url"]
            if key not in pool:
                pool[key] = (card, [])
            pool[key][1].append(rank)

    def read(entry):
        card, ranks = entry
        try:
            doc = fetch(card["url"], scope)
            return card, doc.get("text", ""), ranks
        except httpx.RequestError:
            return card, "", ranks

    loaded = within_budget(read, list(pool.values()), started + TOTAL_IO_BUDGET)
    readable = [entry for entry in loaded if entry and entry[1]]
    if len(readable) < len(loaded):
        warnings.append(f"Не удалось прочитать документов: {len(loaded) - len(readable)}.")
    return {"queries": queries, "candidates": len(pool), "read": len(readable),
            "results": select(question, queries, readable, limit), "warnings": warnings}
