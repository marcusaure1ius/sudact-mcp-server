import unittest
from unittest.mock import Mock

import httpx

from practice_search import formulations, passages, search


class SearchTests(unittest.TestCase):
    def test_query_variants(self):
        for question in ("Можно ли оспорить сделку с квартирой?", "Залог автомобиля"):
            qs = formulations(question, None)
            self.assertTrue(2 <= len(qs) <= 3)
            self.assertEqual(len(qs), len(set(qs)))
        with self.assertRaises(ValueError):
            formulations("Вопрос", ["ЗАЛОГ", " залог "])
        with self.assertRaises(ValueError):
            formulations(" ", ["два", "варианта"])

    def test_selection_deduplicates_skips_unreadable_and_returns_verbatim_evidence(self):
        relevant = {"url": "one", "doc_id": "1"}
        unrelated = {"url": "two", "doc_id": "2"}
        unavailable = {"url": "three", "doc_id": "3"}
        calls = {"залог автомобиля": [unrelated, relevant], "реализация залога": [relevant, unavailable]}
        def lookup(**kw):
            return {"results": calls[kw["query"]]}
        evidence = "Суд установил обстоятельства реализации залога автомобиля при рассмотрении требований банка."
        def fetch(url, scope):
            if url == "three":
                raise httpx.ConnectError("no connection")
            return {"text": evidence if url == "one" else "Разрешен спор о размере арендной платы по договору аренды нежилого помещения."}
        reader = Mock(side_effect=fetch)
        res = search("Залог автомобиля", list(calls), "arbitral", 5, lookup=lookup, fetch=reader)
        self.assertEqual(res["candidates"], 3)
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(res["read"], 2)
        self.assertTrue(res["warnings"])
        self.assertEqual([(r["url"], r["excerpts"]) for r in res["results"]], [("one", [evidence])])

    def test_long_passages_and_footer(self):
        body = "Длинное предложение для проверки. " * 125
        result = list(passages(body + "\nПечать документа\n" + "Ненужный текст. " * 30))
        self.assertGreater(len(result), 2)
        self.assertTrue(all(line in body and len(line) <= 950 for line in result))

    def test_empty_and_bounds(self):
        lookup = Mock(return_value={"results": []})
        fetch = Mock()
        for limit in (0, 6):
            with self.assertRaises(ValueError):
                search("вопрос", ["первая", "вторая"], "arbitral", limit, lookup=lookup, fetch=fetch)
        self.assertEqual(search("вопрос", ["первая", "вторая"], "arbitral", 1,
                                lookup=lookup, fetch=fetch)["results"], [])
        fetch.assert_not_called()

    def test_article_spelling_and_short_legal_terms(self):
        self.assertTrue(any("61.2" in q for q in formulations("Что означает ст. 61.2 закона?", None)))
        self.assertTrue(any("жильё" in q for q in formulations("Как вернуть деньги за жильё?", None)))
        article = "Суд установил сумму НДС ИП исходя из фактически полученных в отчётном периоде платежей."
        res = search("НДС ИП", None, "arbitral", 1, lookup=lambda **kw: {"results": [{"url": "one"}]},
                     fetch=lambda *args: {"text": article})
        self.assertEqual(res["results"][0]["excerpts"], [article])

    def test_partial_timeout_and_unconfirmed_search(self):
        from threading import Event
        from unittest.mock import patch
        import server
        with patch.object(server, "_search_section", return_value=(None, None)):
            self.assertEqual(len(server.search_relevant_practice("НДС ИП", ["вычет НДС", "уплата НДС"])["warnings"]), 2)
        available = {"url": "complete"}
        gate = Event()
        def fetch(url, scope):
            if url == "slow":
                gate.wait(5)
            return {"text": "Суд установил сумму НДС ИП исходя из фактически полученных в отчётном периоде платежей."}
        try:
            with patch("practice_search.TOTAL_IO_BUDGET", 0.12):
                res = search("НДС ИП", None, "arbitral", 2,
                             lookup=lambda **kw: {"results": [available, {"url": "slow"}]}, fetch=fetch)
            self.assertEqual(res["read"], 1)
            self.assertEqual(res["results"][0]["url"], "complete")
            self.assertTrue(res["warnings"])
        finally:
            gate.set()
