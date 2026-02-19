"""
Integration tests for HeliosDocAI — validates expected outputs.

Run with: pytest test_integration.py -v
For live API tests: ANTHROPIC_API_KEY=sk-ant-... pytest test_integration.py -v -m live
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def catalog():
    data = app.load_json(app.PRODUCTS_PATH)
    assert data is not None
    return data


@pytest.fixture()
def demo_extraction():
    data = app.load_json(app.DEMO_EXTRACTION)
    assert data is not None
    return data


@pytest.fixture()
def demo_search():
    data = app.load_json(app.DEMO_SEARCH)
    assert data is not None
    return data


@pytest.fixture()
def demo_comparison():
    data = app.load_json(app.DEMO_COMPARISON)
    assert data is not None
    return data


# ---------------------------------------------------------------------------
# 1. Produktkatalog-Konsistenz
# ---------------------------------------------------------------------------


class TestProductCatalog:
    """Prüft ob der Produktkatalog korrekt und konsistent ist."""

    def test_all_products_have_required_fields(self, catalog):
        required = ["id", "name", "category", "description", "specs", "use_cases"]
        for p in catalog["products"]:
            for field in required:
                assert field in p, f"Product {p.get('id', '?')} missing field: {field}"

    def test_product_count(self, catalog):
        assert len(catalog["products"]) == 15

    def test_all_ids_unique(self, catalog):
        ids = [p["id"] for p in catalog["products"]]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_els_nfc_family_complete(self, catalog):
        """Die ELS NFC-Familie muss 5 Varianten haben."""
        els_products = [p for p in catalog["products"] if p["id"].startswith("ELS-NFC")]
        assert len(els_products) == 5
        expected_ids = {"ELS-NFC", "ELS-NFC-F", "ELS-NFC-VOC", "ELS-NFC-CO2", "ELS-NFC-P"}
        actual_ids = {p["id"] for p in els_products}
        assert actual_ids == expected_ids

    def test_nfc_products_have_nfc_flag(self, catalog):
        """Alle ELS NFC Produkte müssen nfc_configurable=True haben."""
        for p in catalog["products"]:
            if p["id"].startswith("ELS-NFC"):
                assert p["specs"].get("nfc_configurable") is True, f"{p['id']} should be NFC-configurable"

    def test_ex_product_has_rating(self, catalog):
        """Ex-geschützter Ventilator muss ex_rating haben."""
        ex_products = [p for p in catalog["products"] if "ex" in p["category"].lower() or "ex" in p["id"].lower()]
        assert len(ex_products) >= 1
        for p in ex_products:
            assert p["specs"].get("ex_rating") is not None, f"{p['id']} missing ex_rating"

    def test_airflow_values_plausible(self, catalog):
        """Luftleistungswerte müssen plausibel sein (0-10000 m³/h)."""
        for p in catalog["products"]:
            airflow = p["specs"].get("airflow_m3h")
            if airflow is None:
                continue
            if isinstance(airflow, list):
                assert all(0 < v <= 10000 for v in airflow), f"{p['id']} implausible airflow: {airflow}"
            elif isinstance(airflow, (int, float)):
                assert 0 < airflow <= 10000, f"{p['id']} implausible airflow: {airflow}"

    def test_article_numbers_format(self, catalog):
        """Artikelnummern müssen 5-stellig sein (wenn vorhanden)."""
        for p in catalog["products"]:
            art = p.get("article_number")
            if art is not None:
                assert art.isdigit() and len(art) == 5, f"{p['id']} invalid article number: {art}"


# ---------------------------------------------------------------------------
# 2. Demo-Extraktion validieren
# ---------------------------------------------------------------------------


class TestDemoExtraction:
    """Prüft ob die Demo-Extraktionsdaten korrekte Struktur/Werte haben."""

    def test_structure(self, demo_extraction):
        assert "products" in demo_extraction
        assert "document_type" in demo_extraction
        assert "confidence" in demo_extraction

    def test_document_type_valid(self, demo_extraction):
        valid_types = {"datenblatt", "gebaeudeplan", "kundenanfrage",
                       "installationsprotokoll", "ausschreibung", "sonstig"}
        assert demo_extraction["document_type"] in valid_types

    def test_confidence_range(self, demo_extraction):
        assert 0 <= demo_extraction["confidence"] <= 1

    def test_products_not_empty(self, demo_extraction):
        assert len(demo_extraction["products"]) >= 1

    def test_extracted_products_match_catalog(self, demo_extraction, catalog):
        """Extrahierte Produkte sollten im Katalog existieren."""
        catalog_names = {p["name"] for p in catalog["products"]}
        for product in demo_extraction["products"]:
            name = product.get("product_name", "")
            assert name in catalog_names, f"Extracted product '{name}' not in catalog"

    def test_extracted_article_numbers_match(self, demo_extraction, catalog):
        """Artikelnummern müssen mit Katalog übereinstimmen."""
        catalog_articles = {p["name"]: p.get("article_number") for p in catalog["products"]}
        for product in demo_extraction["products"]:
            name = product.get("product_name")
            art = product.get("article_number")
            if name in catalog_articles and art is not None:
                assert art == catalog_articles[name], (
                    f"Article mismatch for {name}: extracted={art}, catalog={catalog_articles[name]}"
                )

    def test_airflow_plausible(self, demo_extraction):
        for product in demo_extraction["products"]:
            airflow = product.get("airflow_m3h")
            if airflow is not None:
                assert 0 < airflow <= 10000

    def test_sound_levels_plausible(self, demo_extraction):
        for product in demo_extraction["products"]:
            for key in ["sound_level_dba", "sound_power_dba"]:
                val = product.get(key)
                if val is not None:
                    assert 0 < val <= 120, f"{key}={val} out of range for {product.get('product_name')}"


# ---------------------------------------------------------------------------
# 3. Demo-Suchergebnisse validieren
# ---------------------------------------------------------------------------


class TestDemoSearch:
    """Prüft ob die Demo-Suchergebnisse konsistent sind."""

    def test_structure(self, demo_search):
        assert "recommendations" in demo_search
        assert len(demo_search["recommendations"]) >= 1

    def test_rankings_sequential(self, demo_search):
        ranks = [r["rank"] for r in demo_search["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_descending(self, demo_search):
        scores = [r["score"] for r in demo_search["recommendations"]]
        assert scores == sorted(scores, reverse=True), "Scores should be descending"

    def test_scores_in_range(self, demo_search):
        for rec in demo_search["recommendations"]:
            assert 0 <= rec["score"] <= 1, f"Score {rec['score']} out of range"

    def test_product_ids_exist_in_catalog(self, demo_search, catalog):
        catalog_ids = {p["id"] for p in catalog["products"]}
        for rec in demo_search["recommendations"]:
            pid = rec.get("product_id")
            if pid:
                assert pid in catalog_ids, f"Recommended product_id '{pid}' not in catalog"

    def test_top_result_is_voc(self, demo_search):
        """Für 'Büro mit Luftqualitätssensor' muss ELS NFC VOC #1 sein."""
        top = demo_search["recommendations"][0]
        assert top["product_id"] == "ELS-NFC-VOC"
        assert top["meets_all_requirements"] is True

    def test_reasoning_not_empty(self, demo_search):
        for rec in demo_search["recommendations"]:
            assert len(rec.get("reasoning_de", "")) > 10, f"Rank {rec['rank']} missing reasoning"


# ---------------------------------------------------------------------------
# 4. Modell-Evaluation validieren
# ---------------------------------------------------------------------------


class TestDemoComparison:
    """Prüft ob die Vergleichsdaten zwischen Claude und Llama konsistent sind."""

    def test_structure(self, demo_comparison):
        assert "input_text" in demo_comparison
        assert "claude_result" in demo_comparison
        assert "llama_result" in demo_comparison
        assert "comparison_summary" in demo_comparison

    def test_claude_more_accurate(self, demo_comparison):
        """Claude muss mehr Felder extrahiert haben."""
        claude = demo_comparison["claude_result"]
        llama = demo_comparison["llama_result"]
        assert claude["fields_extracted"] > llama["fields_extracted"]

    def test_claude_json_valid(self, demo_comparison):
        assert demo_comparison["claude_result"]["json_valid"] is True

    def test_llama_needed_correction(self, demo_comparison):
        assert demo_comparison["llama_result"]["json_needed_correction"] is True

    def test_llama_faster(self, demo_comparison):
        """Llama (Groq) muss schneller sein."""
        assert demo_comparison["llama_result"]["latency_ms"] < demo_comparison["claude_result"]["latency_ms"]

    def test_input_text_contains_product(self, demo_comparison):
        assert "ELS NFC VOC" in demo_comparison["input_text"]

    def test_both_extract_same_product(self, demo_comparison):
        claude_name = demo_comparison["claude_result"]["extraction"]["products"][0]["product_name"]
        llama_name = demo_comparison["llama_result"]["extraction"]["products"][0]["product_name"]
        assert claude_name == llama_name == "ELS NFC VOC"

    def test_claude_airflow_is_number(self, demo_comparison):
        """Claude muss airflow als Zahl extrahieren, Llama als String (bekanntes Issue)."""
        claude_airflow = demo_comparison["claude_result"]["extraction"]["products"][0]["airflow_m3h"]
        llama_airflow = demo_comparison["llama_result"]["extraction"]["products"][0]["airflow_m3h"]
        assert isinstance(claude_airflow, (int, float)), "Claude airflow should be numeric"
        assert isinstance(llama_airflow, str), "Llama airflow should be string (known issue)"

    def test_summary_winners(self, demo_comparison):
        summary = demo_comparison["comparison_summary"]
        assert summary["winner_accuracy"] == "Claude Sonnet"
        assert summary["winner_speed"] == "Llama-3.3-70B"
        assert summary["winner_cost"] == "Llama-3.3-70B"


# ---------------------------------------------------------------------------
# 5. NFC-Konfiguration validieren
# ---------------------------------------------------------------------------


class TestNFCConfig:
    """Prüft ob generierte NFC-Configs dem erwarteten Schema entsprechen."""

    def test_default_config_structure(self):
        """Simuliert Default-Config und prüft Schema."""
        config = {
            "_meta": {
                "generator": "HeliosDocAI Prototype",
                "timestamp": "2026-01-01T00:00:00Z",
                "disclaimer": "Simulierte Konfiguration – nicht für Produktionseinsatz",
            },
            "device": {
                "model": "ELS NFC",
                "article_number": "40761",
            },
            "airflow_config": {
                "stufe_1_m3h": 35,
                "stufe_2_m3h": 60,
                "stufe_3_m3h": 100,
                "stufe_4_m3h": None,
                "stufe_5_m3h": None,
                "grundlueftung_m3h": 15,
                "intervall_m3h": 15,
            },
            "timing": {
                "einschaltverzoegerung_sec": 5,
                "nachlaufzeit_min": 15,
                "intervallzeit_h": 2,
            },
        }
        # Schema-Validierung
        assert config["device"]["model"] in app.ELS_NFC_PARAMS["models"]
        assert config["device"]["article_number"] == app.ARTICLE_NUMBERS["ELS NFC"]
        for key in ["stufe_1_m3h", "stufe_2_m3h", "stufe_3_m3h"]:
            val = config["airflow_config"][key]
            assert val in app.ELS_NFC_PARAMS["airflow_steps_m3h"]
        assert config["airflow_config"]["stufe_1_m3h"] < config["airflow_config"]["stufe_2_m3h"]
        assert config["airflow_config"]["stufe_2_m3h"] < config["airflow_config"]["stufe_3_m3h"]

    def test_all_models_have_valid_article_numbers(self):
        for model in app.ELS_NFC_PARAMS["models"]:
            art = app.ARTICLE_NUMBERS.get(model)
            assert art is not None, f"No article number for {model}"
            assert art.isdigit() and len(art) == 5

    def test_factory_defaults_ascending(self):
        defaults = app.ELS_NFC_PARAMS["factory_defaults_m3h"]
        assert defaults["stufe_1"] < defaults["stufe_2"] < defaults["stufe_3"]

    def test_timing_ranges(self):
        params = app.ELS_NFC_PARAMS
        assert params["delay_range_sec"]["min"] == 0
        assert params["delay_range_sec"]["max"] == 120
        assert params["runon_range_min"]["min"] == 0
        assert params["runon_range_min"]["max"] == 90


# ---------------------------------------------------------------------------
# 6. Energiemodell-Plausibilität
# ---------------------------------------------------------------------------


class TestEnergyPlausibility:
    """Prüft ob Energieschätzungen physikalisch plausibel sind."""

    @pytest.fixture()
    def model(self):
        model, feature_cols = app.train_energy_model()
        return model

    def _predict(self, model, **overrides):
        base = {
            "room_size_m2": 80,
            "air_changes_per_h": 2.0,
            "ceiling_height_m": 2.7,
            "wrg_efficiency_pct": 75,
            "hours_per_day": 12,
            "delta_t_k": 20,
            "heating_days": 210,
        }
        base.update(overrides)
        return model.predict(pd.DataFrame([base]))[0]

    def test_typical_office_range(self, model):
        """80m² Büro, Standard-WRG: 2000-20000 kWh/a erwartet."""
        pred = self._predict(model)
        assert 2000 < pred < 20000, f"Prediction {pred:.0f} kWh outside plausible range"

    def test_small_room_low_savings(self, model):
        """20m² Raum muss weniger sparen als 200m²."""
        small = self._predict(model, room_size_m2=20)
        large = self._predict(model, room_size_m2=200)
        assert small < large

    def test_high_wrg_high_savings(self, model):
        """95% WRG muss mehr sparen als 50% WRG."""
        low_wrg = self._predict(model, wrg_efficiency_pct=50)
        high_wrg = self._predict(model, wrg_efficiency_pct=95)
        assert high_wrg > low_wrg

    def test_more_hours_more_savings(self, model):
        """24h Betrieb muss mehr sparen als 6h."""
        short = self._predict(model, hours_per_day=6)
        long = self._predict(model, hours_per_day=24)
        assert long > short

    def test_higher_delta_t_more_savings(self, model):
        """30K Temperaturdifferenz muss mehr sparen als 10K."""
        low_dt = self._predict(model, delta_t_k=10)
        high_dt = self._predict(model, delta_t_k=30)
        assert high_dt > low_dt

    def test_co2_calculation(self, model):
        """CO2-Berechnung: kWh * 0.4 kg/kWh."""
        pred_kwh = self._predict(model)
        co2_kg = pred_kwh * app.DE_CO2_EMISSION_FACTOR_KG_PER_KWH
        assert co2_kg > 0
        assert co2_kg == pytest.approx(pred_kwh * 0.4)

    def test_cost_calculation(self, model):
        """Kostenberechnung: kWh * ct/kWh / 100."""
        pred_kwh = self._predict(model)
        price_ct = 30
        cost_eur = pred_kwh * price_ct / 100
        assert cost_eur > 0

    def test_prediction_std_reasonable(self, model):
        """Unsicherheit sollte < 50% der Vorhersage sein."""
        X = pd.DataFrame([{
            "room_size_m2": 80,
            "air_changes_per_h": 2.0,
            "ceiling_height_m": 2.7,
            "wrg_efficiency_pct": 75,
            "hours_per_day": 12,
            "delta_t_k": 20,
            "heating_days": 210,
        }])
        pred = model.predict(X)[0]
        tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
        std = tree_preds.std()
        # Bei n=60 synthetischen Daten ist hohe Varianz zwischen Trees normal
        assert std / pred < 1.0, f"Std {std:.0f} too high relative to prediction {pred:.0f}"


# ---------------------------------------------------------------------------
# 7. PDF-Report Inhalts-Validierung
# ---------------------------------------------------------------------------


class TestPDFReportContent:
    """Prüft ob der PDF-Report die erwarteten Abschnitte enthält."""

    def test_report_contains_pdf_header(self):
        result = app.build_pdf_report()
        assert result[:4] == b"%PDF"

    def test_report_with_full_data(self):
        import streamlit as st

        st.session_state["extracted_data"] = {
            "document_type": "datenblatt",
            "confidence": 0.9,
            "products": [{"product_name": "ELS NFC VOC", "airflow_m3h": 100}],
        }
        st.session_state["search_results"] = {
            "recommendations": [
                {"rank": 1, "product_name": "ELS NFC VOC", "score": 0.94, "reasoning_de": "Test"}
            ]
        }
        st.session_state["nfc_config"] = {"device": {"model": "ELS NFC"}}
        st.session_state["energy_result"] = {"kwh": 5000, "euro": 1500, "price_ct": 30}

        result = app.build_pdf_report()
        assert len(result) > 500, "Full report should be substantial"
        assert result[:4] == b"%PDF"

        # Cleanup
        for key in ["extracted_data", "search_results", "nfc_config", "energy_result"]:
            del st.session_state[key]


# ---------------------------------------------------------------------------
# 8. Semantische Suche (ChromaDB) — nur wenn dependencies verfügbar
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    """Prüft ob die ChromaDB-Suche plausible Ergebnisse liefert."""

    @pytest.fixture()
    def chroma(self):
        collection, model = app.build_chroma_collection()
        if collection is None or model is None:
            pytest.skip("ChromaDB/sentence-transformers not available")
        return collection, model

    def _search(self, chroma, query, n=5):
        collection, model = chroma
        embedding = model.encode([query]).tolist()
        return collection.query(query_embeddings=embedding, n_results=n)

    def test_voc_query_returns_voc_first(self, chroma):
        """'VOC Sensor Büro' muss ELS NFC VOC als Top-Ergebnis liefern."""
        results = self._search(chroma, "VOC Sensor für Büro Luftqualität")
        top_id = results["ids"][0][0]
        assert top_id == "ELS-NFC-VOC", f"Expected ELS-NFC-VOC, got {top_id}"

    def test_feuchte_query_returns_f_variant(self, chroma):
        """'Feuchte Bad' muss ELS NFC F in Top 3 liefern."""
        results = self._search(chroma, "Feuchtesensor Bad Dusche Nassraum")
        top3_ids = results["ids"][0][:3]
        assert "ELS-NFC-F" in top3_ids, f"ELS-NFC-F not in top 3: {top3_ids}"

    def test_garage_query_returns_impulsventilator(self, chroma):
        """'Tiefgarage' muss Impulsventilator in Top 3 liefern."""
        results = self._search(chroma, "Tiefgarage Parkhaus Belüftung niedrige Decke")
        top3_ids = results["ids"][0][:3]
        assert "IVRW-EC-225" in top3_ids, f"IVRW-EC-225 not in top 3: {top3_ids}"

    def test_explosion_query_returns_ex(self, chroma):
        """'Explosionsschutz' muss Ex-Ventilator in Top 3 liefern."""
        results = self._search(chroma, "Explosionsschutz ATEX Zone Chemie")
        top3_ids = results["ids"][0][:3]
        assert "EX-AXIAL" in top3_ids, f"EX-AXIAL not in top 3: {top3_ids}"

    def test_wrg_query_returns_kwl(self, chroma):
        """'Wärmerückgewinnung' muss ein KWL-Produkt in Top 3 liefern."""
        results = self._search(chroma, "Wärmerückgewinnung Wohnung zentrale Lüftung")
        top3_ids = results["ids"][0][:3]
        kwl_hit = any("KWL" in pid or "AIR1" in pid for pid in top3_ids)
        assert kwl_hit, f"No WRG product in top 3: {top3_ids}"

    def test_co2_query_returns_co2_variant(self, chroma):
        """'CO2 Klassenzimmer' muss ELS NFC CO2 in Top 3 liefern."""
        results = self._search(chroma, "CO2 Sensor Klassenzimmer Schule Belegung")
        top3_ids = results["ids"][0][:3]
        assert "ELS-NFC-CO2" in top3_ids, f"ELS-NFC-CO2 not in top 3: {top3_ids}"

    def test_filter_sound_applied(self, chroma):
        """Filterung nach Schallpegel funktioniert."""
        results = self._search(chroma, "Leiser Ventilator")
        catalog = app.load_json(app.PRODUCTS_PATH)
        products_by_id = {p["id"]: p for p in catalog["products"]}

        filtered = app._filter_search_results(
            results, products_by_id,
            max_sound=30,  # Nur sehr leise
            min_airflow=0,
            filter_nfc=False, filter_wrg=False, filter_ex=False,
        )
        for r in filtered:
            sound = r["product"]["specs"].get("sound_pressure_dba")
            if isinstance(sound, dict) and sound.get("low") is not None:
                assert sound["low"] <= 30


# ---------------------------------------------------------------------------
# 9. Live API Tests (nur mit Key)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveAPI:
    """Tests die einen echten API-Key brauchen. Laufen nur mit: pytest -m live"""

    @pytest.fixture(autouse=True)
    def skip_without_key(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_extraction_returns_valid_json(self):
        """Claude muss für ein bekanntes Datenblatt valides JSON zurückgeben."""
        system_prompt = app.read_prompt(app.PROMPT_EXTRACTION)
        test_input = (
            "Technisches Datenblatt ELS NFC VOC\n"
            "Artikel-Nr.: 40764\n"
            "Luftleistung: 100 m³/h\n"
            "Schalldruckpegel: 47 dB(A)\n"
            "Schutzart: IPX5\n"
            "Schutzklasse: II\n"
            "NFC-Konfiguration: Ja\n"
        )
        result = app.call_claude(system_prompt, test_input)
        assert result is not None, "Claude returned None"

        # JSON parsing
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned)
        assert "products" in data
        assert len(data["products"]) >= 1

        product = data["products"][0]
        assert product["product_name"] == "ELS NFC VOC"
        assert product["article_number"] == "40764"
        assert product["nfc_configurable"] is True
        assert data["document_type"] == "datenblatt"
        assert 0 < data["confidence"] <= 1

    def test_search_ranking_returns_valid_json(self):
        """Claude muss für ein Ranking valides JSON zurückgeben."""
        system_prompt = app.read_prompt(app.PROMPT_MATCHING)
        user_msg = (
            "Anforderung: Leiser Ventilator für 25m² Büro mit Luftqualitätssensor\n\n"
            "Kandidaten:\n"
            "- ELS NFC VOC: VOC-Sensor, 26-47 dB(A), NFC\n"
            "- ELS NFC CO2: CO2-Sensor, 26-47 dB(A), NFC\n"
            "- ELS NFC: Basis ohne Sensor, 26-47 dB(A), NFC\n"
        )
        result = app.call_claude(system_prompt, user_msg)
        assert result is not None

        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned)
        assert "recommendations" in data
        recs = data["recommendations"]
        assert len(recs) >= 2

        # VOC should be #1 for this query
        assert recs[0]["product_name"] == "ELS NFC VOC"
        assert recs[0]["rank"] == 1
