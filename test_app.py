"""Unit tests for HeliosDocAI core functions."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app


# ---------------------------------------------------------------------------
# safe_latin1
# ---------------------------------------------------------------------------


class TestSafeLatin1:
    def test_ascii_passthrough(self):
        assert app.safe_latin1("Hello World") == "Hello World"

    def test_german_umlauts(self):
        result = app.safe_latin1("Lüftung Wärmerückgewinnung")
        assert "ftung" in result
        assert "rme" in result

    def test_emoji_replaced(self):
        result = app.safe_latin1("Test 🌀 Emoji")
        assert "🌀" not in result
        assert "Test" in result

    def test_empty_string(self):
        assert app.safe_latin1("") == ""

    def test_chinese_chars_replaced(self):
        result = app.safe_latin1("你好世界")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _product_to_text
# ---------------------------------------------------------------------------


class TestProductToText:
    def test_basic_product(self):
        product = {
            "name": "ELS NFC",
            "category": "Einrohr-Lüftungssystem",
            "description": "Multifunktionaler Ventilatoreinsatz",
            "use_cases": ["Bad", "WC"],
            "highlights": ["NFC-Parametrierung"],
            "specs": {
                "airflow_m3h": [7.5, 100],
                "sound_pressure_dba": {"low": 26, "high": 47},
                "nfc_configurable": True,
                "sensor": None,
                "wrg_efficiency_pct": None,
                "ex_rating": None,
            },
        }
        result = app._product_to_text(product)
        assert "ELS NFC" in result
        assert "Bad" in result
        assert "7.5-100 m³/h" in result
        assert "26-47 dB(A)" in result
        assert "NFC-Konfiguration möglich" in result

    def test_product_with_scalar_airflow(self):
        product = {
            "name": "Test",
            "category": "",
            "description": "",
            "use_cases": [],
            "highlights": [],
            "specs": {"airflow_m3h": 200},
        }
        result = app._product_to_text(product)
        assert "200 m³/h" in result

    def test_product_with_sensor(self):
        product = {
            "name": "Test",
            "specs": {"sensor": "VOC"},
            "use_cases": [],
            "highlights": [],
        }
        result = app._product_to_text(product)
        assert "Sensor: VOC" in result

    def test_product_with_wrg(self):
        product = {
            "name": "Test",
            "specs": {"wrg_efficiency_pct": 73},
            "use_cases": [],
            "highlights": [],
        }
        result = app._product_to_text(product)
        assert "73%" in result

    def test_product_with_ex_rating(self):
        product = {
            "name": "Test",
            "specs": {"ex_rating": "II 2G Ex"},
            "use_cases": [],
            "highlights": [],
        }
        result = app._product_to_text(product)
        assert "II 2G Ex" in result

    def test_product_missing_optional_fields(self):
        product = {"name": "Minimal", "specs": {}, "use_cases": [], "highlights": []}
        result = app._product_to_text(product)
        assert "Minimal" in result


# ---------------------------------------------------------------------------
# generate_training_data
# ---------------------------------------------------------------------------


class TestGenerateTrainingData:
    def test_default_sample_count(self):
        df = app.generate_training_data()
        assert len(df) == 60

    def test_custom_sample_count(self):
        df = app.generate_training_data(n=10)
        assert len(df) == 10

    def test_columns_present(self):
        df = app.generate_training_data(n=5)
        expected_cols = [
            "room_size_m2",
            "air_changes_per_h",
            "ceiling_height_m",
            "wrg_efficiency_pct",
            "hours_per_day",
            "delta_t_k",
            "heating_days",
            "energy_saved_kwh",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_values_in_range(self):
        df = app.generate_training_data(n=100)
        assert df["room_size_m2"].min() >= 20
        assert df["room_size_m2"].max() <= 300
        assert df["air_changes_per_h"].min() >= 0.5
        assert df["air_changes_per_h"].max() <= 5.0
        assert df["ceiling_height_m"].min() >= 2.4
        assert df["ceiling_height_m"].max() <= 4.0
        assert df["wrg_efficiency_pct"].min() >= 50
        assert df["wrg_efficiency_pct"].max() <= 95
        assert df["energy_saved_kwh"].min() > 0

    def test_reproducibility(self):
        df1 = app.generate_training_data(n=10)
        df2 = app.generate_training_data(n=10)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_nan_values(self):
        df = app.generate_training_data(n=50)
        assert not df.isnull().any().any()


# ---------------------------------------------------------------------------
# load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_load_existing_file(self, tmp_path):
        data = {"key": "value", "number": 42}
        file = tmp_path / "test.json"
        file.write_text(json.dumps(data), encoding="utf-8")
        result = app.load_json(file)
        assert result == data

    def test_load_nonexistent_file(self, tmp_path):
        result = app.load_json(tmp_path / "nonexistent.json")
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        file = tmp_path / "bad.json"
        file.write_text("not valid json {{{", encoding="utf-8")
        result = app.load_json(file)
        assert result is None

    def test_load_products_catalog(self):
        result = app.load_json(app.PRODUCTS_PATH)
        assert result is not None
        assert "products" in result
        assert len(result["products"]) > 0

    def test_load_demo_extraction(self):
        result = app.load_json(app.DEMO_EXTRACTION)
        assert result is not None
        assert "products" in result
        assert "document_type" in result
        assert "confidence" in result


# ---------------------------------------------------------------------------
# read_prompt
# ---------------------------------------------------------------------------


class TestReadPrompt:
    def test_read_existing_prompt(self):
        result = app.read_prompt(app.PROMPT_EXTRACTION)
        assert len(result) > 0
        assert "JSON" in result

    def test_read_nonexistent_prompt(self, tmp_path):
        result = app.read_prompt(tmp_path / "missing.md")
        assert result == ""


# ---------------------------------------------------------------------------
# _filter_search_results
# ---------------------------------------------------------------------------


class TestFilterSearchResults:
    @pytest.fixture()
    def products_by_id(self):
        return {
            "P1": {
                "id": "P1",
                "name": "Quiet Fan",
                "specs": {
                    "sound_pressure_dba": {"low": 20, "high": 35},
                    "airflow_m3h": [50, 100, 200],
                    "nfc_configurable": True,
                    "wrg_efficiency_pct": 73,
                    "ex_rating": None,
                },
            },
            "P2": {
                "id": "P2",
                "name": "Loud Fan",
                "specs": {
                    "sound_pressure_dba": {"low": 45, "high": 60},
                    "airflow_m3h": [100, 300],
                    "nfc_configurable": False,
                    "wrg_efficiency_pct": None,
                    "ex_rating": None,
                },
            },
            "P3": {
                "id": "P3",
                "name": "Ex Fan",
                "specs": {
                    "sound_pressure_dba": {"low": 30, "high": 50},
                    "airflow_m3h": 150,
                    "nfc_configurable": False,
                    "wrg_efficiency_pct": None,
                    "ex_rating": "II 2G Ex",
                },
            },
        }

    @pytest.fixture()
    def mock_results(self):
        return {
            "ids": [["P1", "P2", "P3"]],
            "distances": [[0.2, 0.4, 0.5]],
        }

    def test_no_filters(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, False, False, False
        )
        assert len(filtered) == 3

    def test_sound_filter(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 30, 0, False, False, False
        )
        assert len(filtered) == 2
        names = [r["product"]["name"] for r in filtered]
        assert "Loud Fan" not in names

    def test_airflow_filter(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 250, False, False, False
        )
        names = [r["product"]["name"] for r in filtered]
        assert "Quiet Fan" not in names

    def test_nfc_filter(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, True, False, False
        )
        assert len(filtered) == 1
        assert filtered[0]["product"]["name"] == "Quiet Fan"

    def test_wrg_filter(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, False, True, False
        )
        assert len(filtered) == 1
        assert filtered[0]["product"]["name"] == "Quiet Fan"

    def test_ex_filter(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, False, False, True
        )
        assert len(filtered) == 1
        assert filtered[0]["product"]["name"] == "Ex Fan"

    def test_empty_results(self, products_by_id):
        empty = {"ids": [], "distances": []}
        filtered = app._filter_search_results(
            empty, products_by_id, 60, 0, False, False, False
        )
        assert len(filtered) == 0

    def test_max_5_results(self, products_by_id):
        results = {
            "ids": [["P1", "P2", "P3", "P1", "P2", "P3"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
        }
        filtered = app._filter_search_results(
            results, products_by_id, 60, 0, False, False, False
        )
        assert len(filtered) <= 5

    def test_score_calculation(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, False, False, False
        )
        assert filtered[0]["score"] == 0.8  # 1 - 0.2
        assert filtered[1]["score"] == 0.6  # 1 - 0.4

    def test_ranking_order(self, mock_results, products_by_id):
        filtered = app._filter_search_results(
            mock_results, products_by_id, 60, 0, False, False, False
        )
        for i, r in enumerate(filtered):
            assert r["rank"] == i + 1

    def test_unknown_product_id_skipped(self):
        results = {"ids": [["UNKNOWN"]], "distances": [[0.1]]}
        filtered = app._filter_search_results(results, {}, 60, 0, False, False, False)
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# HeliosReport (PDF generation)
# ---------------------------------------------------------------------------


class TestHeliosReport:
    def test_empty_report_generates(self):
        result = app.build_pdf_report()
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result[:4] == b"%PDF"

    def test_report_with_extraction_data(self):
        import streamlit as st

        st.session_state["extracted_data"] = {
            "document_type": "datenblatt",
            "confidence": 0.9,
            "products": [
                {"product_name": "ELS NFC", "airflow_m3h": 100}
            ],
        }
        result = app.build_pdf_report()
        assert isinstance(result, bytes)
        assert len(result) > 100
        del st.session_state["extracted_data"]

    def test_report_with_energy_data(self):
        import streamlit as st

        st.session_state["energy_result"] = {
            "kwh": 5000,
            "euro": 1500,
            "price_ct": 30,
        }
        result = app.build_pdf_report()
        assert isinstance(result, bytes)
        del st.session_state["energy_result"]


# ---------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------


class TestConstants:
    def test_els_nfc_params_structure(self):
        params = app.ELS_NFC_PARAMS
        assert "models" in params
        assert len(params["models"]) == 5
        assert "airflow_steps_m3h" in params
        assert params["airflow_steps_m3h"] == sorted(params["airflow_steps_m3h"])

    def test_article_numbers_match_models(self):
        for model in app.ELS_NFC_PARAMS["models"]:
            assert model in app.ARTICLE_NUMBERS, f"Missing article number for {model}"

    def test_sensor_options_reference_valid_models(self):
        for model in app.ELS_NFC_PARAMS["sensor_options"]:
            assert model in app.ELS_NFC_PARAMS["models"], f"Sensor option for unknown model: {model}"

    def test_factory_defaults_in_airflow_steps(self):
        steps = app.ELS_NFC_PARAMS["airflow_steps_m3h"]
        for val in app.ELS_NFC_PARAMS["factory_defaults_m3h"].values():
            assert val in steps, f"Factory default {val} not in airflow steps"


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------


class TestEnergyModel:
    def test_model_trains_successfully(self):
        model, feature_cols = app.train_energy_model()
        assert model is not None
        assert len(feature_cols) == 7

    def test_model_predicts(self):
        model, feature_cols = app.train_energy_model()
        X = pd.DataFrame(
            [{
                "room_size_m2": 80,
                "air_changes_per_h": 2.0,
                "ceiling_height_m": 2.7,
                "wrg_efficiency_pct": 75,
                "hours_per_day": 12,
                "delta_t_k": 20,
                "heating_days": 210,
            }]
        )
        prediction = model.predict(X)[0]
        assert prediction > 0
        assert isinstance(prediction, (float, np.floating))

    def test_higher_wrg_more_savings(self):
        model, _ = app.train_energy_model()
        base = {
            "room_size_m2": 80,
            "air_changes_per_h": 2.0,
            "ceiling_height_m": 2.7,
            "wrg_efficiency_pct": 50,
            "hours_per_day": 12,
            "delta_t_k": 20,
            "heating_days": 210,
        }
        high_wrg = {**base, "wrg_efficiency_pct": 95}
        pred_low = model.predict(pd.DataFrame([base]))[0]
        pred_high = model.predict(pd.DataFrame([high_wrg]))[0]
        assert pred_high > pred_low

    def test_larger_room_more_savings(self):
        model, _ = app.train_energy_model()
        base = {
            "room_size_m2": 30,
            "air_changes_per_h": 2.0,
            "ceiling_height_m": 2.7,
            "wrg_efficiency_pct": 75,
            "hours_per_day": 12,
            "delta_t_k": 20,
            "heating_days": 210,
        }
        large = {**base, "room_size_m2": 250}
        pred_small = model.predict(pd.DataFrame([base]))[0]
        pred_large = model.predict(pd.DataFrame([large]))[0]
        assert pred_large > pred_small
