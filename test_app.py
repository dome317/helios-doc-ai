"""Unit tests for HeliosDocAI core functions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# call_claude error handling
# ---------------------------------------------------------------------------


def _make_mock_response(text="test response"):
    """Create a mock Anthropic API response."""
    block = SimpleNamespace(text=text)
    usage = SimpleNamespace(input_tokens=100, output_tokens=50)
    return SimpleNamespace(content=[block], usage=usage)


class TestCallClaudeErrorHandling:
    """Deep tests for API error handling in call_claude."""

    @pytest.fixture(autouse=True)
    def setup_session(self):
        """Set up a fake API key in session state and clean up after."""
        import streamlit as st
        st.session_state["anthropic_api_key"] = "sk-ant-test-key-123"
        st.session_state["api_call_count"] = 0
        st.session_state["api_total_cost_usd"] = 0.0
        yield
        for key in ["anthropic_api_key", "api_call_count", "api_total_cost_usd"]:
            st.session_state.pop(key, None)

    def test_no_api_key_returns_none(self):
        """Without API key, call_claude must return None immediately."""
        import streamlit as st
        st.session_state["anthropic_api_key"] = ""
        result = app.call_claude("system", "user")
        assert result is None

    def test_usage_limit_returns_none(self):
        """When limit reached, call_claude must return None."""
        import streamlit as st
        st.session_state["api_call_count"] = app.MAX_API_CALLS_PER_SESSION
        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_successful_call(self, mock_anthropic_cls):
        """Successful API call returns text and tracks usage."""
        import streamlit as st
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response("result json")
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system prompt", "user content")
        assert result == "result json"
        assert st.session_state["api_call_count"] == 1
        assert st.session_state["api_total_cost_usd"] > 0

    @patch("anthropic.Anthropic")
    def test_auth_error_returns_none(self, mock_anthropic_cls):
        """AuthenticationError must show error and return None."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="invalid api key",
            response=MagicMock(status_code=401),
            body={"error": {"message": "invalid api key"}},
        )
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_rate_limit_returns_none(self, mock_anthropic_cls):
        """RateLimitError must show warning and return None."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body={"error": {"message": "rate limited"}},
        )
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_connection_error_returns_none(self, mock_anthropic_cls):
        """APIConnectionError must show error and return None."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock(),
        )
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_model_fallback_on_not_found(self, mock_anthropic_cls):
        """When primary model not found, fallback model should be tried."""
        import anthropic
        mock_client = MagicMock()
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model") == app.CLAUDE_MODEL:
                raise anthropic.NotFoundError(
                    message="model not found",
                    response=MagicMock(status_code=404),
                    body={"error": {"message": "model not found"}},
                )
            return _make_mock_response("fallback result")

        mock_client.messages.create.side_effect = side_effect
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result == "fallback result"
        assert call_count == 2  # Tried primary, then fallback

    @patch("anthropic.Anthropic")
    def test_both_models_not_found(self, mock_anthropic_cls):
        """When both models not found, must return None with error."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404),
            body={"error": {"message": "model not found"}},
        )
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_bad_request_returns_none(self, mock_anthropic_cls):
        """BadRequestError must show error and return None."""
        import anthropic
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.BadRequestError(
            message="invalid request",
            response=MagicMock(status_code=400),
            body={"error": {"message": "invalid request"}},
        )
        mock_anthropic_cls.return_value = mock_client

        result = app.call_claude("system", "user")
        assert result is None

    @patch("anthropic.Anthropic")
    def test_usage_tracking_increments(self, mock_anthropic_cls):
        """Each successful call must increment api_call_count."""
        import streamlit as st
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_mock_response("ok")
        mock_anthropic_cls.return_value = mock_client

        app.call_claude("s", "u")
        assert st.session_state["api_call_count"] == 1
        app.call_claude("s", "u")
        assert st.session_state["api_call_count"] == 2

    @patch("anthropic.Anthropic")
    def test_cost_calculation(self, mock_anthropic_cls):
        """Cost must be calculated from input/output tokens."""
        import streamlit as st
        block = SimpleNamespace(text="ok")
        usage = SimpleNamespace(input_tokens=1000, output_tokens=500)
        resp = SimpleNamespace(content=[block], usage=usage)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_anthropic_cls.return_value = mock_client

        app.call_claude("s", "u")
        cost = st.session_state["api_total_cost_usd"]
        # Expected: (1000 * 3 + 500 * 15) / 1_000_000 = 10500 / 1_000_000 = 0.0105
        assert abs(cost - 0.0105) < 0.0001


# ---------------------------------------------------------------------------
# PDF extraction flow
# ---------------------------------------------------------------------------


class TestPDFExtractionFlow:
    """Deep tests for the PDF reading and extraction pipeline."""

    def test_pymupdf_reads_helios_pdf(self):
        """Verify PyMuPDF can read the user's test PDF."""
        import fitz
        pdf_path = Path(r"C:\Users\tsats\Desktop\ELS_GA_98198.002_1205.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")
        doc = fitz.open(str(pdf_path))
        assert doc.page_count >= 1
        text = str(doc[0].get_text())
        assert len(text) > 100, "PDF should contain readable text"
        doc.close()

    def test_pymupdf_text_has_helios_content(self):
        """The Helios PDF must contain relevant technical content."""
        import fitz
        pdf_path = Path(r"C:\Users\tsats\Desktop\ELS_GA_98198.002_1205.pdf")
        if not pdf_path.exists():
            pytest.skip("Test PDF not available")
        doc = fitz.open(str(pdf_path))
        text = str(doc[0].get_text()).lower()
        doc.close()
        # Should contain electrical/installation terms
        assert any(
            term in text
            for term in ["els", "helios", "ventilator", "anschluss", "montage"]
        ), "PDF doesn't contain expected Helios terms"

    def test_extraction_prompt_exists_and_valid(self):
        """Extraction prompt must exist and contain JSON schema."""
        prompt = app.read_prompt(app.PROMPT_EXTRACTION)
        assert len(prompt) > 100
        assert "products" in prompt
        assert "JSON" in prompt
        assert "product_name" in prompt

    def test_matching_prompt_exists_and_valid(self):
        """Matching prompt must exist and contain recommendation schema."""
        prompt = app.read_prompt(app.PROMPT_MATCHING)
        assert len(prompt) > 100
        assert "recommendations" in prompt
        assert "rank" in prompt

    def test_json_cleaning_strips_markdown_fences(self):
        """JSON wrapped in ```json fences should be cleaned correctly."""
        wrapped = '```json\n{"products": []}\n```'
        cleaned = wrapped.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        assert data == {"products": []}

    def test_json_cleaning_handles_plain_json(self):
        """Plain JSON without fences should parse correctly."""
        plain = '{"products": [{"product_name": "ELS NFC"}]}'
        cleaned = plain.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        assert data["products"][0]["product_name"] == "ELS NFC"

    def test_json_cleaning_handles_triple_backticks_only(self):
        """JSON with just ``` (no language tag) should also work."""
        wrapped = '```\n{"test": true}\n```'
        cleaned = wrapped.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        assert data["test"] is True

    def test_pdf_size_limit_constant(self):
        """Size limit must be set."""
        assert app.MAX_PDF_SIZE_MB > 0
        assert app.MAX_PDF_SIZE_MB <= 50

    def test_pdf_page_limit_constant(self):
        """Page limit must be set."""
        assert app.MAX_PDF_PAGES > 0
        assert app.MAX_PDF_PAGES <= 200

    def test_extraction_chars_limit_set(self):
        """Extraction char limit must be reasonable."""
        assert 5000 <= app.MAX_EXTRACTION_CHARS <= 50000

    def test_model_fallback_constant_exists(self):
        """Fallback model ID must be defined."""
        assert hasattr(app, "CLAUDE_MODEL_FALLBACK")
        assert len(app.CLAUDE_MODEL_FALLBACK) > 0
        assert app.CLAUDE_MODEL_FALLBACK != app.CLAUDE_MODEL


# ---------------------------------------------------------------------------
# API connection test
# ---------------------------------------------------------------------------


class TestAPIConnectionTest:
    """Tests for the _test_api_connection helper."""

    def test_function_exists(self):
        """_test_api_connection must be defined."""
        assert callable(app._test_api_connection)
