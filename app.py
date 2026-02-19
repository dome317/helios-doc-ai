"""
HeliosDocAI – AI-gestützte Dokumentenanalyse für Helios Ventilatoren
Prototyp für Inhouse-Prozessoptimierung
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from sklearn.ensemble import RandomForestRegressor

load_dotenv()

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PRODUCTS_PATH = BASE_DIR / "helios_products.json"
DEMO_EXTRACTION = BASE_DIR / "demo_data" / "demo_extraction_result.json"
DEMO_SEARCH = BASE_DIR / "demo_data" / "demo_search_results.json"
DEMO_COMPARISON = BASE_DIR / "demo_data" / "demo_comparison.json"
PROMPT_EXTRACTION = BASE_DIR / "prompts" / "extraction_system.md"
PROMPT_MATCHING = BASE_DIR / "prompts" / "matching_system.md"

HELIOS_RED = "#E2001A"
MAX_EXTRACTION_CHARS = 15000
MAX_PDF_SIZE_MB = 20
MAX_PDF_PAGES = 100
DE_CO2_EMISSION_FACTOR_KG_PER_KWH = 0.4
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_API_CALLS_PER_SESSION = 20

ELS_NFC_PARAMS = {
    "models": ["ELS NFC", "ELS NFC F", "ELS NFC P", "ELS NFC VOC", "ELS NFC CO2"],
    "airflow_steps_m3h": [7.5, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100],
    "factory_defaults_m3h": {"stufe_1": 35, "stufe_2": 60, "stufe_3": 100},
    "max_stages": 5,
    "delay_range_sec": {"min": 0, "max": 120},
    "runon_range_min": {"min": 0, "max": 90},
    "interval_range_h": {"min": 0, "max": 24},
    "basic_ventilation_m3h": 15,
    "sensor_options": {
        "ELS NFC F": {"type": "humidity", "unit": "%rH", "threshold_adjustable": True},
        "ELS NFC P": {"type": "presence", "detection": "PIR"},
        "ELS NFC VOC": {
            "type": "voc",
            "unit": "voc",
            "threshold_range": [100, 450],
            "max_value_range": [100, 450],
            "modes": ["Komfort", "Intensiv"],
        },
        "ELS NFC CO2": {"type": "co2", "unit": "ppm", "threshold_adjustable": True},
    },
    "nfc_features": {
        "offline_config": True,
        "library_support": True,
        "status_readout": True,
        "error_reporting": True,
    },
}

ARTICLE_NUMBERS = {
    "ELS NFC": "40761",
    "ELS NFC F": "40762",
    "ELS NFC P": "40763",
    "ELS NFC VOC": "40764",
    "ELS NFC CO2": "40765",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str | None:
    """Return Anthropic API key from sidebar input, env, or None."""
    key = st.session_state.get("anthropic_api_key", "")
    if key:
        return key
    return os.getenv("ANTHROPIC_API_KEY")


def load_json(path: Path) -> dict | None:
    """Load JSON file with error handling. Returns None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Datei nicht gefunden: {path.name}")
        return None
    except json.JSONDecodeError:
        st.error(f"Ungültige JSON-Datei: {path.name}")
        return None
    except Exception:
        st.error(f"Fehler beim Laden von {path.name}")
        return None


def read_prompt(path: Path) -> str:
    """Read prompt file with error handling. Returns empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"Prompt-Datei nicht gefunden: {path.name}")
        return ""
    except Exception:
        st.warning(f"Fehler beim Laden von {path.name}")
        return ""


def safe_latin1(text: str) -> str:
    """Encode text to latin-1 safe form for PDF Helvetica font."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


@st.cache_resource
def load_product_catalog() -> dict | None:
    try:
        return load_json(PRODUCTS_PATH)
    except Exception:
        st.error("Produktkatalog konnte nicht geladen werden.")
        return None


@st.cache_resource
def build_chroma_collection():
    """Embed products and store in ChromaDB in-memory collection."""
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        catalog = load_json(PRODUCTS_PATH)
        if catalog is None:
            return None, None

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        client = chromadb.Client()

        try:
            client.delete_collection("helios_products")
        except Exception:
            pass

        collection = client.create_collection(
            name="helios_products", metadata={"hnsw:space": "cosine"}
        )

        ids = []
        documents = []
        metadatas = []

        for p in catalog["products"]:
            doc_text = _product_to_text(p)
            ids.append(p["id"])
            documents.append(doc_text)
            metadatas.append(
                {
                    "name": p["name"],
                    "category": p["category"],
                    "nfc": str(p["specs"].get("nfc_configurable", False)),
                    "use_cases": ", ".join(p.get("use_cases", [])),
                }
            )

        embeddings = model.encode(documents).tolist()
        collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        return collection, model
    except Exception:
        st.error("ChromaDB-Setup fehlgeschlagen. Bitte Seite neu laden.")
        return None, None


def _product_to_text(p: dict) -> str:
    """Create a searchable text representation of a product."""
    specs = p.get("specs", {})
    parts = [
        p["name"],
        p.get("category", ""),
        p.get("description", ""),
        f"Einsatz: {', '.join(p.get('use_cases', []))}",
        f"Highlights: {', '.join(p.get('highlights', []))}",
    ]
    airflow = specs.get("airflow_m3h")
    if airflow is not None:
        if isinstance(airflow, list):
            parts.append(f"Luftleistung: {min(airflow)}-{max(airflow)} m³/h")
        else:
            parts.append(f"Luftleistung: {airflow} m³/h")

    sound = specs.get("sound_pressure_dba")
    if isinstance(sound, dict):
        parts.append(
            f"Schallpegel: {sound.get('low', '?')}-{sound.get('high', '?')} dB(A)"
        )

    if specs.get("sensor"):
        parts.append(f"Sensor: {specs['sensor']}")
    if specs.get("nfc_configurable"):
        parts.append("NFC-Konfiguration möglich")
    if specs.get("wrg_efficiency_pct"):
        parts.append(f"WRG-Wirkungsgrad: {specs['wrg_efficiency_pct']}%")
    if specs.get("ex_rating"):
        parts.append(f"Explosionsschutz: {specs['ex_rating']}")

    return " | ".join(parts)


def call_claude(system_prompt: str, user_content: str) -> str | None:
    """Call Anthropic Claude API. Returns raw text or None on failure."""
    api_key = get_api_key()
    if not api_key:
        return None

    # Usage limit check
    call_count = st.session_state.get("api_call_count", 0)
    if call_count >= MAX_API_CALLS_PER_SESSION:
        st.warning(
            f"API-Limit erreicht ({MAX_API_CALLS_PER_SESSION} Aufrufe pro Session). "
            "Seite neu laden für neue Session."
        )
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        # Track usage
        st.session_state["api_call_count"] = call_count + 1
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_cost = st.session_state.get("api_total_cost_usd", 0.0)
        total_cost += (input_tokens * 3 + output_tokens * 15) / 1_000_000
        st.session_state["api_total_cost_usd"] = total_cost

        block = response.content[0]
        return getattr(block, "text", None)
    except Exception:
        st.warning("API-Aufruf fehlgeschlagen. Bitte API-Key prüfen.")
        return None


def generate_training_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    room_size = rng.uniform(20, 300, n)
    air_changes = rng.uniform(0.5, 5.0, n)
    ceiling_height = rng.uniform(2.4, 4.0, n)
    wrg_efficiency = rng.uniform(0.5, 0.95, n)
    hours_per_day = rng.uniform(6, 24, n)
    delta_t = rng.uniform(10, 30, n)
    heating_days = rng.uniform(180, 240, n)

    volume = room_size * ceiling_height
    energy_saved_kwh = (
        volume
        * air_changes
        * 0.34
        * delta_t
        * wrg_efficiency
        * hours_per_day
        * heating_days
        / 1000
    )
    energy_saved_kwh *= rng.uniform(0.85, 1.15, n)

    return pd.DataFrame(
        {
            "room_size_m2": room_size,
            "air_changes_per_h": air_changes,
            "ceiling_height_m": ceiling_height,
            "wrg_efficiency_pct": wrg_efficiency * 100,
            "hours_per_day": hours_per_day,
            "delta_t_k": delta_t,
            "heating_days": heating_days,
            "energy_saved_kwh": energy_saved_kwh,
        }
    )


@st.cache_resource
def train_energy_model():
    df = generate_training_data(60)
    feature_cols = [
        "room_size_m2",
        "air_changes_per_h",
        "ceiling_height_m",
        "wrg_efficiency_pct",
        "hours_per_day",
        "delta_t_k",
        "heating_days",
    ]
    X = df[feature_cols]
    y = df["energy_saved_kwh"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, feature_cols


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------


class HeliosReport(FPDF):
    HELIOS_RED = (226, 0, 26)
    HELIOS_BLACK = (26, 26, 26)
    HELIOS_GRAY = (245, 245, 245)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.HELIOS_RED)
        self.cell(0, 10, safe_latin1("HeliosDocAI – Analyse-Report"), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*self.HELIOS_RED)
        self.line(10, 22, 200, 22)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(
            0, 10,
            safe_latin1(f"Generiert von HeliosDocAI Prototyp | Seite {self.page_no()}"),
            align="C",
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*self.HELIOS_RED)
        self.cell(0, 8, safe_latin1(title), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*self.HELIOS_BLACK)
        self.set_font("Helvetica", "", 10)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*self.HELIOS_BLACK)
        self.multi_cell(0, 5, safe_latin1(text))
        self.ln(2)

    def key_value_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 6, safe_latin1(key), border=0)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, safe_latin1(str(value)), border=0, new_x="LMARGIN", new_y="NEXT")


def build_pdf_report() -> bytes:
    pdf = HeliosReport()
    pdf.add_page()

    pdf.body_text(f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

    # 1. Extraction
    extracted = st.session_state.get("extracted_data")
    if extracted:
        pdf.section_title("1. Extrahierte Daten")
        pdf.body_text(
            f"Dokumenttyp: {extracted.get('document_type', 'k.A.')} | "
            f"Konfidenz: {extracted.get('confidence', 'k.A.')}"
        )
        for product in extracted.get("products", []):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, safe_latin1(product.get("product_name", "Unbekannt")), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            for k, v in product.items():
                if k != "product_name" and v is not None:
                    pdf.key_value_row(k, str(v))
            pdf.ln(3)

    # 2. Search recommendations
    search = st.session_state.get("search_results")
    if search:
        pdf.section_title("2. Produktempfehlungen")
        for rec in search.get("recommendations", [])[:3]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(
                0, 7,
                safe_latin1(f"#{rec['rank']} {rec['product_name']} (Score: {rec['score']})"),
                new_x="LMARGIN", new_y="NEXT",
            )
            pdf.body_text(rec.get("reasoning_de", ""))

    # 3. NFC config
    nfc = st.session_state.get("nfc_config")
    if nfc:
        pdf.section_title("3. NFC-Konfiguration")
        pdf.set_font("Courier", "", 9)
        nfc_text = json.dumps(nfc, indent=2, ensure_ascii=False)
        pdf.multi_cell(0, 4, safe_latin1(nfc_text))
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(3)

    # 4. Energy estimation
    energy = st.session_state.get("energy_result")
    if energy:
        pdf.section_title("4. Energieeinsparung")
        pdf.body_text(
            f"Geschätzte Jahreseinsparung: {energy.get('kwh', 0):,.0f} kWh/a"
        )
        pdf.body_text(
            f"Entspricht ca. {energy.get('euro', 0):,.0f} EUR/a "
            f"(bei {energy.get('price_ct', 30)} ct/kWh)"
        )

    # 5. Methodology
    pdf.section_title("5. Methodik")
    pdf.body_text(
        "LLM: Claude Sonnet 4 (Anthropic) für strukturierte Extraktion und Ranking. "
        "Embeddings: sentence-transformers/all-MiniLM-L6-v2 (lokal). "
        "Vector Store: ChromaDB (in-memory). "
        "ML: RandomForest Regressor (scikit-learn) auf synthetischen Daten (n=60). "
        "PDF-Parsing: PyMuPDF."
    )
    pdf.body_text(
        "Hinweis: Dieser Report wurde von einem Prototyp generiert und dient "
        "der Konzeptdemonstration. Keine Planungsgrundlage."
    )

    output = pdf.output()
    if output is None:
        return b""
    return bytes(output)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        logo_path = BASE_DIR / "assets" / "helios_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=200)
        else:
            st.markdown("### **HELIOS**")

        st.markdown("### HeliosDocAI")
        st.markdown(
            "AI-gestützte Dokumentenanalyse für Helios Ventilatoren. "
            "Prototyp für Inhouse-Prozessoptimierung."
        )

        st.divider()
        st.markdown("**Features:**")
        st.markdown(
            "- PDF-Extraktion\n"
            "- Semantische Produktsuche\n"
            "- NFC-Konfiguration\n"
            "- Energieschätzung\n"
            "- Modell-Evaluation\n"
            "- PDF-Report"
        )

        st.divider()
        st.text_input(
            "Anthropic API-Key (optional)",
            type="password",
            key="anthropic_api_key",
            help="Wird nicht gespeichert. Für Live-AI-Funktionen.",
        )

        api_status = get_api_key()
        if api_status:
            st.success("API-Key vorhanden", icon="✅")
            call_count = st.session_state.get("api_call_count", 0)
            total_cost = st.session_state.get("api_total_cost_usd", 0.0)
            st.caption(
                f"Aufrufe: {call_count}/{MAX_API_CALLS_PER_SESSION} | "
                f"Kosten: ~${total_cost:.3f}"
            )
        else:
            st.info("Demo-Modus aktiv", icon="ℹ️")

        st.divider()
        st.markdown(
            "**Gebaut von Dominik Tsatskin**\n\n"
            "[Stellenausschreibung Job-ID 53926]"
            "(https://karriere.heliosventilatoren.de/helios/job/53926)"
        )


# ---------------------------------------------------------------------------
# Tab 1: PDF Extraction
# ---------------------------------------------------------------------------


def render_tab_extraction():
    st.header("PDF-Extraktion")
    st.markdown(
        "Laden Sie ein beliebiges PDF hoch (Helios-Datenblatt, Gebäudeplan, "
        "Kundenanfrage, Installationsprotokoll). Die App extrahiert automatisch "
        "strukturierte technische Daten."
    )

    uploaded = st.file_uploader("PDF hochladen", type=["pdf"], key="pdf_upload")

    if uploaded is not None:
        try:
            import fitz

            pdf_bytes = uploaded.read()
            if len(pdf_bytes) > MAX_PDF_SIZE_MB * 1024 * 1024:
                st.error(f"PDF zu groß (max. {MAX_PDF_SIZE_MB} MB).")
                return

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.page_count > MAX_PDF_PAGES:
                st.warning(
                    f"PDF hat {doc.page_count} Seiten. "
                    f"Nur die ersten {MAX_PDF_PAGES} werden verarbeitet."
                )

            pages = [doc[i] for i in range(min(doc.page_count, MAX_PDF_PAGES))]
            raw_text = "".join(str(page.get_text()) for page in pages)
            doc.close()

            if not raw_text.strip():
                st.warning("Kein Text im PDF gefunden. Möglicherweise ein Scan/Bild-PDF.")
                return

            with st.expander("Rohtext anzeigen", expanded=False):
                st.text_area("Extrahierter Text", raw_text, height=200, disabled=True)

            if len(raw_text) > MAX_EXTRACTION_CHARS:
                st.info(
                    f"Dokument hat {len(raw_text):,} Zeichen. "
                    f"Für die Extraktion werden die ersten {MAX_EXTRACTION_CHARS:,} verwendet."
                )

            api_key = get_api_key()
            if api_key:
                with st.spinner("Claude analysiert das Dokument..."):
                    system_prompt = read_prompt(PROMPT_EXTRACTION)
                    if not system_prompt:
                        st.warning("Extraktions-Prompt nicht verfügbar. Zeige Demo-Daten.")
                        data = load_json(DEMO_EXTRACTION)
                        if data:
                            st.session_state["extracted_data"] = data
                        return
                    result_text = call_claude(system_prompt, raw_text[:MAX_EXTRACTION_CHARS])

                if result_text:
                    try:
                        cleaned = result_text.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.split("\n", 1)[1]
                            cleaned = cleaned.rsplit("```", 1)[0]
                        data = json.loads(cleaned)
                        st.session_state["extracted_data"] = data
                        st.success(
                            f"Extraktion erfolgreich! Konfidenz: {data.get('confidence', 'k.A.')} | "
                            f"Dokumenttyp: {data.get('document_type', 'k.A.')}"
                        )
                    except json.JSONDecodeError:
                        st.warning("Claude-Antwort war kein valides JSON. Fallback auf Demo-Daten.")
                        _load_demo_extraction()
                else:
                    st.warning("API-Aufruf fehlgeschlagen. Zeige Demo-Daten.")
                    _load_demo_extraction()
            else:
                st.warning(
                    "⚠️ Demo-Modus: Zeigt vorbereitete Beispiel-Extraktion. "
                    "Für Live-Extraktion ANTHROPIC_API_KEY in Sidebar eingeben."
                )
                _load_demo_extraction()

        except Exception:
            st.error("PDF-Verarbeitung fehlgeschlagen.")
            _load_demo_extraction()

    else:
        if "extracted_data" not in st.session_state:
            st.info("Laden Sie ein PDF hoch oder nutzen Sie die Demo-Daten.")
            if st.button("Demo-Daten laden", key="load_demo_extraction"):
                _load_demo_extraction()

    # Display extracted data
    extracted = st.session_state.get("extracted_data")
    if extracted:
        st.subheader("Extrahierte Produkte")
        products = extracted.get("products", [])
        if products:
            df = pd.DataFrame(products)
            st.data_editor(
                df,
                use_container_width=True,
                num_rows="dynamic",
                key="extraction_editor",
            )

        if extracted.get("raw_requirements"):
            st.subheader("Erkannte Anforderungen")
            st.info(extracted["raw_requirements"])


def _load_demo_extraction():
    """Safely load demo extraction data into session state."""
    try:
        data = load_json(DEMO_EXTRACTION)
        if data:
            st.session_state["extracted_data"] = data
    except Exception:
        st.error("Demo-Daten konnten nicht geladen werden.")


# ---------------------------------------------------------------------------
# Tab 2: Semantic Search
# ---------------------------------------------------------------------------


def render_tab_search():
    st.header("Semantische Produktsuche")
    st.markdown(
        "Stellen Sie eine natürlichsprachliche Frage und die App durchsucht "
        "den Helios-Produktkatalog semantisch."
    )

    query = st.text_input(
        "Ihre Anfrage",
        placeholder="z.B. Welcher Ventilator eignet sich für einen 80m² Serverraum mit max. 35 dB?",
        key="search_query",
    )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        max_sound = st.slider("Max. Schallpegel (dB(A))", 20, 60, 50, key="filter_sound")
    with col_f2:
        min_airflow = st.slider("Min. Luftleistung (m³/h)", 0, 200, 0, key="filter_airflow")
    with col_f3:
        filter_nfc = st.checkbox("Nur NFC-fähig", key="filter_nfc")
        filter_wrg = st.checkbox("Nur mit WRG", key="filter_wrg")
        filter_ex = st.checkbox("Nur Ex-Schutz", key="filter_ex")

    if query:
        collection, embed_model = build_chroma_collection()

        if collection is not None and embed_model is not None:
            with st.spinner("Suche im Produktkatalog..."):
                query_embedding = embed_model.encode([query]).tolist()
                results = collection.query(
                    query_embeddings=query_embedding, n_results=10
                )

            catalog = load_product_catalog()
            if catalog is None:
                st.error("Produktkatalog nicht verfügbar.")
                return
            products_by_id = {p["id"]: p for p in catalog["products"]}

            filtered = _filter_search_results(
                results, products_by_id, max_sound, min_airflow,
                filter_nfc, filter_wrg, filter_ex
            )

            # Try Claude ranking
            api_key = get_api_key()
            if api_key and filtered:
                candidates_text = "\n".join(
                    f"- {r['product']['name']}: {r['product']['description']}"
                    for r in filtered
                )
                with st.spinner("Claude erstellt intelligentes Ranking..."):
                    system_prompt = read_prompt(PROMPT_MATCHING)
                    if system_prompt:
                        user_msg = (
                            f"Anforderung: {query}\n\n"
                            f"Filter: max {max_sound} dB(A), min {min_airflow} m³/h\n\n"
                            f"Kandidaten:\n{candidates_text}"
                        )
                        ranking_text = call_claude(system_prompt, user_msg)

                        if ranking_text:
                            try:
                                cleaned = ranking_text.strip()
                                if cleaned.startswith("```"):
                                    cleaned = cleaned.split("\n", 1)[1]
                                    cleaned = cleaned.rsplit("```", 1)[0]
                                ranking_data = json.loads(cleaned)
                                st.session_state["search_results"] = ranking_data
                                _display_search_results(ranking_data)
                                return
                            except json.JSONDecodeError:
                                pass

            # Fallback: show Chroma results directly
            if filtered:
                st.info(
                    "Für intelligentes Ranking API-Key eingeben. "
                    "Zeige ChromaDB-Similarity-Scores."
                )
                for r in filtered:
                    p = r["product"]
                    with st.container():
                        st.markdown(
                            f"**#{r['rank']} {p['name']}** "
                            f"(Similarity: {r['score']:.2f})"
                        )
                        st.markdown(f"*{p['description']}*")
                        st.markdown(f"Einsatz: {', '.join(p.get('use_cases', []))}")
                        st.divider()
            else:
                st.warning("Keine passenden Produkte gefunden. Passen Sie die Filter an.")
        else:
            st.warning("Vektorsuche nicht verfügbar. Zeige Demo-Ergebnisse.")
            _load_demo_search()

    elif "search_results" in st.session_state:
        _display_search_results(st.session_state["search_results"])

    else:
        if st.button("Demo-Suche laden", key="load_demo_search"):
            _load_demo_search()


def _filter_search_results(
    results, products_by_id, max_sound, min_airflow,
    filter_nfc, filter_wrg, filter_ex
) -> list:
    """Filter ChromaDB results by user criteria."""
    filtered = []
    if not results or not results["ids"]:
        return filtered

    for i, pid in enumerate(results["ids"][0]):
        product = products_by_id.get(pid)
        if not product:
            continue
        specs = product.get("specs", {})

        sound = specs.get("sound_pressure_dba")
        if isinstance(sound, dict) and sound.get("low") is not None:
            if sound["low"] > max_sound:
                continue

        airflow = specs.get("airflow_m3h")
        max_af = 0
        if isinstance(airflow, list):
            max_af = max(airflow)
        elif isinstance(airflow, (int, float)):
            max_af = airflow
        if max_af < min_airflow and max_af > 0:
            continue

        if filter_nfc and not specs.get("nfc_configurable"):
            continue
        if filter_wrg and not specs.get("wrg_efficiency_pct"):
            continue
        if filter_ex and not specs.get("ex_rating"):
            continue

        score = 1 - (results["distances"][0][i] if results["distances"] else 0)
        filtered.append(
            {"product": product, "score": round(score, 3), "rank": len(filtered) + 1}
        )
        if len(filtered) >= 5:
            break

    return filtered


def _load_demo_search():
    """Safely load demo search data."""
    try:
        demo = load_json(DEMO_SEARCH)
        if demo:
            st.session_state["search_results"] = demo
            _display_search_results(demo)
    except Exception:
        st.error("Demo-Suchdaten konnten nicht geladen werden.")


def _display_search_results(data: dict):
    """Display ranked search results as cards."""
    if data.get("general_note"):
        st.info(data["general_note"])

    for rec in data.get("recommendations", []):
        with st.container():
            cols = st.columns([1, 6])
            with cols[0]:
                score = rec.get("score", 0)
                st.metric(f"#{rec['rank']}", f"{score:.0%}")
            with cols[1]:
                st.markdown(f"**{rec['product_name']}**")
                st.markdown(rec.get("reasoning_de", ""))
                if rec.get("caveats"):
                    for c in rec["caveats"]:
                        st.caption(f"⚠ {c}")
                meets = rec.get("meets_all_requirements", False)
                if meets:
                    st.success("Erfüllt alle Anforderungen", icon="✅")
            st.divider()


# ---------------------------------------------------------------------------
# Tab 3: NFC Configuration
# ---------------------------------------------------------------------------


def render_tab_nfc():
    st.header("NFC-Konfigurations-Simulation")
    st.info(
        "ℹ️ Simulierte Konfiguration basierend auf öffentlich dokumentierten ELS NFC "
        "Parametern. Zeigt das Konzept einer automatisierten Parametrierung."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Parameter")

        model = st.selectbox("Modell", ELS_NFC_PARAMS["models"], key="nfc_model")

        st.markdown("**Volumenstrom-Stufen**")
        steps = ELS_NFC_PARAMS["airflow_steps_m3h"]
        defaults = ELS_NFC_PARAMS["factory_defaults_m3h"]

        stufe_1 = st.select_slider(
            "Stufe 1 (m³/h)", options=steps,
            value=defaults["stufe_1"], key="nfc_s1"
        )
        stufe_2 = st.select_slider(
            "Stufe 2 (m³/h)", options=steps,
            value=defaults["stufe_2"], key="nfc_s2"
        )
        stufe_3 = st.select_slider(
            "Stufe 3 (m³/h)", options=steps,
            value=defaults["stufe_3"], key="nfc_s3"
        )

        # Stage ordering validation
        if stufe_1 >= stufe_2 or stufe_2 >= stufe_3:
            st.warning("⚠ Stufen sollten aufsteigend sein: Stufe 1 < Stufe 2 < Stufe 3")

        use_s4 = st.checkbox("4. Stufe aktivieren", key="nfc_use_s4")
        stufe_4 = None
        if use_s4:
            stufe_4 = st.select_slider(
                "Stufe 4 (m³/h)", options=steps, value=80, key="nfc_s4"
            )

        use_s5 = st.checkbox("5. Stufe aktivieren", key="nfc_use_s5")
        stufe_5 = None
        if use_s5:
            stufe_5 = st.select_slider(
                "Stufe 5 (m³/h)", options=steps, value=100, key="nfc_s5"
            )

        grundlueftung = st.select_slider(
            "Grundlüftung (m³/h)", options=steps,
            value=ELS_NFC_PARAMS["basic_ventilation_m3h"], key="nfc_grund"
        )

        st.markdown("**Zeiten**")
        delay = st.slider("Einschaltverzögerung (s)", 0, 120, 5, key="nfc_delay")
        runon = st.slider("Nachlaufzeit (min)", 0, 90, 15, key="nfc_runon")
        interval = st.slider("Intervallzeit (h)", 0, 24, 2, key="nfc_interval")

        # Sensor parameters (conditional)
        sensor_config = None
        sensor_opts = ELS_NFC_PARAMS["sensor_options"]
        if model in sensor_opts:
            st.markdown("**Sensor-Parameter**")
            opt = sensor_opts[model]

            if opt["type"] == "humidity":
                sensor_config = {
                    "type": "Feuchte",
                    "unit": opt["unit"],
                    "schwellenwert_rh": st.slider(
                        "Feuchte-Schwellenwert (%rH)", 40, 90, 65, key="nfc_rh"
                    ),
                }
            elif opt["type"] == "presence":
                sensor_config = {
                    "type": "Präsenz",
                    "detection": opt["detection"],
                }
            elif opt["type"] == "voc":
                voc_mode = st.selectbox("VOC-Modus", opt["modes"], key="nfc_voc_mode")
                voc_thresh = st.slider(
                    "VOC-Schwellenwert",
                    opt["threshold_range"][0],
                    opt["threshold_range"][1],
                    250, key="nfc_voc_thresh",
                )
                voc_max = st.slider(
                    "VOC-Maximalwert",
                    opt["max_value_range"][0],
                    opt["max_value_range"][1],
                    400, key="nfc_voc_max",
                )
                sensor_config = {
                    "type": "VOC",
                    "mode": voc_mode,
                    "schwellenwert_voc": voc_thresh,
                    "maximalwert_voc": voc_max,
                    "max_volumenstrom_m3h": stufe_3,
                }
            elif opt["type"] == "co2":
                sensor_config = {
                    "type": "CO2",
                    "unit": opt["unit"],
                    "schwellenwert_ppm": st.slider(
                        "CO2-Schwellenwert (ppm)", 400, 2000, 1000, step=50, key="nfc_co2"
                    ),
                }

        # Buttons
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Werkseinstellung laden", key="nfc_reset"):
                for k in ["nfc_s1", "nfc_s2", "nfc_s3", "nfc_delay", "nfc_runon", "nfc_interval"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        with col_b2:
            if st.button("Aus Extraktion übernehmen", key="nfc_from_extract"):
                extracted = st.session_state.get("extracted_data")
                if extracted and extracted.get("products"):
                    product = extracted["products"][0]
                    name = product.get("product_name", "")
                    for m in ELS_NFC_PARAMS["models"]:
                        if m.lower() in name.lower():
                            st.session_state["nfc_model"] = m
                            break
                    airflow = product.get("airflow_m3h")
                    if isinstance(airflow, (int, float)) and airflow in steps:
                        st.session_state["nfc_s3"] = airflow
                    st.success(f"Daten aus Extraktion übernommen: {name}")
                    st.rerun()
                else:
                    st.warning("Keine extrahierten Daten vorhanden. Zuerst Tab 'Extraktion' nutzen.")

    # Build config JSON
    config = {
        "_meta": {
            "generator": "HeliosDocAI Prototype",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
            "disclaimer": "Simulierte Konfiguration – nicht für Produktionseinsatz",
        },
        "device": {
            "model": model,
            "article_number": ARTICLE_NUMBERS.get(model),
        },
        "airflow_config": {
            "stufe_1_m3h": stufe_1,
            "stufe_2_m3h": stufe_2,
            "stufe_3_m3h": stufe_3,
            "stufe_4_m3h": stufe_4,
            "stufe_5_m3h": stufe_5,
            "grundlueftung_m3h": grundlueftung,
            "intervall_m3h": grundlueftung,
        },
        "timing": {
            "einschaltverzoegerung_sec": delay,
            "nachlaufzeit_min": runon,
            "intervallzeit_h": interval,
        },
    }
    if sensor_config:
        config["sensor"] = sensor_config

    st.session_state["nfc_config"] = config

    with col_right:
        st.subheader("Generierte Konfiguration")
        config_json = json.dumps(config, indent=2, ensure_ascii=False)
        st.code(config_json, language="json")

        st.download_button(
            label="JSON kopieren / herunterladen",
            data=config_json,
            file_name=f"nfc_config_{model.replace(' ', '_')}.json",
            mime="application/json",
            key="nfc_download",
        )

        st.markdown("---")
        st.subheader("Parameter-Erklärung")
        sensor_label = (
            f"mit {sensor_opts[model]['type']}-Sensor"
            if model in sensor_opts
            else "Basis ohne Sensor"
        )
        st.markdown(
            f"- **Modell:** {model} – {sensor_label}\n"
            f"- **Stufe 1-3:** Standard-Leistungsstufen ({stufe_1}/{stufe_2}/{stufe_3} m³/h)\n"
            f"- **Grundlüftung:** {grundlueftung} m³/h Dauerbetrieb\n"
            f"- **Einschaltverzögerung:** {delay}s bis Ventilator anläuft\n"
            f"- **Nachlaufzeit:** {runon} min nach Abschalten des Triggers\n"
            f"- **Intervallzeit:** Alle {interval}h automatischer Lüftungsstoß"
        )


# ---------------------------------------------------------------------------
# Tab 4: Energy Estimation
# ---------------------------------------------------------------------------


def render_tab_energy():
    st.header("Energieeinspar-Schätzung (WRG)")
    st.warning(
        "⚠️ Synthetisches Modell auf Basis physikalischer Näherungsformeln – "
        "dient der Konzeptdemonstration, nicht als Planungsgrundlage. "
        "Für DIN-konforme Auslegungen nutzen Sie KWLeasyPlan."
    )

    model, feature_cols = train_energy_model()

    col1, col2 = st.columns(2)
    with col1:
        room_size = st.slider("Raumgröße (m²)", 20, 300, 80, key="energy_room")
        ceiling_height = st.slider(
            "Deckenhöhe (m)", 2.4, 4.0, 2.7, step=0.1, key="energy_ceil"
        )
        air_changes = st.slider(
            "Luftwechselrate (/h)", 0.5, 5.0, 2.0, step=0.5, key="energy_ach"
        )
        wrg_eff = st.slider("WRG-Wirkungsgrad (%)", 50, 95, 75, key="energy_wrg")
    with col2:
        hours = st.slider("Betriebsstunden/Tag", 6, 24, 12, key="energy_hours")
        delta_t = st.slider(
            "Temperaturdifferenz innen/außen (K)", 10, 30, 20, key="energy_dt"
        )
        heating_days = st.slider("Heiztage/Jahr", 180, 240, 210, key="energy_hd")
        price_ct = st.slider("Strompreis (ct/kWh)", 20, 50, 30, key="energy_price")

    # Predict
    X_input = pd.DataFrame(
        [
            {
                "room_size_m2": room_size,
                "air_changes_per_h": air_changes,
                "ceiling_height_m": ceiling_height,
                "wrg_efficiency_pct": wrg_eff,
                "hours_per_day": hours,
                "delta_t_k": delta_t,
                "heating_days": heating_days,
            }
        ]
    )
    prediction = model.predict(X_input)[0]

    tree_preds = np.array([t.predict(X_input)[0] for t in model.estimators_])
    pred_std = tree_preds.std()

    st.session_state["energy_result"] = {
        "kwh": prediction,
        "euro": prediction * price_ct / 100,
        "price_ct": price_ct,
    }

    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Jahreseinsparung", f"{prediction:,.0f} kWh")
    with col_m2:
        savings_eur = prediction * price_ct / 100
        st.metric("Kosteneinsparung", f"{savings_eur:,.0f} EUR/a")
    with col_m3:
        co2_kg = prediction * DE_CO2_EMISSION_FACTOR_KG_PER_KWH
        st.metric("CO2-Vermeidung", f"{co2_kg:,.0f} kg/a")

    # Monthly chart
    st.subheader("Monatliche Aufschlüsselung")

    month_names = [
        "Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
        "Jul", "Aug", "Sep", "Okt", "Nov", "Dez",
    ]
    month_weights = [0.15, 0.14, 0.12, 0.08, 0.03, 0.0, 0.0, 0.0, 0.03, 0.10, 0.14, 0.15]
    total_w = sum(month_weights)
    monthly_kwh = [prediction * w / total_w for w in month_weights]
    monthly_upper = [(prediction + pred_std) * w / total_w for w in month_weights]
    monthly_lower = [max(0, (prediction - pred_std) * w / total_w) for w in month_weights]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=month_names, y=monthly_kwh,
            name="Einsparung (kWh)", marker_color=HELIOS_RED,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_names, y=monthly_upper, mode="lines",
            line={"dash": "dash", "color": "rgba(226,0,26,0.3)"},
            name="Obere Grenze", showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=month_names, y=monthly_lower, mode="lines",
            line={"dash": "dash", "color": "rgba(226,0,26,0.3)"},
            name="Untere Grenze", fill="tonexty",
            fillcolor="rgba(226,0,26,0.08)", showlegend=True,
        )
    )
    fig.update_layout(
        xaxis_title="Monat", yaxis_title="Einsparung (kWh)",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with st.expander("Modell-Details (Feature Importance)"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame(
            {"Feature": feature_cols, "Importance": importances}
        ).sort_values("Importance", ascending=True)

        fig_fi = go.Figure(
            go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color=HELIOS_RED,
            )
        )
        fig_fi.update_layout(
            title="RandomForest Feature Importance",
            xaxis_title="Importance", template="plotly_white", height=300,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption(
            f"Modell: RandomForestRegressor (n_estimators=100, n_samples=60). "
            f"Prediction Std: ±{pred_std:,.0f} kWh"
        )


# ---------------------------------------------------------------------------
# Tab 5: Model Evaluation
# ---------------------------------------------------------------------------


def render_tab_evaluation():
    st.header("Modell-Evaluation")
    st.markdown(
        '*"Sie evaluieren unterschiedliche Modelle, von klassischen Ansätzen bis hin '
        "zu modernen Foundation-Modellen, vergleichen Architektur, Qualität und "
        'Machbarkeit."* – Stellenausschreibung'
    )

    try:
        comparison = load_json(DEMO_COMPARISON)
        if comparison is None:
            st.error("Vergleichsdaten konnten nicht geladen werden.")
            return
    except Exception:
        st.error("Vergleichsdaten konnten nicht geladen werden.")
        return

    # Input text
    with st.expander("Eingabetext (gleiches Dokument für beide Modelle)", expanded=False):
        st.text_area("Text", comparison["input_text"], height=150, disabled=True)

    # Comparison table
    st.subheader("Vergleich: Claude Sonnet vs. Llama-3.3-70B")

    claude_r = comparison["claude_result"]
    llama_r = comparison["llama_result"]

    comparison_data = {
        "Kriterium": [
            "Vollständigkeit (Felder extrahiert)",
            "JSON-Validität",
            "Latenz",
            "Kosten/1K Tokens",
            "Deutsch-Kompetenz",
            "Strukturtreue",
        ],
        "Claude Sonnet 4": [
            f"{claude_r['fields_extracted']}/{claude_r['fields_total']}",
            "✅ Valide" if claude_r["json_valid"] else "❌ Invalide",
            f"{claude_r['latency_ms']}ms",
            f"~${claude_r['cost_per_1k_tokens']}",
            claude_r["german_quality"],
            claude_r["structure_adherence"],
        ],
        "Llama-3.3-70B (Groq)": [
            f"{llama_r['fields_extracted']}/{llama_r['fields_total']}",
            ("⚠️ Nachkorrektur nötig" if llama_r["json_needed_correction"] else "✅ Valide"),
            f"{llama_r['latency_ms']}ms",
            f"~${llama_r['cost_per_1k_tokens']}",
            llama_r["german_quality"],
            llama_r["structure_adherence"],
        ],
    }

    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)

    if llama_r.get("issues"):
        with st.expander("Llama-3.3-70B: Erkannte Probleme"):
            for issue in llama_r["issues"]:
                st.markdown(f"- {issue}")

    # Side-by-side JSON
    st.subheader("Extraktions-Ergebnisse im Detail")
    col_c, col_l = st.columns(2)
    with col_c:
        st.markdown("**Claude Sonnet 4**")
        st.json(claude_r["extraction"])
    with col_l:
        st.markdown("**Llama-3.3-70B**")
        st.json(llama_r["extraction"])

    # Summary
    st.subheader("Fazit")
    summary = comparison["comparison_summary"]
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Beste Genauigkeit", summary["winner_accuracy"])
    with col_s2:
        st.metric("Schnellstes Modell", summary["winner_speed"])
    with col_s3:
        st.metric("Günstigstes Modell", summary["winner_cost"])

    st.info(summary["recommendation"])
    st.caption(summary["key_insight"])


# ---------------------------------------------------------------------------
# Tab 6: PDF Report
# ---------------------------------------------------------------------------


def render_tab_report():
    st.header("PDF-Report Export")
    st.markdown(
        "Generiert einen strukturierten PDF-Report mit allen Ergebnissen "
        "aus den anderen Tabs."
    )

    st.subheader("Verfügbare Daten")
    data_status = {
        "Extraktion": "extracted_data" in st.session_state,
        "Produktsuche": "search_results" in st.session_state,
        "NFC-Konfiguration": "nfc_config" in st.session_state,
        "Energieschätzung": "energy_result" in st.session_state,
    }

    for name, available in data_status.items():
        if available:
            st.success(f"{name}: Daten vorhanden", icon="✅")
        else:
            st.warning(f"{name}: Keine Daten – Tab zuerst nutzen", icon="⚠️")

    any_data = any(data_status.values())

    if any_data:
        try:
            pdf_bytes = build_pdf_report()
            st.download_button(
                label="📥 PDF-Report herunterladen",
                data=pdf_bytes,
                file_name=f"heliosdocai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="download_report",
            )
        except Exception:
            st.error("Report-Generierung fehlgeschlagen.")
    else:
        st.info(
            "Nutzen Sie zuerst die anderen Tabs, um Daten zu generieren. "
            "Der Report fasst alle verfügbaren Ergebnisse zusammen."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logo_path = BASE_DIR / "assets" / "helios_logo.png"
    st.set_page_config(
        page_title="HeliosDocAI",
        page_icon=str(logo_path) if logo_path.exists() else "🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.markdown(
        "<h1 style='text-align: center;'>HeliosDocAI</h1>"
        "<p style='text-align: center; font-size: 1.1em;'>"
        "<strong>AI-gestützte Dokumentenanalyse für Helios Ventilatoren</strong><br>"
        "<em>Prototyp für Inhouse-Prozessoptimierung – "
        "dokumentenbezogene Workflows & unstrukturierte Daten</em></p>",
        unsafe_allow_html=True,
    )

    with st.expander("Was ist HeliosDocAI? – Projekt-Info & Anleitung zum Ausprobieren", expanded=True):
        st.markdown(
            "**HeliosDocAI** ist ein funktionsfähiger AI-Prototyp, der zeigt, wie "
            "intelligente Dokumentenverarbeitung die Inhouse-Workflows bei Helios "
            "Ventilatoren optimieren kann.\n\n"
            "**Das Problem:** Technische Daten liegen in unstrukturierten Formaten vor "
            "(PDFs, Datenblätter, Kundenanfragen) und müssen heute manuell ausgewertet werden.\n\n"
            "**Die Lösung:** HeliosDocAI extrahiert automatisch technische Parameter aus "
            "beliebigen Dokumenten, findet passende Produkte per natürlichsprachlicher Suche, "
            "generiert NFC-Konfigurationen und schätzt Energieeinsparungen – alles AI-gestützt.\n\n"
            "**Abgrenzung:**\n"
            "- **HeliosSelect** = Regelbasierter Konfigurator (der Nutzer weiß, was er sucht)\n"
            "- **KWLeasyPlan** = DIN-konforme Lüftungsplanung (strukturierte Eingabe)\n"
            "- **HeliosDocAI** = AI-gestützt: unstrukturierte Inputs → strukturierte Outputs\n\n"
            "HeliosDocAI ersetzt keine bestehenden Tools, sondern schließt die Lücke dazwischen."
        )
        st.markdown("---")
        st.markdown("### So können Sie die App ausprobieren")
        st.markdown(
            "Die App funktioniert sofort – **kein Setup, kein API-Key nötig.** "
            "Klicken Sie einfach durch die Tabs:\n\n"
            "**1. Extraktion** – Klicken Sie auf *\"Demo-Daten laden\"*. "
            "Sie sehen, wie ein PDF-Datenblatt automatisch in eine strukturierte Tabelle "
            "umgewandelt wird (Produktname, Luftleistung, Schallpegel, Artikelnummer, …).\n\n"
            "**2. Produktsuche** – Klicken Sie auf *\"Demo-Suche laden\"* oder tippen Sie "
            "eine Anfrage wie *\"Leiser Ventilator für 25m² Büro mit Luftqualitätssensor\"*. "
            "Die App durchsucht den Helios-Produktkatalog semantisch und liefert ein "
            "begründetes Ranking.\n\n"
            "**3. NFC-Konfiguration** – Wählen Sie ein ELS NFC-Modell und bewegen Sie "
            "die Slider. Rechts wird live eine JSON-Konfiguration generiert, wie sie "
            "per NFC auf das Gerät übertragen werden könnte.\n\n"
            "**4. Energieschätzung** – Stellen Sie Raumparameter ein (Größe, WRG-Wirkungsgrad, "
            "Temperaturdifferenz). Das ML-Modell berechnet die geschätzte Jahreseinsparung "
            "in kWh, EUR und CO₂.\n\n"
            "**5. Modell-Evaluation** – Vergleich: Claude Sonnet vs. Llama-3.3-70B. "
            "Gleiches Dokument, gleiches Ziel – welches Modell extrahiert besser?\n\n"
            "**6. PDF-Report** – Fasst alle bisherigen Ergebnisse in einem "
            "herunterladbaren PDF zusammen."
        )
        st.markdown(
            "| Tab | Feature | Technologie |\n"
            "|---|---|---|\n"
            "| Extraktion | PDF → strukturierte Daten | Claude Sonnet 4.6 + PyMuPDF |\n"
            "| Produktsuche | Natürlichsprachliches Retrieval | sentence-transformers + ChromaDB |\n"
            "| NFC-Konfiguration | Automatisierte Parametrierung | Parametrisches Modell |\n"
            "| Energieschätzung | ML-basierte Prognose | scikit-learn RandomForest |\n"
            "| Modell-Evaluation | Systematischer LLM-Vergleich | Claude vs. Llama |"
        )

    tabs = st.tabs(
        [
            "📄 Extraktion",
            "🔍 Produktsuche",
            "📱 NFC-Konfiguration",
            "⚡ Energieschätzung",
            "🔬 Modell-Evaluation",
            "📥 PDF-Report",
        ]
    )

    with tabs[0]:
        render_tab_extraction()
    with tabs[1]:
        render_tab_search()
    with tabs[2]:
        render_tab_nfc()
    with tabs[3]:
        render_tab_energy()
    with tabs[4]:
        render_tab_evaluation()
    with tabs[5]:
        render_tab_report()


if __name__ == "__main__":
    main()
