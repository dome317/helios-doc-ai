# 🌀 HeliosDocAI – AI-Prototyp für Inhouse-Dokumentenworkflows

> **Bewerbung für:** Junior AI Developer & Consultant (m/w/d) | Helios Ventilatoren GmbH + Co KG | Job-ID 53926

🔗 **[Live Demo](URL)** | 🎥 **[Video-Walkthrough – 60 Sekunden](URL)** | 📄 **[Stellenausschreibung](https://karriere.heliosventilatoren.de/helios/job/53926)**

---

## Was ist HeliosDocAI?

Ein funktionsfähiger AI-Prototyp, der zeigt, wie AI-gestützte Dokumentenverarbeitung
die Inhouse-Workflows bei Helios Ventilatoren optimieren kann.

**Abgrenzung zum bestehenden Helios-Ökosystem:**
- **HeliosSelect** = Regelbasierter Produktkonfigurator (manuell, der Nutzer weiß was er sucht)
- **KWLeasyPlan** = DIN-konforme Lüftungsplanung (strukturierte Eingabe)
- **HeliosDocAI** = AI-gestützt: Unstrukturierte Inputs (PDFs, Mails, Pläne) → strukturierte Outputs

HeliosDocAI ersetzt keine bestehenden Tools, sondern schließt die Lücke zwischen
unstrukturierten Dokumenten und den vorhandenen Planungstools.

---

## Stellenanforderungen → Prototyp-Umsetzung

| Anforderung | Feature | Tab |
|---|---|---|
| „dokumentenbezogene Workflows und Verarbeitung unstrukturierter Daten" | PDF → strukturierte JSON-Extraktion | 📄 Extraktion |
| „Prototypen eigenständig umsetzen" | Komplette App in <24h konzipiert und deployed | Gesamt |
| „unterschiedliche Modelle evaluieren, Architektur, Qualität und Machbarkeit vergleichen" | Systematischer Claude vs. Llama Vergleich | 🔬 Evaluation |
| „aus Rohdaten Modellierungsansätze ableiten" | Energieeinspar-Regression auf synthetischen Daten | ⚡ Energie |
| „eigene Ideen für neue Use Cases" | NFC-Config-Simulation, semantische Produktsuche | 📱 NFC, 🔍 Suche |
| „Python + PyTorch/transformers/scikit-learn" | sentence-transformers, scikit-learn, Anthropic SDK | Gesamt |
| „Fast-Prototyping" | 24h von Idee bis Live-Deploy | Gesamt |
| „Entscheidungen klar dokumentieren" | ARCHITECTURE.md, Code-Kommentare, Methodenwahl | Doku |

---

## Architektur

Siehe [ARCHITECTURE.md](ARCHITECTURE.md) für Systemdiagramm und Modellauswahl-Begründungen.

## Tech Stack

| Komponente | Technologie | Begründung |
|---|---|---|
| Frontend | Streamlit | Schnellstes Python-UI-Framework für Prototyping |
| LLM (Primary) | Claude Sonnet via Anthropic API | Beste JSON-Zuverlässigkeit + Deutsch-Kompetenz 2026 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Lokal, kostenlos, kein Vendor-Lock |
| Vector Store | ChromaDB (in-memory) | Embedded, kein Server nötig |
| ML | scikit-learn RandomForestRegressor | Interpretierbar, robust bei kleiner Datenmenge |
| PDF Parsing | PyMuPDF | Schnell, zuverlässig, Open Source |
| Report | fpdf2 | Lightweight PDF-Generierung in Python |
| Deployment | Streamlit Community Cloud | Kostenlos, 1-Click aus GitHub |

## Schnellstart

```bash
git clone https://github.com/[USER]/helios-doc-ai.git
cd helios-doc-ai
pip install -r requirements.txt
cp .env.example .env  # API-Keys eintragen (optional – Demo-Modus funktioniert ohne)
streamlit run app.py
```

## Demo-Modus

Die App funktioniert **auch ohne API-Keys** mit vorbereiteten Beispieldaten.
Für Live-AI-Funktionen: Anthropic API-Key in der Sidebar eingeben.

---

## Anleitung zum Ausprobieren

### Ohne API-Key (Demo-Modus)

Alle Tabs funktionieren sofort mit vorbereiteten Beispieldaten:

1. **Extraktion** — Klick auf "Demo-Daten laden" zeigt eine Beispiel-Extraktion aus einem ELS NFC VOC Datenblatt
2. **Produktsuche** — Klick auf "Demo-Suche laden" zeigt ein semantisches Ranking (5 Produkte mit Scores)
3. **NFC-Konfiguration** — Slider bewegen, JSON-Output aktualisiert sich live
4. **Energieschätzung** — Raumparameter einstellen, Einsparung + CO2-Vermeidung wird berechnet
5. **Modell-Evaluation** — Vergleich Claude Sonnet vs. Llama-3.3-70B (Genauigkeit, Geschwindigkeit, Kosten)
6. **PDF-Report** — Fasst alle Tab-Ergebnisse in einem herunterladbaren PDF zusammen

### Mit API-Key (Live-AI)

API-Key in der Sidebar eingeben. Kostenkontrolle: max. 20 Aufrufe/Session, Kosten-Tracker in der Sidebar.

#### Test-Szenario 1: PDF-Extraktion (Tab "Extraktion")

Ein beliebiges Helios-Datenblatt als PDF hochladen (z.B. von [heliosventilatoren.de](https://www.heliosventilatoren.de)).
Die AI extrahiert automatisch: Produktname, Luftleistung, Schallpegel, Schutzart, Artikelnummer, etc.

**Erwartetes Ergebnis:** Strukturierte JSON-Tabelle mit allen erkannten technischen Daten.

#### Test-Szenario 2: Semantische Produktsuche (Tab "Produktsuche")

Natürlichsprachliche Anfragen eingeben, z.B.:

| Anfrage | Erwartetes Top-Ergebnis |
|---|---|
| "Leiser Ventilator für 25m2 Buero mit Luftqualitaetssensor" | ELS NFC VOC |
| "Feuchtegesteuerter Luefter fuer Badezimmer" | ELS NFC F |
| "Ventilator fuer Tiefgarage mit niedriger Decke" | IVRW EC 225 |
| "Explosionsgeschuetzter Ventilator fuer Lackiererei" | Explosionsgeschuetzter Axialventilator |
| "Waermerueckgewinnung fuer kleine Wohnung" | KWL EC 70 |

**Erwartetes Ergebnis:** Top-5 Ranking mit Scores, Begründung auf Deutsch, Einschränkungen.

#### Test-Szenario 3: NFC-Konfiguration (Tab "NFC-Konfiguration")

1. Modell "ELS NFC VOC" wählen
2. Stufen anpassen: Stufe 1 = 20 m3/h, Stufe 2 = 50 m3/h, Stufe 3 = 80 m3/h
3. VOC-Schwellenwert auf 300 setzen

**Erwartetes Ergebnis:** Valider JSON-Config mit Geräte-Info, Stufen, Timing und Sensor-Parametern.

#### Test-Szenario 4: Energieschätzung (Tab "Energieschätzung")

| Parameter | Wert |
|---|---|
| Raumgröße | 120 m2 |
| Deckenhöhe | 3.0 m |
| Luftwechselrate | 3.0 /h |
| WRG-Wirkungsgrad | 85% |

**Erwartetes Ergebnis:** Jahreseinsparung im Bereich 5.000-15.000 kWh, CO2-Vermeidung, monatliche Aufschlüsselung.

### Qualitätssicherung

99 automatisierte Tests abdeckend:
- Unit Tests (Kernfunktionen)
- Integrationstests (Datenkonsistenz, Plausibilität)
- Semantische Suchtests (ChromaDB liefert korrekte Top-Ergebnisse)
- Energiemodell-Plausibilitätsprüfungen

```bash
pytest test_app.py test_integration.py -v
```

---

**Gebaut von Dominik Tsatskin**
