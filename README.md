# AI Lead & Brand Researcher Pro (Gemini + LangGraph)

An enterprise-grade B2B Lead Generation and Deep Market Intelligence scraper engine. This application orchestrates autonomous search workflows using **LangGraph**, processes unstructured web fragments via **Google Gemini 1.5 Flash**, and outputs validation-secured data schemas directly into production-ready Excel records via a streamlined **Streamlit** dashboard.

## ⚡ Engineering Architecture Highlights

* **Pydantic Structural Enforcement:** Bypasses legacy Regex and standard raw JSON text parsing methods entirely. The system binds the data model explicitly to the Google Gemini Function Calling API via `with_structured_output()`, ensuring a **0% parsing failure rate**.
* **State Machine Pattern:** Uses an iterative `StateGraph` compilation mechanism via LangGraph to cycle cleanly through entity batch pools without state leakage.
* **Dual Search Mode Topology:** Native compatibility with open-access `DuckDuckGo Engines` and enterprise API search topologies like `Tavily Search API`.
* **Proactive Rate-Limit Management:** Integrates standard step-delays (4-second windows) to maintain operational stability under standard public API Tier limits (15 Requests Per Minute throttling benchmarks).

---

## 🛠️ Local Environment Deployment

Follow these sequential steps to establish and deploy the pipeline locally:

### Step 1: Initialize Project Directory
Create a dedicated workspace folder and save the application pipeline engine script as `app.py`.

### Step 2: System Package Installation
Open your terminal window or system Command Prompt, navigate directly to your active root folder, and execute the package installation:
```bash
pip install -r requirements.txt
