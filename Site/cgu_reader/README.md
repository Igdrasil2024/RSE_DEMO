# CGU Reader

Simple Flask web app that:
- takes a website URL,
- tries to find its Terms/CGU + Privacy/Cookies pages,
- sends extracted policy text to DeepSeek,
- returns a risk score (`0-100`) and the model output.

## Setup

1. Create and activate a virtual env.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (faster HTML parsing):

```bash
pip install lxml
```

3. Configure env vars:

```bash
cp .env.example .env
# then edit .env and set DEEPSEEK_API_KEY
```

Optional mode:
- `SIMULATE_SCRAPING=true` (default) to skip live scraping and generate plausible demo input.
- `SIMULATE_SCRAPING=false` to try live scraping.

## Run

```bash
python app.py
```

Open: `http://localhost:5000`

## Notes

- `0` means very bad terms for users.
- `100` means very user-friendly terms.
- Results are cached by website domain in `.cache/cgu_cache.json`.
- Re-analyzing the same site reuses cached output and skips DeepSeek API calls.
- Response includes:
  - detected main policy URL,
  - list of analyzed policy URLs,
  - score,
  - summary,
  - risks/highlights,
  - prompt sent to DeepSeek,
  - raw model response,
  - `simulated` flag,
  - `cached` flag (`true` when served from memory).
