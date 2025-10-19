# VolleySense

VolleySense is a modular volleyball analytics toolkit with a Gradio UI. It ingests video sessions, orchestrates plugin-based analysis, and maintains structured stats in SQLite.

## Setup

```
python tools/install.py
```

## Run

```
python -m app.main
```

Use `--share` for remote debugging and `--auth user pass` to enable simple authentication.

## Tests

```
pytest -q
```
