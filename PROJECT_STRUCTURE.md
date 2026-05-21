# GMS Mission Control Project Structure

This repo now keeps the Flask web application split into backend logic, frontend templates/assets, and configuration.

```text
gms_project/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_STRUCTURE.md
│
├── backend/
│   ├── __init__.py
│   ├── gms_engine.py
│   ├── optimizer.py
│   └── routes.py
│
├── frontend/
│   ├── __init__.py
│   ├── templates/
│   │   ├── __init__.py
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── optimize.html
│   │   ├── map.html
│   │   ├── analysis.html
│   │   ├── alerts.html
│   │   └── about.html
│   │
│   └── static/
│       ├── __init__.py
│       ├── css/
│       │   ├── main.css
│       │   ├── dashboard.css
│       │   └── analysis.css
│       └── js/
│           ├── realtime.js
│           ├── main.js
│           ├── dashboard.js
│           ├── optimize.js
│           ├── map.js
│           └── analysis.js
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── core/
├── data/
├── evaluation/
├── tests/
├── utils/
└── visualization/
```

## Entry Points

- `python app.py` starts the web dashboard at `http://localhost:5000`.
- `python main.py` runs the CLI/research pipeline.
- `python tests/test_gms.py` runs the direct unit test script.

## Web App Flow

1. `app.py` creates the Flask app with `frontend/templates` and `frontend/static`.
2. `backend.gms_engine.GMSEngine` owns simulation, scoring, playback, and SSE state.
3. `backend.routes.register_routes()` wires pages, API endpoints, export, and streaming.
4. Templates are rendered from `frontend/templates`.
5. Shared styling and SSE navbar updates live in `frontend/static/css/main.css` and `frontend/static/js/realtime.js`.

## Configuration

- `config/settings.py` contains the 40-node web application settings.
- `config/__init__.py` keeps legacy `import config` compatibility for the CLI/core modules.

## Notes

The extracted page templates still contain some page-specific CSS and JavaScript inline to preserve the original working UI. Shared CSS and real-time update logic have been moved into static assets, and the documented static files are present for future page-by-page cleanup.
