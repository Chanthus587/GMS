"""Entry point for the GMS Mission Control Flask application."""

from flask import Flask

from backend.gms_engine import GMSEngine
from backend.routes import register_routes
from config.settings import DEBUG, HOST, PORT, THREADED


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static",
    )
    app.config["DEBUG"] = DEBUG
    app.config["THREADED"] = THREADED

    engine = GMSEngine()
    app.config["GMS_ENGINE"] = engine
    register_routes(app, engine)
    return app


app = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  GMS Mission Control")
    print(f"  http://localhost:{PORT}")
    print("  Ctrl+C to stop")
    print("=" * 55 + "\n")
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=THREADED)
