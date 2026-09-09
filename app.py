"""
app.py — MarineEye Flask application entry point.

Creates the Flask app, registers configuration, and mounts the
route blueprint from routes.py.
"""

from flask import Flask
import os


def create_app():
    """Application factory."""
    app = Flask(__name__)

    # Secret key for flash messages / sessions
    app.secret_key = os.environ.get("SECRET_KEY", "marineeye-dev-secret-key-change-in-prod")

    # Paths
    app.config["WEIGHTS_PATH"] = os.path.join("weights", "best.pt")
    app.config["UPLOAD_DIR"] = os.path.join("static", "uploads")
    app.config["RESULT_DIR"] = os.path.join("static", "results")
    app.config["HISTORY_FILE"] = os.path.join("static", "history.json")

    # Ensure directories exist
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["RESULT_DIR"], exist_ok=True)

    # Register blueprint
    from routes import main
    app.register_blueprint(main)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)