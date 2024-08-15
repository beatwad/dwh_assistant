import os

from dotenv import load_dotenv

from .app import assistant_app as app


def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)


def create_app():
    load_env()
    return app
