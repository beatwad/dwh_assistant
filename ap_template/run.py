import os
os.environ["SERVICE_START"] = "False"

from app import create_app
from dotenv import load_dotenv


def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)


if __name__ == "__main__":
    load_dotenv()
    os.environ["SERVICE_START"] = "True"
    app = create_app()
    app.run(debug=True)
