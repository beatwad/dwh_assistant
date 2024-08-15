from flask import Flask, jsonify, render_template, request
from services.database import execute_sql_query, load_dbml_schema
# from services.database import load_table_names, build_dbml_schema
from services.llm import natural_language_to_sql

from dotenv import load_dotenv

load_dotenv()

assistant_app = Flask(__name__)


@assistant_app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return handle_post_request()
    return render_template("index.html")


def handle_post_request():
    if (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        and request.headers.get("Content-Type") == "application/json"
    ):
        return handle_ajax_request()
    return handle_form_request()


def handle_ajax_request():
    data = request.get_json()
    manual_query = data.get("manualQuery")
    if manual_query:
        print("*", manual_query)
        return process_query(manual_query)
    return jsonify({"error": "No manual query provided"}), 400


def handle_form_request():
    user_query = request.form.get("user_query")
    if user_query:
        return process_natural_language_query(user_query)
    return jsonify({"error": "No user query provided"}), 400


def process_query(query):
    try:
        ch_answer = execute_sql_query(query)
        if ch_answer["error"]:
            return jsonify({"error": str(ch_answer["error"])})
        df = ch_answer["result"].head(10)
        return jsonify(
            {"result": df.to_html(classes="table table-striped"), "sql": query}
        )
    except Exception as e:
        return jsonify({"error": f"Произошла ошибка при выполнении запроса: {str(e)}"})


def process_natural_language_query(user_query):

    # table_names = load_table_names()
    # dbml_schema = build_dbml_schema(table_names)
    
    dbml_schema = load_dbml_schema()
    
    llm_answer = natural_language_to_sql(user_query, dbml_schema)

    if llm_answer["status"] == "success":
        return process_query(llm_answer["sql"])
    return jsonify(
        {
            "error": llm_answer["error_description"],
            "sql": llm_answer.get("sql", ""),
            "rawResponse": llm_answer.get("raw_response", "Нет сырого ответа"),
        }
    )
