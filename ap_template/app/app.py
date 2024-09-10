import os
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from services.llm import natural_language_to_sql, prune_dialogue
from services.database import execute_sql_query, load_dbml_schema

from config.load_config import load_config

config_dict = load_config()

assistant_app = Flask(__name__)

assistant_app.config['SECRET_KEY'] = os.getenv("SQLAL_SECRET_KEY")

db_path = os.path.join(os.getcwd(), 'data', 'instance', 'users.db')
assistant_app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

db = SQLAlchemy(assistant_app)
login_manager = LoginManager(assistant_app)
login_manager.login_view = 'login'

dialogue = {}
dataframes = {}

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except ValueError as e:
        print(f"[ERROR] Catch a ValueError: {e}")
        return User.query.get(0)

# Registration form
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

# Login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@assistant_app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@assistant_app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form=form)

@assistant_app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@assistant_app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        return handle_post_request()
    return render_template("index.html", user=current_user)

@assistant_app.route("/download", methods=["GET"])
@login_required
def download_dataframe():
    user_id = current_user.id
    print(dataframes)
    if user_id in dataframes:
        df = dataframes[user_id]
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='dataframe.csv')
    # Return a JSON response if no dataframe is available
    response = jsonify({'error': 'No dataframe available to download.'})
    response.status_code = 404  # Set the status code to 404 or any other appropriate code
    return response

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
        # add previous dialogue to prompt
        query_history =  "User's request: " + user_query + "\n" + dialogue.get(current_user.id, "")
        return process_natural_language_query(user_query, query_history)
    return jsonify({"error": "No user query provided"}), 400

def process_query(query):
    try:
        ch_answer = execute_sql_query(query)
        if ch_answer["error"]:
            return jsonify({"error": str(ch_answer["error"])})
        df = ch_answer["result"].head(10)
        # save the last query result to dataframes storage
        dataframes[current_user.id] = df
        return jsonify(
            {"result": df.to_html(classes="table table-striped"), "sql": query}
        )
    except Exception as e:
        return jsonify({"error": f"There was an error during query execution: {str(e)}"})

def process_natural_language_query(user_query: str, query_history: str):
    dbml_schema = load_dbml_schema()
    llm_answer = natural_language_to_sql(user_query, query_history, dbml_schema)
    
    # store dialog history for the current user
    if llm_answer["status"] == "success":
        dialogue[current_user.id] = query_history + llm_answer["raw_response"] + "\n"
        # prune dialogue by deleting the very first conversations, so the overall
        # prompt can fit in LLM's context window
        dialogue[current_user.id] = prune_dialogue(dialogue[current_user.id])
        return process_query(llm_answer["sql"])
    return jsonify(
        {
            "error": llm_answer["error_description"],
            "sql": llm_answer.get("sql", ""),
            "rawResponse": llm_answer.get("raw_response", "No raw response"),
        }
    )

if __name__ == "__main__":
    with assistant_app.app_context():
        db.create_all()
    assistant_app.run(debug=True)
