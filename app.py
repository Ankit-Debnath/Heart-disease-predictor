from flask import Flask, render_template, request, jsonify, redirect, session, flash 
import pandas as pd
import joblib
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Required secret key for tracking flashes and sessions safely
app.secret_key = "super_secure_random_heart_key" 

# 🚀 AUTOMATIC DATABASE INITIALIZATION ON RENDER
# This runs once before the very first request hits the server
@app.before_request
def initialize_database():
    if not os.path.exists('users.db'):
        print("Database not found. Initializing users.db dynamically...")
        try:
            import database
            # If your database script uses a function wrapper like init_db, uncomment below:
            # database.init_db() 
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")

@app.route("/dashboard")
def dashboard():
    # Force access restriction unless user is authenticated
    if "user" not in session:
        return redirect("/")
    return render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["user"] = username  # Save username session state
            return redirect("/dashboard")
        else:
            flash("Invalid Username or Password") 
            return redirect("/")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form["fullname"]
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm_password"]

        if password != confirm:
            flash("Passwords do not match")
            return redirect("/signup")

        hashed_password = generate_password_hash(password)
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        try:
            cursor.execute("""
            INSERT INTO users(fullname, username, password)
            VALUES(?,?,?)
            """, (fullname, username, hashed_password))
            conn.commit()
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Username already exists")
            return redirect("/signup")
        finally:
            conn.close()

    return render_template("signup.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        patient = pd.DataFrame({
            "age": [data["age"]],
            "sex": [data["sex"]],
            "cp": [data["cp"]],
            "trestbps": [data["trestbps"]],
            "chol": [data["chol"]],
            "fbs": [data["fbs"]],
            "restecg": [data["restecg"]],
            "thalach": [data["thalach"]],
            "exang": [data["exang"]],
            "oldpeak": [data["oldpeak"]],
            "slope": [data["slope"]],
            "ca": [data["ca"]],
            "thal": [data["thal"]]
        })

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0]
        heart_probability = round(probability[1] * 100, 2)
        no_heart_probability = round(probability[0] * 100, 2)

        if prediction == 1:
            return jsonify({
                "prediction": 1,
                "probability": heart_probability,
                "message": f"There is a {heart_probability}% probability of heart disease based on clinical indicators."
            })
        else:
            return jsonify({
                "prediction": 0,
                "probability": no_heart_probability,
                "message": f"There is a {no_heart_probability}% probability of clear health indicators."
            })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

@app.route("/delete_account", methods=["POST"])
def delete_account():
    if "user" not in session:
        return redirect("/")
        
    username_to_delete = session["user"]
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM users WHERE username=?", (username_to_delete,))
        conn.commit()
        session.clear() 
        flash("Your account has been permanently deleted.")
    except Exception as e:
        flash("An error occurred during account deletion.")
    finally:
        conn.close()
        
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)