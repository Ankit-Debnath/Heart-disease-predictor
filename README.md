# ❤️ Heart Disease Prediction System

A streamlined, production-ready full-stack machine learning application that screens for cardiovascular disease markers using patient telemetry. This decoupled system links an optimized data pipeline, an authenticated Flask api runtime, and a fast, non-blocking asynchronous user dashboard.

## 🏗️ System Architecture & Workflow

  [🌐 Client UI Dashboard]  ──(Async JSON Payload via Fetch API)──► [⚡ Flask API Backend]
                                                                        │
        ┌───────────────────────────────────────────────────────────────┤
        ▼                                                               ▼
  [🗃️ SQLite Database] (users.db)                                 [🤖 ML Inference Pipeline]
  🔒 PBKDF2-SHA256 Password Salting                             📦 joblib (StandardScaler + SVM)
  
Frontend: Captured user telemetry is serialized into non-blocking JSON strings, avoiding full browser reloads.
Backend API: Validates incoming payloads, ensures route security via active session tracking, and handles relational auth states.
ML Engine: Features are scaled on-the-fly and evaluated against a pre-trained serialized model matrix.

🛠️ Tech Stack & Dependencies
Core Runtime: Python 3.11, Vanilla JavaScript (ES6+), HTML5, CSS3 Grid/Flexbox.
Machine Learning: Scikit-Learn 1.9.0, Pandas 3.0.3, Joblib 1.5.3.
Web Server: Flask 3.1.3, Werkzeug 3.1.8.
Database Layer: SQLite3 (native relational engine).

📊 Pipeline Optimization & Diagnostics
1. Data Integrity & Preprocessing
   Deduplication: Initial processing checked for redundancies via df.duplicated().sum(). The framework dropped 723 clone entries with df.drop_duplicates(), settling on 302 unique records to enforce cross-validation independence.
   Leakage Prevention: Continuous features (age, trestbps, chol, oldpeak, thalach) are nested with classification parameters inside a strict Scikit-Learn Pipeline construct. Values are scaled on training folds using StandardScaler without exposing parameters to unseen data boundaries.
   
2. Deployment Verdict: The SVM model was selected for production output (heart_disease_model.pkl) due to its superior stability and minimal risk profile (identifying 31 true positive disease vectors and missing only 2 across test samples).

🔒 Security Implementations
Cryptographic Credentials: Plaintext passwords pass through generate_password_hash() implementing modern PBKDF2-SHA256 salting algorithms before database insertion. Matches are handled symmetrically via check_password_hash().
State Lifecycle Guard: Internal prediction routes monitor active cookie identifiers (session['user']). Unauthorized access requests trigger instant redirects to login prompts.

