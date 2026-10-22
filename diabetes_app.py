

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# matplotlib / seaborn are only needed for the graphs. If they are missing,
# the rest of the app still works and a clear message is shown instead.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "diabetes.csv"
MODEL_FILE = APP_DIR / "Diabetesmodel.pkl"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
# In this dataset a 0 in these columns means "not measured", not a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# The four models trained and compared in this app.
MODEL_NAMES = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree"]

RANDOM_STATE = 42


def build_search(name):
    """Return a GridSearchCV that tunes one model (preprocessing included)."""
    # Zeros in the "not measured" columns are treated as missing and replaced by the
    # median of the training data; the other columns pass through unchanged.
    steps = [
        ("impute", ColumnTransformer(
            [("zeros_as_missing", SimpleImputer(missing_values=0, strategy="median"),
              ZERO_AS_MISSING)],
            remainder="passthrough",
        )),
    ]
    if name == "Logistic Regression":
        steps += [("scale", StandardScaler()),
                  ("model", LogisticRegression(max_iter=2000))]
        grid = {"model__C": [0.01, 0.1, 1, 10]}
    elif name == "KNN":
        steps += [("scale", StandardScaler()),
                  ("model", KNeighborsClassifier())]
        grid = {
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
        }
    elif name == "Random Forest":
        steps += [("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        grid = {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        }
    else:  # Decision Tree
        steps += [("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]
        grid = {
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__min_samples_leaf": [1, 5, 10, 20],
        }
    return GridSearchCV(
        Pipeline(steps),
        grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
    )


def load_dataframe():
    """Read diabetes.csv and tidy the column names."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE.name}. Put diabetes.csv in the same folder as diabetes_app.py."
        )
    df = pd.read_csv(DATA_FILE)
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {
        "blood pressure": "BloodPressure", "bloodpressure": "BloodPressure",
        "glucose level": "Glucose", "skin thickness": "SkinThickness",
        "skinthickness": "SkinThickness", "bmi": "BMI", "age": "Age",
        "outcome": "Outcome", "pregnancies": "Pregnancies", "insulin": "Insulin",
        "diabetespedigreefunction": "DiabetesPedigreeFunction",
        "diabetes pedigree function": "DiabetesPedigreeFunction",
    }
    df = df.rename(columns={
        c: aliases[c.replace("_", " ").strip().lower()]
        for c in df.columns if c.replace("_", " ").strip().lower() in aliases
    })
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing these required columns: " + ", ".join(missing)
            + ". Expected the standard Pima diabetes columns."
        )
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    if len(df) < 50 or df[TARGET].nunique() != 2:
        raise ValueError("Not enough valid rows, or both outcome classes are not present.")
    return df


@st.cache_resource
def train_all_models():
    """Train + tune all four models and collect their results."""
    df = load_dataframe()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models, rows, matrices, best_params = {}, [], {}, {}
    for name in MODEL_NAMES:
        search = build_search(name).fit(X_train, y_train)
        best = search.best_estimator_
        y_pred = best.predict(X_test)
        models[name] = best
        matrices[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        rows.append({
            "Model": name,
            "Cross-validated accuracy": search.best_score_,
            "Held-out test accuracy": accuracy_score(y_test, y_pred),
        })

    # Save every full pipeline (preprocessing + model) together.
    with MODEL_FILE.open("wb") as f:
        pickle.dump(models, f)

    results = pd.DataFrame(rows)
    return models, results, matrices, best_params, len(df), df


@st.cache_resource
def make_pairplot(df):
    """Build the scatter-plot grid once (it is slow to draw on every rerun)."""
    cols = ["Glucose", "BloodPressure", "BMI", "Age"]
    plot_data = df[cols + [TARGET]].copy()
    for c in ["Glucose", "BloodPressure", "BMI"]:
        plot_data.loc[plot_data[c] == 0, c] = np.nan  # zeros are "not measured"
    plot_data[TARGET] = plot_data[TARGET].map({0: "No Diabetes", 1: "Diabetes"})
    grid = sns.pairplot(plot_data, vars=cols, hue=TARGET, diag_kind="hist", height=2)
    return grid.fig


st.set_page_config(page_title="AI Diabetes Risk Assessment", page_icon="🩺", layout="wide")


st.markdown(
    """
    <style>
    /* Overall page */
    .stApp {
        background: #f3f6fb;
        color: #1e293b;
    }

    /* Main dashboard spacing */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Fix headings and text in the light main area */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label {
        color: #1e293b;
    }

    /* Dark blue banner */
    .hero {
        background: linear-gradient(120deg, #102447, #2458a6);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(16, 36, 71, 0.15);
    }

    .hero h1,
    .hero p {
        color: white !important;
    }

    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.8rem;
    }

    .hero p {
        font-size: 1.1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #252832;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #f8fafc !important;
    }

    /* Input boxes */
    [data-testid="stNumberInput"] input {
        border-radius: 8px;
    }

    /* Prediction metrics */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1e293b !important;
    }

    /* Predict button */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: #ef4444;
        border: none;
        border-radius: 9px;
        min-height: 3rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: #dc2626;
        border: none;
    }

    /* Status messages */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Explanatory text */
    .small-note {
        color: #526174;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🩺 AI Diabetes Risk Assessment</h1>'
    '<p>Enter the patient measurements to see the model\'s prediction.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Patient Data")
    st.write("Enter the values below.")
    model_choice = st.selectbox(
        "Model", MODEL_NAMES, index=0,
        help="Which of the four trained models makes the prediction.",
    )
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1)
    glucose = st.number_input("Glucose", min_value=1.0, max_value=300.0, value=120.0, step=1.0)
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=1.0, max_value=200.0, value=70.0, step=1.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness (0 = not measured)", min_value=0.0, max_value=100.0, value=20.0, step=1.0
    )
    insulin = st.number_input(
        "Insulin (0 = not measured)", min_value=0.0, max_value=900.0, value=79.0, step=1.0
    )
    bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
    pedigree = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

st.subheader("Patient Measurements")
r1 = st.columns(4)
r1[0].metric("Pregnancies", str(pregnancies))
r1[1].metric("Glucose", f"{glucose:g}")
r1[2].metric("Blood Pressure", f"{blood_pressure:g}")
r1[3].metric("Skin Thickness", f"{skin_thickness:g}")
r2 = st.columns(4)
r2[0].metric("Insulin", f"{insulin:g}")
r2[1].metric("BMI", f"{bmi:.1f}")
r2[2].metric("Pedigree Function", f"{pedigree:.2f}")
r2[3].metric("Age", str(age))


try:
    with st.spinner("Training and tuning the four models (first run only)..."):
        models, results, matrices, best_params, training_rows, df = train_all_models()
    st.success(f"Model ready. Trained with {training_rows} dataset rows.")
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.info("Check that diabetes.csv is in the same folder and has the required columns.")
    st.stop()

model = models[model_choice]
labels = ["No Diabetes", "Diabetes"]

# ---------------------------------------------------------------------------
# Graphs and dataset (shown directly below the "Model ready" message)
# ---------------------------------------------------------------------------

# Dataset table
st.header("Dataset")
st.dataframe(df.head(10), use_container_width=True)

if PLOTS_AVAILABLE:
    # Diabetes outcome graph
    st.header("Data Visualization")
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Distribution of Diabetes Outcome")
    ax.set_xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    ax.set_ylabel("Number of Patients")
    st.pyplot(fig)
    plt.close(fig)

    # Scatter-plot grid
    st.header("Feature Relationships")
    st.pyplot(make_pairplot(df))
else:
    st.warning(
        "Graphs are disabled because matplotlib and/or seaborn are not installed in the "
        "Python that runs this app. Install them with: "
        "`python -m pip install matplotlib seaborn` and restart the app."
    )

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
st.header("Model Comparison")
st.dataframe(
    results.style.format({
        "Cross-validated accuracy": "{:.1%}",
        "Held-out test accuracy": "{:.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)


if PLOTS_AVAILABLE:
    chart_df = results.melt(
        id_vars="Model",
        value_vars=["Cross-validated accuracy", "Held-out test accuracy"],
        var_name="Measure", value_name="Accuracy",
    )
    fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4))
    sns.barplot(data=chart_df, x="Model", y="Accuracy", hue="Measure", ax=ax_cmp)
    ax_cmp.set_ylim(0.6, 0.85)
    ax_cmp.set_title("Accuracy by model")
    ax_cmp.legend(loc="lower right", fontsize=8)
    st.pyplot(fig_cmp)
    plt.close(fig_cmp)

# Model performance on held-out test data (for the selected model)
st.header(f"Model Performance on Test Data: {model_choice}")
chosen = results.set_index("Model").loc[model_choice]
st.write(f"**Test accuracy:** {chosen['Held-out test accuracy']:.2%}")
st.write(f"**Cross-validated accuracy:** {chosen['Cross-validated accuracy']:.2%}")
st.caption("Best settings found: " + ", ".join(f"{k} = {v}" for k, v in best_params[model_choice].items()))

if PLOTS_AVAILABLE:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrices[model_choice],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_cm,
    )
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, pedigree, age]],
        columns=FEATURES,
    )
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    st.subheader(f"Prediction ({model_choice})")
    if prediction == 1:
        st.error("The model predicts a higher likelihood of diabetes.")
    else:
        st.success("The model predicts a lower likelihood of diabetes.")

    st.subheader("Prediction Probabilities")
    p1, p2 = st.columns(2)
    p1.metric("Not having diabetes", f"{probabilities[0]:.2%}")
    p2.metric("Having diabetes", f"{probabilities[1]:.2%}")

    if PLOTS_AVAILABLE:
        fig_prob, ax_prob = plt.subplots(figsize=(5, 2.5))
        sns.barplot(x=labels, y=probabilities, ax=ax_prob)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel("Probability")
        ax_prob.set_title("Prediction Probabilities")
        st.pyplot(fig_prob)
        plt.close(fig_prob)

    st.caption(
        "Test accuracy describes performance on this dataset split only. "
        "It is not the probability that this individual has diabetes."
    )
else:
    st.info("Adjust the values in the left panel, then click **Predict**.")

