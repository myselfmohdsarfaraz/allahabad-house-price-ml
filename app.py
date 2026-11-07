import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Melbourne House Price ML", page_icon="🏡", layout="wide")

st.title("🏡 Allahabad House Price Prediction")
st.caption("An end-to-end Machine Learning project using Decision Tree Regression")

st.markdown("""
This app demonstrates a complete ML workflow:
**load data → choose features → split data → train → validate → predict**.

Upload the Allahabad housing CSV you are using in your Kaggle lesson. The app does not
silently invent real-world house prices: predictions are only meaningful after training
on a suitable real dataset.
""")

with st.sidebar:
    st.header("Project settings")
    uploaded = st.file_uploader("Upload your Allahabad housing CSV", type=["csv"])
    st.markdown("---")
    st.write("**Expected target column:** `Price`")
    st.caption("If your CSV uses another target name, choose it below after uploading.")

@st.cache_data
def read_csv(file_bytes):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded is None:
    st.info("👈 Upload your Allahabad housing CSV to begin. You can download the same CSV used in your Kaggle lesson and upload it here.")
    st.subheader("What this project contains")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Decision Tree")
    c2.metric("Evaluation", "MAE")
    c3.metric("Validation", "Hold-out split")
    st.markdown("""
    ### How to get your dataset
    1. Open the Melbourne Housing dataset used in your course (often named `melb_data.csv` or `Melbourne_housing_FULL.csv`).
    2. Download the CSV file.
    3. Upload it using the box in the left sidebar.

    ### Important
    This is an educational project, not a professional property valuation service.
    A model trained on an old or incomplete dataset may produce inaccurate estimates.
    """)
    st.stop()

try:
    df = read_csv(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

st.success(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")
with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Dataset shape:", df.shape)
    st.write("Column names:", list(df.columns))

st.header("1. Choose the target and input features")
default_target_index = list(df.columns).index("Price") if "Price" in df.columns else 0
target = st.selectbox("Target column (the value the model must predict)", list(df.columns), index=default_target_index)

# Avoid obvious identifiers and target leakage; user can still choose features manually.
id_like = {"id", "address", "date", "sellerG".lower(), "postcode"}
candidate_features = [
    col for col in df.columns
    if col != target and col.strip().lower() not in id_like
]
preferred = [
    "Rooms", "Distance", "Bedroom2", "Bathroom", "Car", "Landsize",
    "BuildingArea", "YearBuilt", "Type", "Method", "Regionname", "CouncilArea"
]
defaults = [c for c in preferred if c in candidate_features]
if not defaults:
    defaults = candidate_features[:min(8, len(candidate_features))]

features = st.multiselect(
    "Feature columns (information used to make predictions)",
    options=candidate_features,
    default=defaults,
    help="Do not select the target column. Start with meaningful house features such as Rooms, Distance, Bathroom and Landsize."
)

if not features:
    st.warning("Choose at least one feature column.")
    st.stop()

try:
    test_size = st.slider("Validation data (%)", min_value=10, max_value=40, value=25, step=5)
    max_depth = st.slider("Decision Tree maximum depth", min_value=2, max_value=30, value=10)
    min_samples_leaf = st.slider("Minimum samples per leaf", min_value=1, max_value=20, value=5)
except Exception:
    test_size, max_depth, min_samples_leaf = 25, 10, 5

work = df[features + [target]].copy()
work[target] = pd.to_numeric(work[target], errors="coerce")
work = work.dropna(subset=[target])
if len(work) < 10:
    st.error("Not enough rows with a numeric target. Please upload a larger dataset with a numeric target such as Price.")
    st.stop()

X = work[features]
y = work[target]

numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [c for c in features if c not in numeric_features]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_features:
    transformers.append(("numeric", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("categorical", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=1
    ))
])

train_X, val_X, train_y, val_y = train_test_split(
    X, y, test_size=test_size / 100, random_state=0
)

st.header("2. Train and validate the model")
if st.button("🚀 Train model and calculate validation MAE", type="primary"):
    model.fit(train_X, train_y)
    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)
    train_mae = mean_absolute_error(train_y, train_predictions)
    val_mae = mean_absolute_error(val_y, val_predictions)

    st.session_state["trained_model"] = model
    st.session_state["feature_names"] = features
    st.session_state["numeric_features"] = numeric_features
    st.session_state["categorical_features"] = categorical_features
    st.session_state["train_mae"] = train_mae
    st.session_state["val_mae"] = val_mae
    st.session_state["val_actual"] = val_y.reset_index(drop=True)
    st.session_state["val_predicted"] = pd.Series(val_predictions)

if "trained_model" in st.session_state and st.session_state.get("feature_names") == features:
    st.subheader("Model validation results")
    train_mae = st.session_state["train_mae"]
    val_mae = st.session_state["val_mae"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Training MAE", f"{train_mae:,.2f}")
    m2.metric("Validation MAE", f"{val_mae:,.2f}")
    m3.metric("Training rows", f"{len(train_X):,}")

    st.markdown(f"""
    **Interpretation:** On average, validation predictions differ from actual values by
    **{val_mae:,.2f} target units**. MAE uses the same unit as your target column.

    If training MAE is much smaller than validation MAE, the tree may be overfitting.
    A lower validation MAE is generally better when comparing models on the same validation split.
    """)
    if val_mae > train_mae * 2:
        st.warning("The validation error is more than twice the training error. This can be a sign of overfitting.")
    else:
        st.info("Compare training and validation MAE together. A small gap is not proof of a perfect model; always inspect predictions and data quality.")

    actual = st.session_state["val_actual"]
    predicted = st.session_state["val_predicted"]
    chart_df = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    st.subheader("Actual vs predicted values")
    st.dataframe(chart_df.head(15).round(2), use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.55)
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Actual vs predicted (closer to dashed line is better)")
    st.pyplot(fig)
    plt.close(fig)

    st.header("3. Predict a house price")
    st.write("Enter values for the selected features. The app will use the trained model to estimate the target.")
    with st.form("prediction_form"):
        input_values = {}
        cols = st.columns(2)
        for i, col in enumerate(features):
            series = X[col]
            with cols[i % 2]:
                if col in numeric_features:
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    default_value = float(numeric_series.median()) if not numeric_series.empty else 0.0
                    input_values[col] = st.number_input(
                        col, value=default_value, help=f"Typical dataset value: {default_value:.2f}"
                    )
                else:
                    choices = series.dropna().astype(str).value_counts().index.tolist()
                    if not choices:
                        choices = ["Unknown"]
                    input_values[col] = st.selectbox(col, choices)
        submitted = st.form_submit_button("Estimate house price")
    if submitted:
        input_df = pd.DataFrame([input_values], columns=features)
        prediction = st.session_state["trained_model"].predict(input_df)[0]
        st.success(f"Estimated {target}: {prediction:,.2f}")
        st.caption("This is a model estimate based on the uploaded dataset, not a guaranteed market price.")

    st.download_button(
        "⬇️ Download validation predictions as CSV",
        data=chart_df.to_csv(index=False).encode("utf-8"),
        file_name="allahabad_validation_predictions.csv",
        mime="text/csv"
    )
else:
    st.caption("Choose your settings, then click **Train model and calculate validation MAE**.")

st.markdown("---")
st.caption("Built with Python, Pandas, Scikit-learn, Matplotlib and Streamlit. Educational use only.")
