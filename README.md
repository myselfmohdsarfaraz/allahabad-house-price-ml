# Allahabad House Price Prediction — Machine Learning Web App

A beginner-friendly end-to-end Machine Learning project using **Decision Tree Regression**.
The app lets you upload a Allahabad housing CSV, choose the target and features, train a model,
evaluate it with Mean Absolute Error (MAE), visualize actual vs predicted values, and estimate a
house price from user-entered features.

## Features

- Upload your own Allahabad housing CSV
- Preview dataset rows, shape, and column names
- Select the target column (usually `Price`) and input features
- Handle missing numeric and categorical values
- Convert categorical features using one-hot encoding
- Split data into training and validation sets
- Train a Decision Tree Regressor
- Compare training MAE with validation MAE
- Visualize actual vs predicted values
- Make a prediction using a simple form
- Download validation predictions as CSV

## 1. Get the dataset

Use the Allahabad housing CSV supplied by your Kaggle course/dataset. Common filenames include
`allahabad_housing.csv` and `Allahabad_housing_FULL.csv`.

The app expects a CSV with a numeric target column such as `Price`. You upload the file inside
the app; the dataset is not included in this repository.

## 2. Run locally

Install Python 3.10 or newer.

Open Terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the website:

```bash
streamlit run app.py
```

A browser tab should open. Upload your CSV using the uploader in the left sidebar.

## 3. Push the project to GitHub

1. Sign in to GitHub and create a new **public** repository named `allahabad-house-price-ml`.
2. Do not initialize it with a README (this folder already contains one).
3. Open Terminal inside this project folder.
4. Run:

```bash
git init
git add .
git commit -m "Build Allahabad house price prediction ML app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/allahabad-house-price-ml.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

If Git says `remote origin already exists`, run:

```bash
git remote set-url origin https://github.com/YOUR-USERNAME/allahabad-house-price-ml.git
git push -u origin main
```

## 4. Deploy it free with Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**.
4. Select your repository, branch `main`, and main file path `app.py`.
5. Deploy.
6. Open your live app and upload the CSV.

The deployed app needs the CSV to be uploaded by the user each time the app session needs it. Do not commit a private or restricted dataset to a public repository.

## 5. Understand the ML workflow

- `X`: selected input features, such as Rooms, Bathroom, and Landsize.
- `y`: target values, usually Price.
- `train_test_split`: holds out data for validation.
- `fit()`: learns patterns from training examples.
- `predict()`: generates predictions for new feature rows.
- `mean_absolute_error()`: average absolute difference between actual and predicted values.

### Why use a preprocessing Pipeline?

Real housing datasets can contain missing values and text categories. The pipeline:
- fills missing numeric values with the median,
- fills missing categories with the most frequent category,
- one-hot encodes text categories,
- trains the Decision Tree.

This ensures the same transformations are applied during training and prediction.

## Important limitations

- This is an educational demonstration, not a professional property valuation tool.
- Model quality depends on the dataset, selected features, data quality, and time period.
- A validation MAE is an estimate of performance on held-out data, not a guarantee for future market prices.
- Do not use the validation set repeatedly to tune the model and then treat it as a final unbiased test. For a stronger evaluation, keep a separate test set.
- House-price datasets can contain leakage, outdated records, duplicates, and geographic/time bias. Inspect your data before drawing conclusions.
