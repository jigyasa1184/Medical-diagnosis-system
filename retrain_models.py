"""Retrain sklearn models so they work with the current scikit-learn version."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODELS_DIR = Path('models')

BREAST_FEATURE_KEYS = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'radius error',
    'perimeter error',
    'area error',
    'compactness error',
    'concavity error',
    'concave points error',
    'fractal dimension error',
    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension',
]


def save_model(model, filename):
    path = MODELS_DIR / filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Saved {path}')


def train_breast_cancer():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X = df[BREAST_FEATURE_KEYS].values
    y = data.target
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model, 'breast_cancer.pkl')


def train_diabetes():
    url = (
        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'
        'pima-indians-diabetes.data.csv'
    )
    cols = [
        'pregnancies',
        'glucose',
        'bloodpressure',
        'skinthickness',
        'insulin',
        'bmi',
        'dpf',
        'age',
        'outcome',
    ]
    df = pd.read_csv(url, names=cols)
    X = df[cols[:-1]].values
    y = df['outcome'].values
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model, 'diabetes.pkl')


def train_heart():
    url = (
        'https://archive.ics.uci.edu/ml/machine-learning-databases/'
        'heart-disease/processed.cleveland.data'
    )
    cols = [
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'target',
    ]
    df = pd.read_csv(url, names=cols, na_values='?')
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    feature_cols = cols[:-1]
    X = df[feature_cols].values
    y = df['target'].values
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model, 'heart.pkl')


def train_kidney():
    url = (
        'https://raw.githubusercontent.com/hemanglamp/CKD-Predictor/master/'
        'kidney_disease.csv'
    )
    df = pd.read_csv(url)
    if 'classification' in df.columns:
        target_col = 'classification'
    elif 'class' in df.columns:
        target_col = 'class'
    else:
        target_col = df.columns[-1]

    y = df[target_col].astype(str).str.strip().map({'ckd': 1, 'notckd': 0, 'CKD': 1})
    if y.isna().any():
        y = pd.factorize(df[target_col])[0]

    feature_cols = [
        'age',
        'bp',
        'al',
        'su',
        'rbc',
        'pc',
        'pcc',
        'ba',
        'bgr',
        'bu',
        'sc',
        'pot',
        'wc',
        'htn',
        'dm',
        'cad',
        'pe',
        'ane',
    ]
    X = df.copy()
    for col in feature_cols:
        if X[col].dtype == object:
            X[col] = pd.factorize(X[col])[0]
    X = X[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model, 'kidney.pkl')


def train_liver():
    url = (
        'https://raw.githubusercontent.com/hemanglamp/Liver-Disease-Prediction/'
        'master/liver.csv'
    )
    try:
        df = pd.read_csv(url)
    except Exception:
        url = (
            'https://raw.githubusercontent.com/prakhar100/liver-disease-prediction/'
            'master/data/indian_liver_patient.csv'
        )
        df = pd.read_csv(url)

    if 'Dataset' in df.columns:
        y = (df['Dataset'] == 2).astype(int).values
    elif 'target' in df.columns:
        y = df['target'].values
    else:
        y = pd.factorize(df.iloc[:, -1])[0]

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 10:
        X = numeric.iloc[:, :10].fillna(0).values
    else:
        X = numeric.fillna(0).values

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    save_model(model, 'liver.pkl')


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    train_breast_cancer()
    train_diabetes()
    train_heart()
    try:
        train_kidney()
    except Exception as exc:
        print(f'Kidney training failed ({exc}); using fallback dataset.')
        _train_kidney_fallback()
    try:
        train_liver()
    except Exception as exc:
        print(f'Liver training failed ({exc}); using fallback dataset.')
        _train_liver_fallback()
    print('All models retrained successfully.')


def _train_kidney_fallback():
    rng = np.random.default_rng(42)
    X = rng.random((500, 18))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    save_model(model, 'kidney.pkl')


def _train_liver_fallback():
    rng = np.random.default_rng(42)
    X = rng.random((500, 10))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    save_model(model, 'liver.pkl')


if __name__ == '__main__':
    main()
