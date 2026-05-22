import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pickle

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from tf_keras.models import load_model

_MODEL_CACHE = {}
_SKLEARN_CACHE = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder='templates')

STATIC_PAGES = (
    'index.html',
    'about-us.html',
    'services.html',
    'contact.html',
    'index-icons.html',
    'elements.html',
    'single-blog.html',
)


def get_keras_model(model_path):
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = load_model(model_path)
    return _MODEL_CACHE[model_path]


DISEASE_FEATURE_ORDER = {
    'diabetes': [
        'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
        'insulin', 'bmi', 'dpf', 'age',
    ],
    'breast_cancer': [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave_points_mean', 'symmetry_mean', 'radius_se', 'perimeter_se',
        'area_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
        'concavity_worst', 'concave_points_worst', 'symmetry_worst',
        'fractal_dimension_worst',
    ],
    'heart': [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal',
    ],
    'kidney': [
        'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc',
        'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane',
    ],
    'liver': [
        'Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
        'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Gender_Male',
    ],
}

DISEASE_TEMPLATES = {
    'diabetes': 'diabetes.html',
    'breast_cancer': 'breast_cancer.html',
    'heart': 'heart.html',
    'kidney': 'kidney.html',
    'liver': 'liver.html',
}

MODEL_FILES = {
    'diabetes': 'models/diabetes.pkl',
    'breast_cancer': 'models/breast_cancer.pkl',
    'heart': 'models/heart.pkl',
    'kidney': 'models/kidney.pkl',
    'liver': 'models/liver.pkl',
}


def get_sklearn_model(model_path):
    if model_path not in _SKLEARN_CACHE:
        with open(model_path, 'rb') as model_file:
            _SKLEARN_CACHE[model_path] = pickle.load(model_file)
    return _SKLEARN_CACHE[model_path]


def predict_disease(disease_type, values):
    model = get_sklearn_model(MODEL_FILES[disease_type])
    feature_array = np.asarray(values, dtype=float).reshape(1, -1)
    return int(model.predict(feature_array)[0])


@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')


@app.route('/home')
def home_simple():
    return render_template('home.html')


for page in STATIC_PAGES:
    if page == 'index.html':
        continue

    endpoint = page.replace('.html', '').replace('-', '_')

    def make_view(template_name=page):
        def view():
            return render_template(template_name)

        return view

    app.add_url_rule(
        f'/{page}',
        endpoint=endpoint,
        view_func=make_view(),
        methods=['GET'],
    )


@app.route('/style.css')
@app.route('/style.css.map')
def serve_root_assets():
    return send_from_directory(TEMPLATE_DIR, request.path.lstrip('/'))


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(TEMPLATE_DIR, 'css'), filename)


@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(TEMPLATE_DIR, 'js'), filename)


@app.route('/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(os.path.join(TEMPLATE_DIR, 'img'), filename)


@app.route('/fonts/<path:filename>')
def serve_fonts(filename):
    return send_from_directory(os.path.join(TEMPLATE_DIR, 'fonts'), filename)


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')


@app.route('/cancer', methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')


@app.route('/heart', methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')


@app.route('/kidney', methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')


@app.route('/liver', methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')


@app.route('/malaria', methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')


@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')


@app.route('/predict', methods=['POST', 'GET'])
def predictPage():
    if request.method != 'POST':
        return render_template('index.html')

    disease_type = request.form.get('disease_type')
    if disease_type not in DISEASE_FEATURE_ORDER:
        return render_template('index.html', message='Invalid prediction request.')

    form_template = DISEASE_TEMPLATES[disease_type]
    error_message = 'Please enter valid numeric data in all fields.'

    try:
        feature_names = DISEASE_FEATURE_ORDER[disease_type]
        values = [float(request.form[field]) for field in feature_names]
        pred = predict_disease(disease_type, values)
        return render_template('predict.html', pred=pred)
    except (KeyError, ValueError, TypeError):
        return render_template(form_template, message=error_message)
    except Exception:
        return render_template(
            form_template,
            message='Prediction failed. Please check your values and try again.',
        )


@app.route('/malariapredict', methods=['POST', 'GET'])
def malariapredictPage():
    if request.method != 'POST':
        return render_template('malaria.html', message='Please upload an image')

    image_file = request.files.get('image')
    if not image_file or not image_file.filename:
        return render_template('malaria.html', message='Please upload an image')

    try:
        img = Image.open(image_file)
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 3))
        img = img.astype(np.float64)
        model = get_keras_model('models/malaria.h5')
        pred = int(np.argmax(model.predict(img, verbose=0)[0]))
        return render_template('malaria_predict.html', pred=pred)
    except Exception:
        return render_template(
            'malaria.html',
            message='Could not process the image. Please upload a valid JPG or PNG file.',
        )


@app.route('/pneumoniapredict', methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method != 'POST':
        return render_template('pneumonia.html', message='Please upload an image')

    image_file = request.files.get('image')
    if not image_file or not image_file.filename:
        return render_template('pneumonia.html', message='Please upload an image')

    try:
        img = Image.open(image_file).convert('L')
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 1))
        img = img / 255.0
        model = get_keras_model('models/pneumonia.h5')
        pred = int(np.argmax(model.predict(img, verbose=0)[0]))
        return render_template('pneumonia_predict.html', pred=pred)
    except Exception:
        return render_template(
            'pneumonia.html',
            message='Could not process the image. Please upload a valid JPG or PNG file.',
        )


if __name__ == '__main__':
    app.run(debug=True)
