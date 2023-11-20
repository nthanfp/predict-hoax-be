import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model dari file
loaded_pac = joblib.load('data_indonesia_model/model_hoax_detection.pkl')
loaded_tfidf_vectorizer = joblib.load('data_indonesia_model/tfidf_vectorizer.pkl')

def predict_hoax(title, text_new):
    # Gabungkan title dan text_new menjadi satu teks
    combined_text = f'{title} {text_new}'

    # Clean text
    cleaned_text = clean_text(combined_text)

    # Transform teks menggunakan TfidfVectorizer yang sudah di-load
    tfidf_text = loaded_tfidf_vectorizer.transform([cleaned_text])  # Pass as a list

    # Lakukan prediksi dengan model Passive Aggressive Classifier yang sudah di-load
    prediction = loaded_pac.predict(tfidf_text)

    return prediction[0]

def clean_text(text):
    # Menghapus karakter non-alfanumerik dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Mengonversi teks ke huruf kecil
    text = text.lower()

    # Menghapus stop words menggunakan Sastrawi StopWordRemover
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    text = stopword_remover.remove(text)

    # Stemming teks menggunakan Sastrawi Stemmer
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    text = stemmer.stem(text)

    return text

# Endpoint 1: /api/status
@app.route('/api/status', methods=['GET'])
def get_status():
    response = {
        'httpCode': 200,
        'status': True,
        'error_msg': None
    }
    return jsonify(response)

# Endpoint 2: /api/news/predict
@app.route('/api/news/predict', methods=['POST'])
def predict_news():
    # Validasi input
    if 'news' not in request.json or 'news_title' not in request.json:
        response = {
            'httpCode': 400,
            'status': False,
            'error_msg': 'Parameter "news" dan "news_title" diperlukan'
        }
        return jsonify(response), 400

    # Mendapatkan data input
    news = request.json['news']
    news_title = request.json['news_title']

    result = predict_hoax(news, news_title)

    # Menyiapkan respons API
    response = {
        'httpCode': 200,
        'status': True,
        'error_msg': None,
        'data': {
            'news': news,
            'news_title': news_title,
            'result_predict': result
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
