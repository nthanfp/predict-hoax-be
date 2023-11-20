from flask import Flask, request, jsonify

app = Flask(__name__)

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

    # Menyiapkan respons API
    response = {
        'httpCode': 200,
        'status': True,
        'error_msg': None,
        'input_params': {
            'news': news,
            'news_title': news_title
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
