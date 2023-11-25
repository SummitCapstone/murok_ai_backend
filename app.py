from flask import Flask, request
from diagnosis import diagnose
app = Flask(__name__)


@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    image = request.files['image']
    crop_type = request.form['crop_type']
    top_diseases, top_probabilities = diagnose(image, crop_type)
    return {'top_diseases': top_diseases, 'top_probabilities': top_probabilities}
