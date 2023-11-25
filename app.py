from flask import Flask, request
from diagnosis import diagnose
app = Flask(__name__)


@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    image = request.files['image']
    crop_type = request.form['crop_type']
    # Handle both GET and POST requests here
    if request.method == 'GET':
        return {"crop_type": crop_type}
    # Assuming you want to handle POST requests as well

    top_diseases, top_probabilities = diagnose(image, crop_type)
    return {'top_diseases': top_diseases, 'top_probabilities': top_probabilities}
