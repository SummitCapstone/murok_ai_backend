import logging
from flask import Flask, request
from diagnosis import diagnose

app = Flask(__name__)

logging.basicConfig(filename='app.log', level=logging.DEBUG)

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    try:
        image = request.files['image']
        crop_type = request.form['crop_type']
        
        logging.debug(f"Received POST Request for crop_type={crop_type}")

        top_diseases, top_probabilities = diagnose(image, crop_type)

        logging.debug("Diagnosis completed successfully")

        return {'top_diseases': top_diseases, 'top_probabilities': top_probabilities}

    except Exception as e:
        logging.error(f"Error during diagnosis: {str(e)}")
        return {'error': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
