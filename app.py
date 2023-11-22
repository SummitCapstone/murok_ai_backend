from flask import Flask, request
from diagnosis import diagnose
app = Flask(__name__)


@app.route('/diagnosis', methods=['POST'])
def detectIssues():
    board_image_id = request.form.get('boardImageId')
    image = request.files['image']
    result = diagnose(image)

    return {'boardImageId': board_image_id, 'issues': result}