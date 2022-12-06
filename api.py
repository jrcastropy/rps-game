import cv2
from utils import MediaPipeHand
from utils import GestureRecognition
import requests
import os

from flask_restful import Resource, Api
from flask import Flask, request, jsonify
from flask_cors import CORS

def download_file(url, custom_path=None, custom_fn=None):
    file_path = os.path.join('downloads', 'rps.png')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return file_path

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

# Load mediapipe hand class
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=1)

# Load gesture recognition class
gest = GestureRecognition(mode='eval')

# file = '/root/rps-game/Rock-paper-scissors_(scissors).png'
# file = '/root/rps-game/20069339.jpg'
# file = '/root/rps-game/rock-paper-scissors-rock-hand-isolated-white-31662043.jpg'
# file = '/root/rps-game/hand-2704013_1280.jpg'
# file = 'sci2.png'

class CApp(Resource):
    @staticmethod
    def get():

        # file = 'https://facts.net/wp-content/uploads/2020/11/hand-2704013_1280.jpg'

        file = request.args.get('hand')

        if 'https' in file:
            download_file(file)
            file = os.path.join('downloads', 'rps.png')

        img = cv2.flip(cv2.imread(file), 1)

        param = pipe.forward(img)
        gesture = 'no gesture detected'

        for p in param:
            if p['class'] is not None:
                gesture = gest.eval(p['angle'])  

        if gesture == 'fist' or gesture == 'one':
            hn = 'rock'
        elif gesture == 'five':
            hn = 'paper'
        elif gesture == 'yeah':
            hn = 'paper'
        else:
            hn = 'no sign detected'

        return jsonify({'result':hn})

api.add_resource(CApp, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234, debug=True)