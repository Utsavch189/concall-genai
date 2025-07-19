import os
import requests
from flask import Flask, request, jsonify,render_template
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEy')

@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    print("AUDIO FILE : ",audio_file," ",audio_file.content_length)

    response = requests.post(
        "https://api.openai.com/v1/audio/translations",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={
            "file": (audio_file.filename, audio_file.stream, audio_file.mimetype),
            "model": (None, "whisper-1"),
        },
    )

    if response.status_code != 200:
        return jsonify({"error": response.text}), response.status_code

    return jsonify({"text": response.json()["text"]})

if __name__ == "__main__":
    app.run(debug=True,port=8000,host="0.0.0.0")