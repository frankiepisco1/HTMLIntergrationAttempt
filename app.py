from flask import Flask, render_template, request, jsonify
import threading
import speech_recognition as sr
import whisper
import torch
import numpy as np
import os
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

data_queue = Queue()
transcription = ['']
buffer = ""
transcription_started = False

def transcribe_loop():
    global buffer
    global transcription_started
    audio_model = whisper.load_model("large")
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=1)

    while transcription_started:
        if not data_queue.empty():
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = audio_model.transcribe(audio_np, language="en", fp16=torch.cuda.is_available())
            text = result['text'].strip()

            transcription.append(text)
            buffer = ' '.join(transcription[-10:])  # Keep last 10 phrases
            sleep(0.25)
        else:
            sleep(0.25)

@app.route('/')
def index():
    logging.info("Serving the index page")
    return render_template('index.html')

@app.route('/start_transcription', methods=['POST'])
def start_transcription():
    global transcription_started
    if not transcription_started:
        transcription_started = True
        threading.Thread(target=transcribe_loop, daemon=True).start()
    logging.info("Started transcription from microphone")
    return jsonify({"status": "started"})

@app.route('/transcribe_file', methods=['POST'])
def transcribe_file():
    logging.info("Received request to transcribe file")
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        # Convert .m4a file to .wav using pydub if necessary
        if filepath.endswith('.m4a'):
            logging.info("Converting .m4a file to .wav")
            audio = AudioSegment.from_file(filepath, format='m4a')
            filepath = filepath.replace('.m4a', '.wav')
            audio.export(filepath, format='wav')
            logging.info(f"File converted to {filepath}")

        audio_model = whisper.load_model("large")
        result = audio_model.transcribe(filepath, language="en", fp16=torch.cuda.is_available())
        text = result['text'].strip()
        logging.info("Transcription completed")
        return jsonify({"transcription": text})
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_transcription', methods=['GET'])
def get_transcription():
    global buffer
    logging.info("Sending transcription buffer")
    return jsonify({"transcription": buffer})

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
