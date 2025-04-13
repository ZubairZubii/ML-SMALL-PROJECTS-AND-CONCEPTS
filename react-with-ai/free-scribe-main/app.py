from flask import Flask, request, jsonify
from transformers import pipeline
import io

app = Flask(__name__)

# Initialize pipelines
asr_pipeline = pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en')
translation_pipeline = pipeline('translation', model='Xenova/nllb-200-distilled-600M')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_file = request.files['audio'].read()
    audio = io.BytesIO(audio_file)
    
    transcription_result = asr_pipeline(audio)
    
    return jsonify(transcription_result)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    text = data.get('text')
    tgt_lang = data.get('tgt_lang')
    src_lang = data.get('src_lang')
    
    translation_result = translation_pipeline(text, tgt_lang=tgt_lang, src_lang=src_lang)
    
    return jsonify(translation_result)

if __name__ == '__main__':
    app.run(debug=True)
