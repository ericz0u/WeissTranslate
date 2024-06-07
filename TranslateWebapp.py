from flask import Flask, request, jsonify, render_template
import os
import io
from google.cloud import vision
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__, template_folder='templates')

# Set up the Vision API client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/ericzou/PycharmProjects/huggingfacefinetune/WeissTLServiceKey.json'
client = vision.ImageAnnotatorClient()
tokenizer = AutoTokenizer.from_pretrained("EricZ0u/WeissTranslate")
model = AutoModelForSeq2SeqLM.from_pretrained("EricZ0u/WeissTranslate")

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        texts = detect_text(file_path)
        detected_text = texts[0].description if texts else ""
        translated_text = translate_text(detected_text)
        return render_template('result.html', original_text=detected_text, translated_text=translated_text)

@app.route('/upload_another')
def upload_another():
    return render_template('upload.html')



def detect_text(path):
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            f'{response.error.message}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'
        )

    return texts

def translate_text(text):
    tokenizer.src_lang = "ja_XX"
    encoded_ar = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_ar,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    return(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

if __name__ == '__main__':
    app.run(debug=True)
