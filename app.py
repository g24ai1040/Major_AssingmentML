import io
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string, redirect, url_for

# Minimal HTML form
HTML = """
<!doctype html>
<title>Olivetti Face Classifier</title>
<h1>Upload an image (grayscale or color) to predict Olivetti class</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h2>Predicted class: {{ pred }}</h2>
  <p>Class label is a number 0..39 corresponding to persons in Olivetti dataset.</p>
{% endif %}
"""

app = Flask(__name__)
model_data = joblib.load("savedmodel.pth")
model = model_data["model"]

def preprocess_image(file_stream):
    # Open image, convert to grayscale, resize to 64x64, normalize to match Olivetti (0..1 floats)
    img = Image.open(io.BytesIO(file_stream)).convert("L")  # grayscale
    img = img.resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, -1)  # flatten to 1 x 4096
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded:
            file_bytes = uploaded.read()
            x = preprocess_image(file_bytes)
            y_pred = model.predict(x)
            pred = int(y_pred[0])
    return render_template_string(HTML, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
