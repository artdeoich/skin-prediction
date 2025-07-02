import os
import gdown
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io

# === Config. ===
MODEL_PATH = "skin-prediction-0-93.keras"

IMG_SIZE = 256  # Doit correspondre à l'input du modèle

GDRIVE_URL = "https://drive.google.com/uc?id=1dy4Dq_pQeMPPZXAr--W8djCGRxZ8JX-_"  # Remplace par l'ID du fichier .h5

# === Download model from Google Drive ===
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# === Load model ===
model = load_model(MODEL_PATH)

# === Class labels ===
CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# === Create Flask app ===
app = Flask(__name__, template_folder='app/templates')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide"}), 400

    try:
        # Lecture de l'image depuis le buffer
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prédiction
        preds = model.predict(image)[0]
        pred_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))
        predictions_by_class = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, preds)
        }

        return jsonify({
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "predictions_by_class": predictions_by_class
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
