from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from io import BytesIO
import base64
import gdown
import os
import json
import torch.nn.functional as F

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "skin_classifier.pt")

MODEL_GDRIVE_ID = "TON_ID_FICHIER_GDRIVE_MODEL"
CLASS_NAMES_GDRIVE_ID = "TON_ID_FICHIER_GDRIVE_CLASS_NAMES"

def download_file_if_not_exists(url_id, output_path):
    if not os.path.exists(output_path):
        print(f"Téléchargement du fichier dans {output_path} ...")
        gdown.download(f"https://drive.google.com/uc?id={url_id}", output_path, quiet=False)
    else:
        print(f"Fichier {output_path} déjà présent, téléchargement ignoré.")

os.makedirs(MODEL_FOLDER, exist_ok=True)
download_file_if_not_exists(MODEL_GDRIVE_ID, MODEL_PATH)

class_names = ['benign', 'malignant']

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = None
    scores = None  # pourcentages des classes
    bar_color = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream).convert('RGB')

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_data = base64.b64encode(buffered.getvalue()).decode()

            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()[0]
                predicted_idx = probs.argmax()
                prediction = class_names[predicted_idx]
                scores = {class_names[i]: round(float(probs[i]*100), 2) for i in range(len(class_names))}

            # Choix couleur barre: vert si benign majoritaire, rouge si malignant majoritaire
            bar_color = "#4CAF50" if prediction == "benign" else "#F44336"

    return render_template('index.html', prediction=prediction, image_data=image_data, scores=scores, bar_color=bar_color)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
