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

class_names = ['benign', 'malignant']
file_id = "1xbAvjokOhu_uo0jq7_brPWwuf5bRnLwe"
destination = "skin_classifier.pt"

# Crée une vraie URL de téléchargement
url = f"https://drive.google.com/uc?id={file_id}"

# Télécharge si nécessaire
if not os.path.exists(destination):
    gdown.download(url, destination, quiet=False)
    if os.path.getsize(destination) < 1_000_000:  # Moins de 1 Mo → probablement mauvais fichier
        raise RuntimeError("Fichier téléchargé invalide ou trop petit. Vérifiez le lien Google Drive.")

with open(destination, "rb") as f:
    start = f.read(200)
    print("=== DEBUT DU FICHIER ===")
    print(start)
    
# === CHARGEMENT DU MODÈLE ===
# Créer le modèle avec même architecture qu'à l'entraînement
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

# Charger les poids
model.load_state_dict(torch.load(destination, map_location=device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(destination, map_location=device, weights_only=False)  # ou "cuda" selon le besoin
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
