# Utiliser une image officielle Python slim
FROM python:3.10-slim

# Installer dépendances système nécessaires (Pillow, torch, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Créer dossier app
WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Exposer port Flask
EXPOSE 5000

# Lancer Flask
CMD ["python", "main.py"]
