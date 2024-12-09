# Uporabimo uradno PyTorch sliko z vgrajeno podporo za CUDA in Python 3.8
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Nastavimo delovno mapo
WORKDIR /AMS_izziv_trainingCT

# Kopiramo datoteko requirements.txt
COPY requirements.txt .

# Namestimo pakete
RUN pip install -r requirements.txt

# Kopiramo vsebino iz gostitelja v vsebino mape (nahajamo se v 'WORKDIR')
COPY . .

# Določimo ukaz, ki bo zagnal aplikacijo
CMD ["python", "ViT-V-Net/train.py"]
