import torch
from data_loader import SRDataset
from torch.utils.data import DataLoader
from models.models import SRCNN
import matplotlib.pyplot as plt

# 1. Pregătim datele
dataset = SRDataset(folder_imagini="data/Set5", factor_marire=3)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
input_lr, target_hr = next(iter(loader))

# 2. Inițializăm Modelul SRCNN
model = SRCNN()
model.eval()

# 3. Facem o primă "predicție"
with torch.no_grad():
    output = model(input_lr)

print(f"--- Progres Proiect IA (Pasul 2) ---")
print(f"Modelul SRCNN a fost inițializat cu succes.")
print(f"Dimensiune Output de la AI: {output.shape}")

# 4. Vizualizăm rezultatele
output_img = output[0].permute(1, 2, 0).clamp(0, 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input (Low-Res)")
plt.imshow(input_lr[0].permute(1, 2, 0))

plt.subplot(1, 3, 2)
plt.title("AI Output (Untrained)")
plt.imshow(output_img)

plt.subplot(1, 3, 3)
plt.title("Target (High-Res)")
plt.imshow(target_hr[0].permute(1, 2, 0))
plt.show()