import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import SRDataset
from models.models import SRCNN

# 1. Configurare (Căutăm placa video)
# Asta îți va folosi RTX 5060-ul dacă driverele CUDA sunt instalate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Antrenăm pe: {device}")

# 2. Încărcăm Datele
dataset = SRDataset(folder_imagini="data/Set5", factor_marire=3)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 3. Inițializăm AI-ul, Funcția de Eroare și "Antrenorul"
model = SRCNN().to(device)
criterion = nn.MSELoss()  # Funcția care măsoară diferența dintre poze
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr = "viteza" de învățare

# 4. Bucla de Antrenament
epoci = 50  # De câte ori vede AI-ul TOATE pozele din folderr

print("🚀 Începem antrenamentul...")
for epoca in range(epoci):
    loss_total = 0  # Ținem scorul erorii pentru fiecare epocă

    for input_lr, target_hr in loader:
        # Mutăm pozele pe placa video pentru viteză
        input_lr = input_lr.to(device)
        target_hr = target_hr.to(device)

        # A. Curățăm memoria antrenorului
        optimizer.zero_grad()

        # B. AI-ul ghicește poza (Forward)
        output = model(input_lr)

        # C. Calculăm cât a greșit (Loss)
        loss = criterion(output, target_hr)

        # D. Ne dăm seama ce trebuie corectat (Backward)
        loss.backward()

        # E. Aplicăm corecturile pe neuroni (Optimize)
        optimizer.step()

        loss_total += loss.item()

    # Afișăm progresul la finalul fiecărei epoci
    eroare_medie = loss_total / len(loader)
    print(f"Epoca [{epoca + 1}/{epoci}] finalizată | Eroare Medie (Loss): {eroare_medie:.6f}")

print("✅ Antrenament complet! Modelul a învățat să repare poze.")

# 5. Salvăm creierul antrenat într-un fișier ca să nu o luăm de la zero data viitoare
torch.save(model.state_dict(), "srcnn_model_antrenat.pth")
print("💾 Modelul a fost salvat ca 'srcnn_model_antrenat.pth'")