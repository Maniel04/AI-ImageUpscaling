import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import SRDataset
from models.models import SRCNN, ESPCN

print("=========================================")
print("🤖 Meniu Antrenament AI (Super-Rezoluție)")
print("=========================================")

# 1. Alegem Modelul
print("Ce model vrei să antrenezi?")
print("1. SRCNN (Algoritmul Vechi)")
print("2. ESPCN (Algoritmul Modern)")
alegere_model = input("👉 Introdu 1 sau 2: ")

# 2. Alegem Datele și Epocile
nume_folder = input("📁 Introdu numele folderului cu date (ex: Set_Extins): ")
cale_date = f"data/{nume_folder}"
epoci_alese = int(input("⏳ Câte epoci vrei să ruleze? (ex: 5000): "))

# 3. Configurare Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Folosim placa video/procesorul: {device}")

dataset = SRDataset(cale_date)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 4. Setăm Modelul și Numele Salvării în funcție de alegere
if alegere_model == "1":
    model = SRCNN().to(device)
    nume_salvare = "srcnn_model_antrenat.pth"
    print("⚙️ AI configurat: SRCNN. Va scoate imagini la dimensiune 1:1.")
else:
    model = ESPCN(scale_factor=3).to(device)
    nume_salvare = "espcn_model_antrenat.pth"
    print("⚙️ AI configurat: ESPCN. Va mări imaginile de 3 ori (Pixel Shuffle).")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 5. BUCLA DE ANTRENAMENT
# ==========================================
model.train()
print(f"🚀 Pornim antrenamentul pe '{cale_date}' pentru {epoci_alese} epoci...")

for epoch in range(epoci_alese):
    pierdere_totala = 0

    for input_lr, target_hr in dataloader:
        input_lr = input_lr.to(device)
        target_hr = target_hr.to(device)

        # --- Magia de adaptare automată a dimensiunilor ---
        if alegere_model == "1":
            # SRCNN are nevoie ca intrarea și ieșirea să aibă ACEEAȘI dimensiune
            input_train = F.interpolate(input_lr, size=(256, 256), mode='bilinear', align_corners=False)
            target_train = F.interpolate(target_hr, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            # ESPCN are nevoie ca intrarea să fie de 3 ori MAI MICĂ
            input_train = F.interpolate(input_lr, size=(85, 85), mode='bilinear', align_corners=False)
            target_train = F.interpolate(target_hr, size=(255, 255), mode='bilinear', align_corners=False)

        # Antrenarea propriu-zisă
        optimizer.zero_grad()
        output = model(input_train)
        loss = criterion(output, target_train)
        loss.backward()
        optimizer.step()

        pierdere_totala += loss.item()

    if (epoch + 1) % 10 == 0 or (epoch + 1) == epoci_alese:
        print(f"📈 Epoca [{epoch + 1}/{epoci_alese}] | Eroare (Loss): {pierdere_totala / len(dataloader):.6f}")

# ==========================================
# 6. SALVAREA "CREIERULUI"
# ==========================================
torch.save(model.state_dict(), nume_salvare)
print(f"\n✅ Antrenament terminat cu succes! Fișierul '{nume_salvare}' a fost salvat și suprascris corect.")