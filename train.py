import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import SRDataset
from models.models import SRCNN, ESPCN

print("=========================================")
print("🌙 Meniu Antrenament AI - NIGHT RIDER")
print("=========================================")

print("Ce model vrei să antrenezi peste noapte?")
print("1. SRCNN (Algoritmul Vechi)")
print("2. ESPCN (Algoritmul Modern)")
alegere_model = input("👉 Introdu 1 sau 2: ")

nume_folder = input("📁 Introdu numele folderului (ex: Set_Extins): ")
cale_date = f"data/{nume_folder}"
epoci_alese = int(input("⏳ Câte epoci vrei să ruleze? (Recomandat 15000 - 30000 pentru noapte): "))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️ Folosim: {device}")

dataset = SRDataset(cale_date)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

if alegere_model == "1":
    model = SRCNN().to(device)
    nume_salvare = "srcnn_model_antrenat.pth"
else:
    model = ESPCN(scale_factor=3).to(device)
    nume_salvare = "espcn_model_antrenat.pth"

# TRUCUL 1: Folosim L1 Loss pentru margini mai clare, în loc de MSE
criterion = nn.L1Loss()

# TRUCUL 2: Setăm Optimizer-ul și un Scheduler pentru frânare treptată
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Reduce viteza (LR) la jumătate (gamma=0.5) la fiecare 3000 de epoci
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

# TRUCUL 3: Variabilă pentru a urmări recordul absolut
cel_mai_bun_loss = float('inf')

model.train()
print(f"\n🚀 Start antrenament pe '{cale_date}'. Te poți duce la somn, modelul are grijă de el!")

for epoch in range(epoci_alese):
    pierdere_totala = 0

    for input_lr, target_hr in dataloader:
        input_lr = input_lr.to(device)
        target_hr = target_hr.to(device)

        if alegere_model == "1":
            input_train = F.interpolate(input_lr, size=(256, 256), mode='bilinear', align_corners=False)
            target_train = F.interpolate(target_hr, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            input_train = F.interpolate(input_lr, size=(85, 85), mode='bilinear', align_corners=False)
            target_train = F.interpolate(target_hr, size=(255, 255), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        output = model(input_train)
        loss = criterion(output, target_train)
        loss.backward()
        optimizer.step()

        pierdere_totala += loss.item()

    # Anunțăm scheduler-ul că a mai trecut o epocă
    scheduler.step()

    pierdere_medie = pierdere_totala / len(dataloader)

    # Afișăm în consolă progresul la fiecare 50 de epoci
    if (epoch + 1) % 50 == 0:
        viteza_curenta = scheduler.get_last_lr()[0]
        print(
            f"📈 Epoca [{epoch + 1}/{epoci_alese}] | Eroare (L1): {pierdere_medie:.6f} | Viteză LR: {viteza_curenta:.6f}")

    # SALVAREA INTELIGENTĂ (Checkpointing)
    if pierdere_medie < cel_mai_bun_loss:
        cel_mai_bun_loss = pierdere_medie
        torch.save(model.state_dict(), nume_salvare)
        # Afișăm un mesaj discret doar din 100 în 100 de epoci pentru a nu face spam pe ecran,
        # dar salvarea se face fizic la fiecare record.
        if (epoch + 1) % 100 == 0:
            print(f"   🌟 Nou record de claritate! Model salvat în siguranță.")

print("\n✅ Antrenamentul de cursă lungă s-a terminat! Te așteaptă cel mai bun model posibil.")