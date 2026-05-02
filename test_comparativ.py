import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.models import SRCNN, ESPCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ÎNCĂRCĂM AMBELE MODELE
# ==========================================
# Încărcăm SRCNN
model_srcnn = SRCNN().to(device)
model_srcnn.load_state_dict(torch.load("srcnn_model_antrenat.pth", weights_only=True))
model_srcnn.eval()

# Încărcăm ESPCN
model_espcn = ESPCN(scale_factor=3).to(device)
model_espcn.load_state_dict(torch.load("espcn_model_antrenat.pth", weights_only=True))
model_espcn.eval()

# ==========================================
# 2. PREGĂTIM IMAGINEA PENTRU AMBELE
# ==========================================
imagine_cale = 'data/Set5/img_002_SRF_2_HR.png'
imagine_hr = Image.open(imagine_cale).convert('RGB')

# Ajustăm dimensiunea pentru a fi divizibilă cu 3 (cerința ESPCN)
w, h = imagine_hr.size
w, h = (w // 3) * 3, (h // 3) * 3
imagine_hr = imagine_hr.resize((w, h), Image.BICUBIC)

# Versiunea MICĂ (pentru intrarea în ESPCN)
transform_low_res = transforms.Resize((h // 3, w // 3), Image.BICUBIC)
imagine_lr_mica = transform_low_res(imagine_hr)

# Versiunea MARE, dar blurată (pentru intrarea în SRCNN și afișare)
imagine_lr_mare = imagine_lr_mica.resize((w, h), Image.BICUBIC)

# Transformăm în numere pentru placa video
tensor_mic = transforms.ToTensor()(imagine_lr_mica).unsqueeze(0).to(device)
tensor_mare = transforms.ToTensor()(imagine_lr_mare).unsqueeze(0).to(device)

# ==========================================
# 3. Trecem poza prin ambele rețele
# ==========================================
print("⏳ AI-urile procesează imaginea...")
with torch.no_grad():
    output_srcnn = model_srcnn(tensor_mare)
    output_espcn = model_espcn(tensor_mic)

# Transformăm înapoi în poze
poza_srcnn = transforms.ToPILImage()(output_srcnn.squeeze(0).cpu().clamp(0, 1))
poza_espcn = transforms.ToPILImage()(output_espcn.squeeze(0).cpu().clamp(0, 1))

# ==========================================
# 4. AFIȘĂM COMPARAȚIA FINALĂ
# ==========================================
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

ax[0].imshow(imagine_lr_mare)
ax[0].set_title("1. Input (Blurat)")
ax[0].axis('off')

ax[1].imshow(poza_srcnn)
ax[1].set_title("2. SRCNN (Algoritm Vechi)")
ax[1].axis('off')

ax[2].imshow(poza_espcn)
ax[2].set_title("3. ESPCN (Algoritm Modern)")
ax[2].axis('off')

ax[3].imshow(imagine_hr)
ax[3].set_title("4. Poza Originală (Ținta)")
ax[3].axis('off')

plt.tight_layout()
print("✅ Gata! Se deschide fereastra cu comparația.")
plt.show()