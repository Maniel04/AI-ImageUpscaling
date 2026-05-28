import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.models import SRCNN, ESPCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ARHITECTURA SRGAN (Generatorul)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, canale=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(canale, canale, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(canale)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(canale, canale, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(canale)

    def forward(self, x):
        rezidual = self.conv1(x)
        rezidual = self.bn1(rezidual)
        rezidual = self.prelu(rezidual)
        rezidual = self.conv2(rezidual)
        rezidual = self.bn2(rezidual)
        return x + rezidual

class GeneratorGAN(nn.Module):
    def __init__(self, factor_marire=3):
        super(GeneratorGAN, self).__init__()
        self.conv_initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.blocuri_reziduale = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.conv_mijloc = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (factor_marire ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(factor_marire),
            nn.PReLU()
        )
        self.conv_final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out_initial = self.conv_initial(x)
        out = self.blocuri_reziduale(out_initial)
        out = self.conv_mijloc(out)
        out = out + out_initial
        out = self.upsample(out)
        out = self.conv_final(out)
        return (torch.tanh(out) + 1) / 2

# ==========================================
# 2. ÎNCĂRCĂM TOATE MODELELE (SRCNN, ESPCN + 3x SRGAN)
# ==========================================
# SRCNN & ESPCN
model_srcnn = SRCNN().to(device)
model_srcnn.load_state_dict(torch.load("srcnn_model_antrenat_V3.pth", weights_only=True))
model_srcnn.eval()

model_espcn = ESPCN(scale_factor=3).to(device)
model_espcn.load_state_dict(torch.load("espcn_model_antrenat_V3.pth", weights_only=True))
model_espcn.eval()

# VARIANTĂ 1: SRGAN Antrenat pe 100 poze (Basic)
model_srgan_v1 = GeneratorGAN(factor_marire=3).to(device)
model_srgan_v1.load_state_dict(torch.load("srgan_generator_antrenat.pth", weights_only=True))
model_srgan_v1.eval()

# VARIANTĂ 2: SRGAN Antrenat pe 2000 poze (Basic)
model_srgan_v2 = GeneratorGAN(factor_marire=3).to(device)
model_srgan_v2.load_state_dict(torch.load("srgan_generator_V2.pth", weights_only=True))
model_srgan_v2.eval()

# VARIANTĂ 3: SRGAN Antrenat pe 2000 poze (Cu Filtru Culoare)
model_srgan_v3 = GeneratorGAN(factor_marire=3).to(device)
model_srgan_v3.load_state_dict(torch.load("srgan_generator_V3.pth", weights_only=True))
model_srgan_v3.eval()

# ==========================================
# 3. PREGĂTIM IMAGINEA PENTRU REȚELE
# ==========================================
imagine_cale = 'data/Set5/img_002_SRF_2_HR.png'
try:
    imagine_hr = Image.open(imagine_cale).convert('RGB')
except FileNotFoundError:
    print(f"⚠️ Nu am găsit imaginea la calea: {imagine_cale}")
    exit()

w, h = imagine_hr.size
w, h = (w // 3) * 3, (h // 3) * 3
imagine_hr = imagine_hr.resize((w, h), Image.BICUBIC)

transform_low_res = transforms.Resize((h // 3, w // 3), Image.BICUBIC)
imagine_lr_mica = transform_low_res(imagine_hr)
imagine_lr_mare = imagine_lr_mica.resize((w, h), Image.BICUBIC)

tensor_mic = transforms.ToTensor()(imagine_lr_mica).unsqueeze(0).to(device)
tensor_mare = transforms.ToTensor()(imagine_lr_mare).unsqueeze(0).to(device)

# ==========================================
# 4. TRECEM POZA PRIN TOATE REȚELELE
# ==========================================
print("⏳ AI-urile procesează imaginea...")
with torch.no_grad():
    output_srcnn = model_srcnn(tensor_mare)
    output_espcn = model_espcn(tensor_mic)
    # Cele 3 variante de SRGAN
    out_v1 = model_srgan_v1(tensor_mic)
    out_v2 = model_srgan_v2(tensor_mic)
    out_v3 = model_srgan_v3(tensor_mic)

# Transformăm înapoi în poze
poza_srcnn = transforms.ToPILImage()(output_srcnn.squeeze(0).cpu().clamp(0, 1))
poza_espcn = transforms.ToPILImage()(output_espcn.squeeze(0).cpu().clamp(0, 1))
poza_v1 = transforms.ToPILImage()(out_v1.squeeze(0).cpu().clamp(0, 1))
poza_v2 = transforms.ToPILImage()(out_v2.squeeze(0).cpu().clamp(0, 1))
poza_v3 = transforms.ToPILImage()(out_v3.squeeze(0).cpu().clamp(0, 1))

# ==========================================
# 5. AFIȘĂM COMPARAȚIA FINALĂ (7 Coloane)
# ==========================================
fig, ax = plt.subplots(1, 7, figsize=(30, 6))

ax[0].imshow(imagine_lr_mare)
ax[0].set_title("1. Input (Blurat)")
ax[0].axis('off')

ax[1].imshow(poza_srcnn)
ax[1].set_title("2. SRCNN (V3)")
ax[1].axis('off')

ax[2].imshow(poza_espcn)
ax[2].set_title("3. ESPCN (V3)")
ax[2].axis('off')

ax[3].imshow(poza_v1)
ax[3].set_title("4. SRGAN (100 poze)")
ax[3].axis('off')

ax[4].imshow(poza_v2)
ax[4].set_title("5. SRGAN (2000 poze)")
ax[4].axis('off')

ax[5].imshow(poza_v3)
ax[5].set_title("6. SRGAN (+ Filtru)")
ax[5].axis('off')

ax[6].imshow(imagine_hr)
ax[6].set_title("7. Poza Originală")
ax[6].axis('off')

plt.tight_layout()
print("✅ Gata! Vizualizează evoluția SRGAN în fereastra deschisă.")
plt.show()