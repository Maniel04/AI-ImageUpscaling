import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.models import SRCNN

# 1. Găsim placa video (1660 Ti-ul tău)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Încărcăm modelul și "creierul" salvat
model = SRCNN().to(device)
model.load_state_dict(torch.load("srcnn_model_antrenat.pth", weights_only=True))
model.eval() # Punem AI-ul în modul de "Testare" (nu mai învață, doar aplică)

# 3. Încărcăm o poză de test (asigură-te că pui calea corectă către o poză din folderul tău)
# Am pus 'bird.png' ca exemplu, dar poți pune orice poză ai în Set5
imagine_cale = 'data/Set5/img_002_SRF_2_HR.png'
imagine_hr = Image.open(imagine_cale).convert('RGB')

# Creăm intenționat o versiune blurată a pozei (pentru a-i da AI-ului ceva de reparat)
transform_low_res = transforms.Compose([
    transforms.Resize((imagine_hr.size[1] // 3, imagine_hr.size[0] // 3), Image.BICUBIC),
    transforms.Resize((imagine_hr.size[1], imagine_hr.size[0]), Image.BICUBIC)
])
imagine_lr = transform_low_res(imagine_hr)

# 4. Transformăm poza în numere (tensori) ca să o înțeleagă AI-ul
transform_tensor = transforms.ToTensor()
input_tensor = transform_tensor(imagine_lr).unsqueeze(0).to(device)

# 5. Trecem poza prin rețeaua antrenată
print("⏳ AI-ul procesează imaginea...")
with torch.no_grad(): # Îi spunem să nu mai calculeze erori, doar să genereze poza
    output_tensor = model(input_tensor)

# Transformăm numerele înapoi în imagine normală
output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu().clamp(0, 1))

# 6. Afișăm rezultatul final
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(imagine_lr)
ax[0].set_title("1. Input (Poză Blurată)")
ax[0].axis('off')

ax[1].imshow(output_image)
ax[1].set_title("2. Rezultat AI (SRCNN)")
ax[1].axis('off')

ax[2].imshow(imagine_hr)
ax[2].set_title("3. Target (Poza Originală)")
ax[2].axis('off')

print("✅ Gata! Se deschide fereastra cu rezultatele.")
plt.show()