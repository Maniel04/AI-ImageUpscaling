import os
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random

from PIL import Image

# ACTIVARE OPTIMIZARE CUDNN (Specific pentru NVIDIA)
torch.backends.cudnn.benchmark = True
# V2/V3: Optimizare viteză pură pentru arhitectura Blackwell (RTX 5060)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


# ==========================================
# 1. PREGĂTIREA DATELOR (DYNAMIC CROPPING)
# ==========================================
class BazaDatePoze(Dataset):
    def __init__(self, director_hr, crop_hr=510, crop_lr=170):
        self.hr_dir = director_hr
        self.nume_fisiere = [f for f in os.listdir(director_hr) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.crop_hr = crop_hr
        self.crop_lr = crop_lr

    def __len__(self):
        return len(self.nume_fisiere)

    def __getitem__(self, index):
        nume = self.nume_fisiere[index]
        img_hr_full = Image.open(os.path.join(self.hr_dir, nume)).convert('RGB')

        # Decupăm un cadru complet ALEATORIU din imaginea mare 2K
        transform_crop = transforms.RandomCrop(self.crop_hr)
        img_hr_crop = transform_crop(img_hr_full)

        # Generăm versiunea Low-Res din acel crop, folosind Bicubic
        transform_lr = transforms.Resize(self.crop_lr, interpolation=transforms.InterpolationMode.BICUBIC)
        img_lr_crop = transform_lr(img_hr_crop)

        # Transformăm în Tensori pentru placa video
        img_hr = transforms.ToTensor()(img_hr_crop)
        img_lr = transforms.ToTensor()(img_lr_crop)

        return img_lr, img_hr


# ==========================================
# 2. MODELE CLASICE (MSE/L1)
# ==========================================
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=3, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ESPCN(nn.Module):
    def __init__(self, factor_marire=3):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3 * (factor_marire ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(factor_marire)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x


# ==========================================
# 3. MODELE NOI (GAN)
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def bloc_conv(in_c, out_c, stride=1, bn=True):
            straturi = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                        nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                straturi.append(nn.BatchNorm2d(out_c))
            return nn.Sequential(*straturi)

        self.model = nn.Sequential(
            bloc_conv(3, 64, bn=False),
            bloc_conv(64, 64, stride=2),
            bloc_conv(64, 128),
            bloc_conv(128, 128, stride=2),
            bloc_conv(128, 256),
            bloc_conv(256, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.extractor = nn.Sequential(*list(vgg19.children())[:18])
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.extractor(x)


# ==========================================
# 4. LOGICA PRINCIPALĂ ȘI MENIUL
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Se folosește: {device} (Arhitectura detectată)\n")

    print("Alege modelul pentru antrenament:")
    print("1. SRCNN (Clasic)")
    print("2. ESPCN (Modern, rapid)")
    print("3. SRGAN (Realism Fotografic - GAN)")
    alegere = input("Introdu numarul (1/2/3): ")

    nume_folder = "DIV2K_train_HR"
    numar_epoci = int(input("Introdu numarul de epoci: "))

    # ==========================================
    # MODIFICARE: SETĂM NOUL FOLDER "poze_test_progres_V3_nou"
    # ==========================================


    ###AICI SCHIMBI FOLDERUL CA SA AI POZELE IN ALTUL


    nume_folder_salvare = "poze_test_progres_V3_nou"
    os.makedirs(nume_folder_salvare, exist_ok=True)

    cale_absoluta = os.path.abspath(nume_folder_salvare)
    print("\n" + "=" * 60)
    print("!!! ATENȚIE: FOLDERUL CU POZE SE AFLĂ EXACT AICI !!!")
    print(f"-> {cale_absoluta}")
    print("=" * 60 + "\n")

    try:
        os.startfile(cale_absoluta)  # Deschide automat folderul nou pe ecran
    except Exception as e:
        print(f"Nu am putut deschide folderul automat.")
    # ==========================================

    dataset = BazaDatePoze(
        director_hr=f"data/{nume_folder}",
        crop_hr=510,
        crop_lr=170
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    # ==========================================
    # 5. ANTRENAMENTUL
    # ==========================================
    if alegere in ['1', '2']:
        if alegere == '1':
            model = SRCNN().to(device)
            nume_fisier_salvare = "srcnn_model_antrenat_V3.pth"
        else:
            model = ESPCN().to(device)
            nume_fisier_salvare = "espcn_model_antrenat_V3.pth"

        if os.path.exists(nume_fisier_salvare):
            model.load_state_dict(torch.load(nume_fisier_salvare, map_location=device, weights_only=True))
            print(f"🔄 Progres găsit! Se continuă...")

        criteriu_pierdere = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(numar_epoci):
            model.train()
            for imagini_blurate, imagini_clare in dataloader:
                imagini_blurate = imagini_blurate.to(device, non_blocking=True)
                imagini_clare = imagini_clare.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    output = model(imagini_blurate)
                    loss = criteriu_pierdere(output, imagini_clare)

                scaler_G.scale(loss).backward()
                scaler_G.step(optimizer)
                scaler_G.update()

            print(f"Epoca [{epoch + 1}/{numar_epoci}], Eroare Medie: {loss.item():.4f}")
            torch.save(model.state_dict(), nume_fisier_salvare)

    elif alegere == '3':
        generator = GeneratorGAN(factor_marire=3).to(device)
        discriminator = Discriminator().to(device)
        extractor_vgg = FeatureExtractor().to(device)

        fisier_gen_vechi = "srgan_generator_V2.pth"
        fisier_disc_vechi = "srgan_discriminator_V2.pth"
        fisier_gen = "srgan_generator_V3.pth"
        fisier_disc = "srgan_discriminator_V3.pth"

        if os.path.exists(fisier_gen) and os.path.exists(fisier_disc):
            generator.load_state_dict(torch.load(fisier_gen, map_location=device, weights_only=True))
            discriminator.load_state_dict(torch.load(fisier_disc, map_location=device, weights_only=True))
            print("🔄 Progres GAN V3 găsit! Continuăm...")
        elif os.path.exists(fisier_gen_vechi) and os.path.exists(fisier_disc_vechi):
            generator.load_state_dict(torch.load(fisier_gen_vechi, map_location=device, weights_only=True))
            discriminator.load_state_dict(torch.load(fisier_disc_vechi, map_location=device, weights_only=True))
            print("🔄 Model V2 încărcat cu succes! Începem etapa V3...")

        criteriu_adversarial = nn.BCEWithLogitsLoss()
        criteriu_textura = nn.MSELoss()
        criteriu_pixel = nn.MSELoss()

        opt_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

        scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=50, gamma=0.5)

        for epoch in range(numar_epoci):
            generator.train()
            discriminator.train()

            for i, (imagini_blurate, imagini_clare) in enumerate(dataloader):
                imagini_blurate = imagini_blurate.to(device, non_blocking=True)
                imagini_clare = imagini_clare.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    poze_generate = generator(imagini_blurate)

                # --- ANTRENAMENT DISCRIMINATOR ---
                opt_D.zero_grad()
                with torch.amp.autocast('cuda'):
                    scor_reale = discriminator(imagini_clare)
                    loss_reale = criteriu_adversarial(scor_reale, torch.ones_like(scor_reale))

                    scor_false = discriminator(poze_generate.detach())
                    loss_false = criteriu_adversarial(scor_false, torch.zeros_like(scor_false))

                    loss_D = (loss_reale + loss_false) / 2

                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()

                # --- ANTRENAMENT GENERATOR ---
                opt_G.zero_grad()
                with torch.amp.autocast('cuda'):
                    scor_fals_nou = discriminator(poze_generate)
                    loss_G_adversarial = criteriu_adversarial(scor_fals_nou, torch.ones_like(scor_fals_nou))

                    texturi_generate = extractor_vgg(poze_generate)
                    texturi_reale = extractor_vgg(imagini_clare).detach()
                    loss_G_textura = criteriu_textura(texturi_generate, texturi_reale)

                    loss_G_pixel = criteriu_pixel(poze_generate, imagini_clare)

                    loss_mean = torch.abs(poze_generate.mean() - imagini_clare.mean())
                    loss_std = torch.abs(poze_generate.std() - imagini_clare.std())

                    loss_G_total = (loss_G_textura +
                                    0.001 * loss_G_adversarial +
                                    0.1 * loss_G_pixel +
                                    0.1 * loss_mean +
                                    0.1 * loss_std)

                scaler_G.scale(loss_G_total).backward()
                scaler_G.step(opt_G)
                scaler_G.update()

            print(
                f"Epoca [{epoch + 1}/{numar_epoci}] - Loss D: {loss_D.item():.4f} | Loss G: {loss_G_total.item():.4f}")

            torch.save(generator.state_dict(), fisier_gen)
            torch.save(discriminator.state_dict(), fisier_disc)

            scheduler_G.step()
            scheduler_D.step()

            generator.eval()
            with torch.no_grad():
                test_lr = imagini_blurate[0:1]
                test_hr = imagini_clare[0:1]
                test_gen = generator(test_lr)

                test_lr_resized = torch.nn.functional.interpolate(test_lr, size=test_hr.shape[2:], mode='bicubic')
                imagine_comparativa = torch.cat([test_lr_resized, test_gen, test_hr], dim=3)

                # SALVARE FIX: Folosește variabila dinamică pentru a salva în folderul corect
                save_image(imagine_comparativa, f"{nume_folder_salvare}/epoca_{epoch + 1}.png")
            generator.train()