import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image


# ==========================================
# 1. PREGĂTIREA DATELOR (DATASET)
# ==========================================
class BazaDatePoze(Dataset):
    def __init__(self, director_poze, crop_size=512):
        self.director = director_poze
        self.nume_fisiere = os.listdir(director_poze)

        # MATEMATICA SALVATOARE: Forțăm dimensiunea să fie perfect divizibilă cu 3
        # Dacă îi dăm 512, variabila de mai jos devine automat 510.
        dimensiune_perfecta = (crop_size // 3) * 3

        self.transformare_clara = transforms.Compose([
            transforms.CenterCrop(dimensiune_perfecta),
            transforms.ToTensor()
        ])

        self.transformare_blurata = transforms.Compose([
            transforms.CenterCrop(dimensiune_perfecta),
            transforms.Resize(dimensiune_perfecta // 3, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.nume_fisiere)

    def __getitem__(self, index):
        cale_poza = os.path.join(self.director, self.nume_fisiere[index])
        poza_originala = Image.open(cale_poza).convert('RGB')

        img_hr = self.transformare_clara(poza_originala)
        img_lr = self.transformare_blurata(poza_originala)

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

    def forward(self, x):
        return self.extractor(x)


# ==========================================
# 4. LOGICA PRINCIPALĂ ȘI MENIUL
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Se folosește: {device}\n")

    print("Alege modelul pentru antrenament:")
    print("1. SRCNN (Clasic)")
    print("2. ESPCN (Modern, rapid)")
    print("3. SRGAN (Realism Fotografic - GAN)")
    alegere = input("Introdu numarul (1/2/3): ")

    nume_folder = input("Introdu numele folderului cu date (ex: Set_1000): ")
    numar_epoci = int(input("Introdu numarul de epoci: "))

    # Inițializăm datele
    dataset = BazaDatePoze(f"data/{nume_folder}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # ==========================================
    # 5. ANTRENAMENTUL CU "RESUME"
    # ==========================================
    if alegere in ['1', '2']:
        if alegere == '1':
            model = SRCNN().to(device)
            nume_fisier_salvare = "srcnn_model_antrenat.pth"
        else:
            model = ESPCN().to(device)
            nume_fisier_salvare = "espcn_model_antrenat.pth"

        # --- LOGICA DE RESUME PENTRU MODELE CLASICE ---
        if os.path.exists(nume_fisier_salvare):
            model.load_state_dict(torch.load(nume_fisier_salvare, map_location=device, weights_only=True))
            print(f"🔄 Progres găsit! Am încărcat modelul {nume_fisier_salvare}. Se continuă antrenamentul...")
        else:
            print("🚀 Nu s-a găsit progres anterior. Se începe un antrenament de la zero.")

        criteriu_pierdere = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        cea_mai_mica_eroare = float('inf')

        for epoch in range(numar_epoci):
            model.train()
            eroare_totala = 0.0

            for imagini_blurate, imagini_clare in dataloader:
                imagini_blurate = imagini_blurate.to(device)
                imagini_clare = imagini_clare.to(device)

                optimizer.zero_grad()
                output = model(imagini_blurate)
                loss = criteriu_pierdere(output, imagini_clare)
                loss.backward()
                optimizer.step()

                eroare_totala += loss.item()

            scheduler.step()
            eroare_medie = eroare_totala / len(dataloader)
            print(f"Epoca [{epoch + 1}/{numar_epoci}], Eroare Medie: {eroare_medie:.4f}")

            if eroare_medie < cea_mai_mica_eroare:
                cea_mai_mica_eroare = eroare_medie
                torch.save(model.state_dict(), nume_fisier_salvare)

    elif alegere == '3':
        generator = GeneratorGAN().to(device)
        discriminator = Discriminator().to(device)
        extractor_vgg = FeatureExtractor().to(device)

        # --- LOGICA DE RESUME PENTRU GAN ---
        fisier_gen = "srgan_generator_antrenat.pth"
        fisier_disc = "srgan_discriminator_antrenat.pth"

        if os.path.exists(fisier_gen) and os.path.exists(fisier_disc):
            generator.load_state_dict(torch.load(fisier_gen, map_location=device, weights_only=True))
            discriminator.load_state_dict(torch.load(fisier_disc, map_location=device, weights_only=True))
            print("🔄 Progres GAN găsit! Am încărcat ambele rețele. Meciul continuă...")
        else:
            print("🚀 Nu s-a găsit progres complet anterior pentru GAN. Se începe de la zero.")

        criteriu_adversarial = nn.BCEWithLogitsLoss()
        criteriu_textura = nn.MSELoss()

        opt_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
        opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

        for epoch in range(numar_epoci):
            generator.train()
            discriminator.train()

            for imagini_blurate, imagini_clare in dataloader:
                imagini_blurate = imagini_blurate.to(device)
                imagini_clare = imagini_clare.to(device)

                # --- ANTRENAMENT DISCRIMINATOR ---
                opt_D.zero_grad()
                poze_generate = generator(imagini_blurate)

                scor_reale = discriminator(imagini_clare)
                loss_reale = criteriu_adversarial(scor_reale, torch.ones_like(scor_reale))

                scor_false = discriminator(poze_generate.detach())
                loss_false = criteriu_adversarial(scor_false, torch.zeros_like(scor_false))

                loss_D = (loss_reale + loss_false) / 2
                loss_D.backward()
                opt_D.step()

                # --- ANTRENAMENT GENERATOR ---
                opt_G.zero_grad()
                scor_fals_nou = discriminator(poze_generate)
                loss_G_adversarial = criteriu_adversarial(scor_fals_nou, torch.ones_like(scor_fals_nou))

                texturi_generate = extractor_vgg(poze_generate)
                texturi_reale = extractor_vgg(imagini_clare).detach()
                loss_G_textura = criteriu_textura(texturi_generate, texturi_reale)

                loss_G_total = loss_G_textura + 0.001 * loss_G_adversarial
                loss_G_total.backward()
                opt_G.step()

            # Afișare și Salvare pentru GAN la final de epocă
            print(
                f"Epoca [{epoch + 1}/{numar_epoci}] - Loss D: {loss_D.item():.4f} | Loss G: {loss_G_total.item():.4f}")

            # Salvăm AMBELE rețele la finalul fiecărei epoci
            torch.save(generator.state_dict(), fisier_gen)
            torch.save(discriminator.state_dict(), fisier_disc)