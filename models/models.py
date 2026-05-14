import torch
import torch.nn as nn

# --- ALGORITMUL 1: SRCNN (Modelul Clasic) ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Stratul 1: Extracția trăsăturilor (9x9 kernel)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        # Stratul 2: Maparea neliniară (1x1 kernel)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # Stratul 3: Reconstrucția (5x5 kernel)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        # Funcția de activare (element cheie în Machine Learning)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ESPCN(nn.Module):
    def __init__(self, scale_factor=3):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU() # Folosim ReLU ca să nu mai iasă poza verde/grilaj

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x