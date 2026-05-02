import os
import cv2
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, folder_imagini, factor_marire=3):
        # Listăm toate fișierele din folderul Set5
        self.nume_fisiere = [os.path.join(folder_imagini, f) for f in os.listdir(folder_imagini)]
        self.factor = factor_marire

    def __len__(self):
        return len(self.nume_fisiere)

    def __getitem__(self, index):
        # 1. Citim imaginea originală (High-Res) - Aceasta este "Eticheta" (Target)
        imagine_hr = cv2.imread(self.nume_fisiere[index])
        imagine_hr = cv2.cvtColor(imagine_hr, cv2.COLOR_BGR2RGB)

        # 2. Creăm imaginea Low-Res (Input) prin micșorare
        h, w, _ = imagine_hr.shape
        h_mic, w_mic = h // self.factor, w // self.factor
        imagine_lr = cv2.resize(imagine_hr, (w_mic, h_mic), interpolation=cv2.INTER_CUBIC)

        # Pentru modelul SRCNN, intrarea trebuie să fie mărită la loc (dar blurată)
        imagine_lr_srcnn = cv2.resize(imagine_lr, (w, h), interpolation=cv2.INTER_CUBIC)

        # 3. Transformăm în Tensori (formatul cerut de Deep Learning)
        # Permutăm din (H, W, C) în (C, H, W) și normalizăm la [0, 1]
        hr_tensor = torch.from_numpy(imagine_hr).permute(2, 0, 1).float() / 255.0
        lr_tensor = torch.from_numpy(imagine_lr_srcnn).permute(2, 0, 1).float() / 255.0

        return lr_tensor, hr_tensor