import os
import requests
import zipfile
from tqdm import tqdm

# ==========================================
# SETĂRI PRINCIPALE - DATASET OFICIAL SR
# ==========================================
# Descărcăm setul DIV2K de antrenament (High Resolution - imagini 2K native)
URL_DATASET = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
FOLDER_DATA = "data"
FOLDER_DESTINATIE = os.path.join(FOLDER_DATA, "DIV2K_train_HR")
FISIER_ZIP = os.path.join(FOLDER_DATA, "DIV2K_train_HR.zip")

def descarca_div2k():
    """Descărcăm arhiva DIV2K dacă nu există deja."""
    os.makedirs(FOLDER_DATA, exist_ok=True)
    
    if os.path.exists(FISIER_ZIP):
        print(f"✅ Arhiva {FISIER_ZIP} există deja. Trecem direct la dezarhivare.")
    else:
        print(f"🚀 Începem descărcarea dataset-ului DIV2K (aprox. 3.5 GB)...")
        # Folosim stream=True pentru a descărca fișierul mare în bucăți fără să umplem memoria RAM
        raspuns = requests.get(URL_DATASET, stream=True)
        raspuns.raise_for_status()
        
        # Preluăm dimensiunea totală din headere pentru bara de progres
        dimensiune_totala = int(raspuns.headers.get('content-length', 0))
        
        with open(FISIER_ZIP, 'wb') as f, tqdm(
            desc="📥 Progres Descărcare",
            total=dimensiune_totala,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in raspuns.iter_content(chunk_size=8192):
                if chunk: # Filtrăm bucățile goale (keep-alive)
                    f.write(chunk)
                    bar.update(len(chunk))
                    
    dezarhiveaza_dataset()

def dezarhiveaza_dataset():
    """Extragem imaginile din arhiva ZIP."""
    print(f"\n📦 Se extrag imaginile High-Res...")
    
    with zipfile.ZipFile(FISIER_ZIP, 'r') as arhiva:
        fisiere = arhiva.namelist()
        for fisier in tqdm(fisiere, desc="📂 Progres Dezarhivare"):
            arhiva.extract(fisier, FOLDER_DATA)
            
    print(f"\n🎉 Proces finalizat! Imaginile de înaltă rezoluție se găsesc în '{FOLDER_DESTINATIE}'.")
    print("Sistemul tău este gata să hrănească placa RTX 5060 cu date de calitate academică!")

if __name__ == "__main__":
    descarca_div2k()