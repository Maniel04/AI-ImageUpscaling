import os
import requests
import concurrent.futures
from tqdm import tqdm

# ==========================================
# SETĂRI PRINCIPALE
# ==========================================
NUMAR_TOTAL_POZE = 2500
REZOLUTIE = 1920 # 1920x1920 este perfect pentru decupaje de 512x512
FOLDER_DESTINATIE = "data/Set_SRGAN_2K"
FIRE_DE_EXECUTIE = 16 # Câte poze să descarce simultan

def descarca_o_poza(index):
    """Funcție care descarcă o singură imagine."""
    url = f"https://picsum.photos/{REZOLUTIE}/{REZOLUTIE}?random={index}"
    nume_fisier = os.path.join(FOLDER_DESTINATIE, f"imagine_{index}.jpg")

    # Sistem de "Resume": Dacă poza există deja, trecem mai departe rapid
    if os.path.exists(nume_fisier):
        return True

    try:
        # Cerem poza (timeout de 15 secunde pentru a evita blocajele)
        raspuns = requests.get(url, timeout=15)
        raspuns.raise_for_status()

        with open(nume_fisier, 'wb') as f:
            f.write(raspuns.content)
        return True
    except Exception:
        # Dacă pică internetul sau serverul dă eroare, ignorăm în liniște
        return False

def porneste_descarcarea():
    if not os.path.exists(FOLDER_DESTINATIE):
        os.makedirs(FOLDER_DESTINATIE)
        print(f"📁 Folder creat: '{FOLDER_DESTINATIE}'")

    print(f"🚀 Începem descărcarea a {NUMAR_TOTAL_POZE} imagini la rezoluția {REZOLUTIE}x{REZOLUTIE}...")
    print(f"⚡ Se folosesc {FIRE_DE_EXECUTIE} conexiuni paralele pentru viteză maximă.\n")

    poze_descarcate_cu_succes = 0

    # Folosim ThreadPoolExecutor pentru a descărca în paralel
    with concurrent.futures.ThreadPoolExecutor(max_workers=FIRE_DE_EXECUTIE) as executor:
        # Trimitem toate "comenzile" de descărcare către firele de execuție
        viitoare = [executor.submit(descarca_o_poza, i) for i in range(1, NUMAR_TOTAL_POZE + 1)]

        # tqdm afișează bara de progres actualizată în timp real
        for viitor in tqdm(concurrent.futures.as_completed(viitoare), total=NUMAR_TOTAL_POZE, desc="📥 Progres Descărcare"):
            if viitor.result(): # Dacă funcția a returnat True
                poze_descarcate_cu_succes += 1

    print(f"\n🎉 Proces finalizat! S-au asigurat {poze_descarcate_cu_succes} / {NUMAR_TOTAL_POZE} imagini.")
    print("Sistemul tău este gata să hrănească placa RTX 5060 cu date de calitate!")

if __name__ == "__main__":
    porneste_descarcarea()