import os
import requests


def descarca_poze(numar_poze=100, folder_destinatie="data/set_extins"):
    # 1. Creăm folderul dacă nu există
    if not os.path.exists(folder_destinatie):
        os.makedirs(folder_destinatie)
        print(f"📁 Folderul '{folder_destinatie}' a fost creat.")

    print(f"🚀 Începem descărcarea a {numar_poze} poze (rezoluție 960x960)...")

    # 2. Descărcăm pozele pe rând
    for i in range(1, numar_poze + 1):
        # Folosim 960x960 - dimensiunea perfect divizibilă cu 3 pentru ESPCN
        url = f"https://picsum.photos/960/960?random={i}"

        try:
            # Cerem poza de pe internet
            raspuns = requests.get(url, timeout=10)
            raspuns.raise_for_status()

            # O salvăm în folderul nostru
            nume_fisier = os.path.join(folder_destinatie, f"imagine_{i}.jpg")
            with open(nume_fisier, 'wb') as f:
                f.write(raspuns.content)

            print(f"✅ Descărcat: {nume_fisier}")

        except Exception as e:
            print(f"❌ Eroare la descărcarea pozei {i}: {e}")

    print("\n🎉 Gata! Baza ta de date este completă și pregătită.")


# Pornim funcția
if __name__ == "__main__":
    descarca_poze()