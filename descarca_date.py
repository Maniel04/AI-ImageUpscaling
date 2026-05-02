import os
import requests


def descarca_poze(numar_poze=1000, folder_destinatie="data/Set_1000"):
    # Creăm folderul dacă nu există
    if not os.path.exists(folder_destinatie):
        os.makedirs(folder_destinatie)
        print(f"📁 Folderul '{folder_destinatie}' a fost creat.")

    print(f"🚀 Începem descărcarea a {numar_poze} poze (rezoluție 960x960)...")
    print("Așteaptă, procesul va dura câteva minute în funcție de viteza de internet.\n")

    # Descărcăm pozele pe rând
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

            # Afișăm progresul din 50 în 50 de poze pentru a nu bloca terminalul
            if i % 50 == 0 or i == numar_poze:
                print(f"✅ Descărcate: {i} / {numar_poze} poze")

        except Exception as e:
            print(f"❌ Eroare la descărcarea pozei {i}: {e}")

    print("\n🎉 Gata! Baza ta de date uriașă este completă și pregătită pentru noaptea asta.")


# Pornim funcția
if __name__ == "__main__":
    descarca_poze()