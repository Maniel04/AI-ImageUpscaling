import torch

print(f"Versiune PyTorch: {torch.__version__}")
print(f"CUDA disponibil: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"Placa Video: {device_name}")
    print(f"Compute Capability detectat: sm_{capability[0]}{capability[1]}")

    if capability >= (12, 0):
        print(">>> CONFIGURAȚIE CORECTĂ: Placa ta Blackwell este suportată nativ!")
    else:
        print(">>> ATENȚIE: Arhitectura detectată este mai veche de sm_120.")
else:
    print(">>> EROARE: CUDA nu este detectat. Verifică driverul NVIDIA.")