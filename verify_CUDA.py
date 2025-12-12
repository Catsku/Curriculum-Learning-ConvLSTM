import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Deve retornar True
print(torch.cuda.get_device_name(0))  # Mostra o nome da GPU