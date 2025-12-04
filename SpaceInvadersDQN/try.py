import torch
import torch_directml
import time

dml = torch_directml.device()

# Büyük matris
size = 3000

print("=== CPU ===")
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_t = time.time() - start
print(f"CPU süre: {cpu_t:.4f} sn\n")

print("=== GPU (DirectML) ===")
a_gpu = torch.randn(size, size, device=dml)
b_gpu = torch.randn(size, size, device=dml)
start = time.time()
c_gpu = a_gpu @ b_gpu
gpu_t = time.time() - start
print(f"GPU süre: {gpu_t:.4f} sn\n")

print(f"Hızlandırma oranı: {cpu_t / gpu_t:.2f}x")