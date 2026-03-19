import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Create a randomly initialized tensor
x = torch.rand(5, 3)
print("Random 5x3 Tensor:")
print(x)
