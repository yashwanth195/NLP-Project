import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Get the current device
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
    print(f"Device ID: {current_device}")
else:
    print("CUDA is not available.")
