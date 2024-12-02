import torch
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the appropriate device (GPU or CPU)
model_1.to(device)

# Create your input tensor (example shape: batch_size=1, channels=3, height=224, width=224)
inputs = torch.randn(1, 3, 224, 224)

# Move the input tensor to the same device as the model
inputs = inputs.to(device)

# Now run the model
y = model_1(inputs)
