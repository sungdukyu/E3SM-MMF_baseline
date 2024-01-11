import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Define the CustomMLP model
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size1, output_size2):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
        self.fc6a = nn.Linear(hidden_sizes[4], output_size1)
        self.fc6b = nn.Linear(hidden_sizes[4], output_size2)
        self.leaky_relu = nn.LeakyReLU(0.15)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        out_a = self.fc6a(x)  # No activation for out_a
        out_b = F.relu(self.fc6b(x))  # ReLU activation for out_b
        out_combined = torch.cat((out_a, out_b), dim=1)
        return out_combined

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training and validation data
# (file paths on PSC Bridges-2 HPC)
x_train = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/train_input.npy')
y_train = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/train_target.npy')
x_val = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/val_input.npy')
y_val = np.load('/ocean/projects/atm200007p/jlin96/neurips_proj/e3sm_train_npy/val_target.npy')

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=3072, shuffle=True)
val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=3072, shuffle=False)

# Define model, optimizer, and loss function
model = CustomMLP(124, [768, 640, 512, 640, 640], 120, 8).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
counter = 0

# To store losses for each epoch
training_losses = []
validation_losses = []

# Training loop with early stopping
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)
    training_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    val_loss /= len(val_loader)
    validation_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

# Training completed

