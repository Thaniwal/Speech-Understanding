import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")

# Set dataset paths
dataset_path = "/home/rl_gaming/hari/UrbanSound8K/UrbanSound8K"
metadata_path = os.path.join(dataset_path, "metadata/UrbanSound8K.csv")
audio_folder = os.path.join(dataset_path, "audio")
plots_folder = os.path.join(dataset_path, "plots")
os.makedirs(plots_folder, exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_path)

# Define fixed spectrogram size
fixed_height = 1025
fixed_width = 431

# Function to extract features using STFT with different window types
def extract_features(file_path, window_type='hann'):
    y, sr = librosa.load(file_path, sr=None)

    if window_type == 'hann':
        window = np.hanning(2048)
    elif window_type == 'hamming':
        window = np.hamming(2048)
    else:
        window = np.ones(2048)

    stft = librosa.stft(y, n_fft=2048, hop_length=1024, window=window)
    S = np.abs(stft)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    if S_db.shape[1] < fixed_width:
        pad_amount = fixed_width - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_amount)), mode='constant')
    else:
        S_db = S_db[:, :fixed_width]

    return torch.tensor(S_db, dtype=torch.float32).contiguous(), S_db, sr  # Ensure tensor is contiguous

# Visualizing spectrograms for a sample file
sample_file = os.path.join(audio_folder, "fold1/101415-3-0-2.wav")
window_types = ['hann', 'hamming', 'rectangular']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

for i, window in enumerate(window_types):
    _, spectrogram, sr = extract_features(sample_file, window_type=window)
    ax = axes[i]
    img = librosa.display.specshow(spectrogram, sr=sr, hop_length=1024, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(f"{window.capitalize()} Window")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout()
plt.savefig(os.path.join(plots_folder, "sample_spectrograms.png"))
plt.show()

# Define the neural network
class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        self.fc1 = nn.Linear(fixed_height * fixed_width, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.output = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.output(self.fc2(x))
        return x

# Prepare data
features, labels = [], []

for index, row in metadata.iterrows():
    file_path = os.path.join(audio_folder, f"fold{row['fold']}/{row['slice_file_name']}")
    
    try:
        feature_tensor, _, _ = extract_features(file_path)
        feature_tensor = feature_tensor.reshape(-1)  # Use reshape instead of view
        features.append(feature_tensor)
        labels.append(row['classID'])
    except Exception as e:
        print(f"Skipping file {file_path} due to error: {e}")

# Convert list to tensor
features = torch.stack(features).to(device)
labels = torch.tensor(labels, dtype=torch.long).to(device)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features.cpu().numpy(), labels.cpu().numpy(), test_size=0.2, random_state=42)

# Convert back to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# Initialize model, loss, and optimizer
model = AudioNet().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on test data: {accuracy * 100:.2f}%')

# Save model
torch.save(model.state_dict(), os.path.join(dataset_path, "audio_classification_model.pth"))
