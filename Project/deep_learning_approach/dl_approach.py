import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import librosa
import random
from scipy.signal import butter, lfilter
import torch.nn.functional as F
from sklearn.manifold import TSNE


###########################
# Data Loading and Preprocessing
###########################

class GTZANDataset(Dataset):
    """
    Dataset class for loading and preprocessing audio files from the GTZAN dataset.
    """
    def __init__(self, data_dir, sample_rate=22050, duration=3, noise_level=None, transforms=None):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.noise_level = noise_level
        self.transforms = transforms
        
        self.genres = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.file_paths = []
        self.labels = []
        
        for i, genre in enumerate(self.genres):
            genre_dir = os.path.join(data_dir, genre)
            if os.path.isdir(genre_dir):
                for file in os.listdir(genre_dir):
                    if file.endswith('.wav'):
                        self.file_paths.append(os.path.join(genre_dir, file))
                        self.labels.append(i)
        
        self.label_to_genre = {i: genre for i, genre in enumerate(self.genres)}
        self.genre_to_label = {genre: i for i, genre in enumerate(self.genres)}
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Trim to fixed duration
        num_samples = int(self.sample_rate * self.duration)
        if waveform.shape[1] > num_samples:
            # Randomly select a segment
            start = np.random.randint(0, waveform.shape[1] - num_samples)
            waveform = waveform[:, start:start+num_samples]
        else:
            # Pad if necessary
            waveform = F.pad(waveform, (0, num_samples - waveform.shape[1]))
        
        # Apply noise if specified
        if self.noise_level is not None:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # Apply transforms if specified
        if self.transforms:
            waveform = self.transforms(waveform)
        
        return waveform, label, file_path


###########################
# Feature Extraction
###########################

def extract_melspectrogram(waveform, sample_rate=22050, n_fft=1024, hop_length=512, n_mels=64):
    """
    Extract Mel-spectrogram from a waveform.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to power spectrogram
    melspec = mel_spectrogram(waveform)
    
    # Convert to log scale
    melspec = torch.log(melspec + 1e-9)
    
    return melspec


###########################
# Noise Handling Functions
###########################

def butter_lowpass(cutoff, fs, order=5):
    """Create a butterworth lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Apply a lowpass filter to the data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def spectral_gating(waveform, sample_rate=22050, threshold=3, n_fft=2048, hop_length=512):
    """
    Perform spectral gating for noise reduction (similar to Audacity's noise reduction).
    """
    # Convert to numpy for processing
    signal = waveform.numpy()[0]
    
    # Compute STFT
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    
    # Estimate noise profile (assuming first 0.5 seconds is noise or silent)
    noise_frames = int(0.5 * sample_rate / hop_length)
    noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Compute gain based on SNR
    snr = magnitude / (noise_magnitude + 1e-10)
    gain = (snr - threshold).clip(0, 1) / (snr + 1e-10)
    
    # Apply gain to the magnitude
    magnitude_filtered = magnitude * gain
    
    # Reconstruct signal
    stft_filtered = magnitude_filtered * phase
    signal_filtered = librosa.istft(stft_filtered, hop_length=hop_length)
    
    # Convert back to torch tensor
    return torch.from_numpy(signal_filtered).unsqueeze(0).float()

def spectral_subtraction(signal, noise_profile, alpha=2):
    """
    Perform spectral subtraction for noise reduction.
    """
    # Compute power spectrum of the signal
    signal_stft = torch.stft(signal[0], n_fft=1024, hop_length=512, window=torch.hann_window(1024), return_complex=True)
    signal_power = torch.abs(signal_stft) ** 2
    
    # Compute power spectrum of the noise profile
    noise_stft = torch.stft(noise_profile[0], n_fft=1024, hop_length=512, window=torch.hann_window(1024), return_complex=True)
    noise_power = torch.abs(noise_stft) ** 2
    
    # Subtract noise power from signal power
    subtracted_power = torch.maximum(signal_power - alpha * noise_power, torch.tensor(0.0))
    
    # Reconstruct the signal
    phase = torch.angle(signal_stft)
    reconstructed_stft = torch.polar(torch.sqrt(subtracted_power), phase)
    
    # Inverse STFT
    reconstructed_signal = torch.istft(reconstructed_stft, n_fft=1024, hop_length=512, window=torch.hann_window(1024))
    
    return reconstructed_signal.unsqueeze(0)

def add_noise(waveform, noise_type, snr):
    """
    Add noise to a waveform at a specified SNR level.
    """
    signal_power = torch.mean(waveform ** 2)
    
    # Create initial noise with same shape as waveform
    base_noise = torch.randn_like(waveform)
    
    if noise_type == 'white':
        noise = base_noise
    elif noise_type == 'pink':
        try:
            # Approximate pink noise
            noise_stft = torch.stft(base_noise[0], n_fft=1024, hop_length=512, window=torch.hann_window(1024), return_complex=True)
            freqs = torch.fft.rfftfreq(1024, 1/22050)
            filter_coeff = 1 / torch.sqrt(freqs + 1e-9)
            filtered_stft = noise_stft * filter_coeff.unsqueeze(1)
            processed_noise = torch.istft(filtered_stft, n_fft=1024, hop_length=512, window=torch.hann_window(1024))
            
            # Resize to match waveform
            if len(processed_noise) > waveform.shape[1]:
                processed_noise = processed_noise[:waveform.shape[1]]
            else:
                processed_noise = F.pad(processed_noise, (0, waveform.shape[1] - len(processed_noise)))
            
            noise = processed_noise.unsqueeze(0).float()
        except Exception as e:
            print(f"Error in pink noise generation: {e}. Using white noise instead.")
            noise = base_noise
    elif noise_type == 'street':
        try:
            # Simulate street noise (this is a simplification)
            processed_noise = butter_lowpass_filter(base_noise.numpy()[0], cutoff=2000, fs=22050)
            
            # Resize to match waveform
            if len(processed_noise) > waveform.shape[1]:
                processed_noise = processed_noise[:waveform.shape[1]]
            else:
                processed_noise = np.pad(processed_noise, (0, waveform.shape[1] - len(processed_noise)))
            
            noise = torch.from_numpy(processed_noise).unsqueeze(0).float()
        except Exception as e:
            print(f"Error in street noise generation: {e}. Using white noise instead.")
            noise = base_noise
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    # Ensure noise has exactly the same shape as waveform
    if noise.shape != waveform.shape:
        print(f"Warning: Noise shape {noise.shape} does not match waveform shape {waveform.shape}. Reshaping noise.")
        noise = base_noise  # Fallback to white noise if shapes don't match
    
    # Calculate noise power
    noise_power = torch.mean(noise ** 2)
    
    # Calculate scaling factor for desired SNR
    scaling_factor = torch.sqrt(signal_power / (noise_power * (10 ** (snr / 10))))
    
    # Add scaled noise to the signal
    noisy_waveform = waveform + scaling_factor * noise
    
    return noisy_waveform


###########################
# Model Architecture
###########################

class AudioFingerprinter(nn.Module):
    """
    Neural network architecture for generating audio fingerprints.
    Features attention mechanism and dropout for robustness to noise.
    """
    def __init__(self, embedding_dim=128, input_channels=1, input_height=64, input_width=128):
        super(AudioFingerprinter, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(0.3)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate the size of the flattened features
        self.flattened_size = 256 * (input_height // 16) * (input_width // 16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        x = self.dropout4(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)
        x = self.fc2(x)
        
        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x


###########################
# Training Components
###########################

class TripletDataset(Dataset):
    """
    Dataset for triplet loss training (anchor, positive, negative samples).
    """
    def __init__(self, base_dataset, triplets):
        self.base_dataset = base_dataset
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        
        anchor, _, _ = self.base_dataset[anchor_idx]
        positive, _, _ = self.base_dataset[positive_idx]
        negative, _, _ = self.base_dataset[negative_idx]
        
        # Extract mel-spectrograms
        anchor_melspec = extract_melspectrogram(anchor)
        positive_melspec = extract_melspectrogram(positive)
        negative_melspec = extract_melspectrogram(negative)
        
        return anchor_melspec, positive_melspec, negative_melspec

def create_triplet_dataset(dataset, same_class_pairs=1000, different_class_pairs=1000):
    """
    Create triplets (anchor, positive, negative) for training.
    """
    triplets = []
    
    # Group file paths by label
    label_to_files = {}
    for i in range(len(dataset)):
        waveform, label, file_path = dataset[i]
        if label not in label_to_files:
            label_to_files[label] = []
        label_to_files[label].append((i, file_path))
    
    # Create same class pairs
    for label, files in label_to_files.items():
        if len(files) < 2:
            continue
        
        pairs = min(same_class_pairs, len(files) * (len(files) - 1) // 2)
        
        for _ in range(pairs):
            idx1, idx2 = random.sample(range(len(files)), 2)
            anchor_idx, anchor_path = files[idx1]
            positive_idx, positive_path = files[idx2]
            
            # Find a negative sample from a different class
            negative_label = random.choice([l for l in label_to_files.keys() if l != label])
            negative_idx, negative_path = random.choice(label_to_files[negative_label])
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
    
    return triplets

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """
    Train the audio fingerprinting model using triplet loss.
    """
    model = model.to(device)
    
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            
            # Compute loss
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (anchor, positive, negative) in enumerate(val_loader):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                anchor_embedding = model(anchor)
                positive_embedding = model(positive)
                negative_embedding = model(negative)
                
                loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


###########################
# Locality-Sensitive Hashing
###########################

class LSH:
    """
    Locality-Sensitive Hashing for efficient similarity search.
    """
    def __init__(self, embedding_dim=128, num_tables=10, hash_size=10):
        self.embedding_dim = embedding_dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # Random projection vectors for each hash table
        self.projection_vectors = [
            torch.randn(hash_size, embedding_dim) for _ in range(num_tables)
        ]
        
        # Hash tables
        self.hash_tables = [{} for _ in range(num_tables)]
        
        # Store all embeddings
        self.all_embeddings = []
        self.file_paths = []
    
    def _hash_embedding(self, embedding, table_idx):
        """
        Hash an embedding vector using random projections.
        """
        projections = torch.matmul(self.projection_vectors[table_idx], embedding)
        hash_bits = (projections > 0).int()
        hash_value = sum([bit * (2 ** i) for i, bit in enumerate(hash_bits)])
        return int(hash_value)
    
    def index_embeddings(self, embeddings, file_paths):
        """
        Index a batch of embeddings.
        """
        for i, embedding in enumerate(embeddings):
            self.all_embeddings.append(embedding)
            self.file_paths.append(file_paths[i])
            
            for table_idx in range(self.num_tables):
                hash_value = self._hash_embedding(embedding, table_idx)
                
                if hash_value not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_value] = []
                
                self.hash_tables[table_idx][hash_value].append(i)
    
    def query(self, query_embedding, top_k=5):
        """
        Query the LSH tables for similar embeddings.
        """
        candidate_indices = set()
        
        # Collect candidates from all hash tables
        for table_idx in range(self.num_tables):
            hash_value = self._hash_embedding(query_embedding, table_idx)
            
            if hash_value in self.hash_tables[table_idx]:
                candidate_indices.update(self.hash_tables[table_idx][hash_value])
        
        # Calculate actual distances for candidates
        distances = {}
        for idx in candidate_indices:
            distance = torch.norm(query_embedding - self.all_embeddings[idx], p=2).item()
            distances[idx] = distance
        
        # Sort by distance
        sorted_candidates = sorted(distances.items(), key=lambda x: x[1])
        
        # Return top-k results
        results = []
        for idx, distance in sorted_candidates[:top_k]:
            results.append((self.file_paths[idx], distance))
        
        return results


###########################
# Evaluation
###########################

def evaluate_model(model, dataset, lsh, device='cuda', noise_types=['white', 'pink', 'street'], snr_levels=[0, 5, 10, 15, 20]):
    """
    Evaluate the audio fingerprinting model on different noise conditions.
    """
    model.eval()
    
    results = {
        'clean': {'top1': 0, 'top5': 0},
    }
    
    for noise_type in noise_types:
        for snr in snr_levels:
            key = f"{noise_type}_{snr}dB"
            results[key] = {'top1': 0, 'top5': 0}
    
    for i in range(len(dataset)):
        waveform, label, file_path = dataset[i]
        
        # Evaluate on clean audio
        melspec = extract_melspectrogram(waveform).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            embedding = model(melspec)[0].cpu()
        query_time = time.time() - start_time
        
        query_results = lsh.query(embedding, top_k=5)
        
        if query_results[0][0] == file_path:
            results['clean']['top1'] += 1
        
        if any(res[0] == file_path for res in query_results):
            results['clean']['top5'] += 1
        
        # Evaluate on noisy audio
        for noise_type in noise_types:
            for snr in snr_levels:
                key = f"{noise_type}_{snr}dB"
                
                # Add noise
                noisy_waveform = add_noise(waveform, noise_type, snr)
                
                # Extract features
                noisy_melspec = extract_melspectrogram(noisy_waveform).unsqueeze(0).to(device)
                
                # Generate embedding
                with torch.no_grad():
                    noisy_embedding = model(noisy_melspec)[0].cpu()
                
                # Query LSH
                query_results = lsh.query(noisy_embedding, top_k=5)
                
                # Check if the correct file is in the results
                if query_results[0][0] == file_path:
                    results[key]['top1'] += 1
                
                if any(res[0] == file_path for res in query_results):
                    results[key]['top5'] += 1
    
    # Calculate percentages
    num_samples = len(dataset)
    
    for condition, metrics in results.items():
        metrics['top1'] = (metrics['top1'] / num_samples) * 100
        metrics['top5'] = (metrics['top5'] / num_samples) * 100
    
    print("\nEvaluation Results:")
    print(f"Clean audio - Top-1: {results['clean']['top1']:.2f}%, Top-5: {results['clean']['top5']:.2f}%")
    
    for noise_type in noise_types:
        print(f"\n{noise_type.capitalize()} Noise:")
        for snr in snr_levels:
            key = f"{noise_type}_{snr}dB"
            print(f"  SNR {snr} dB - Top-1: {results[key]['top1']:.2f}%, Top-5: {results[key]['top5']:.2f}%")
    
    print(f"\nAverage query time: {query_time:.4f} seconds")
    
    return results


###########################
# Visualization Functions
###########################

def visualize_spectrograms(clean_waveform, noisy_waveform, sample_rate=22050, title=None):
    """
    Visualize spectrograms of clean and noisy audio.
    """
    plt.figure(figsize=(12, 8))
    
    # Extract mel-spectrograms
    clean_melspec = extract_melspectrogram(clean_waveform)[0].numpy()
    noisy_melspec = extract_melspectrogram(noisy_waveform)[0].numpy()
    
    # Plot clean spectrogram
    plt.subplot(2, 1, 1)
    plt.title("Clean Audio Spectrogram")
    plt.imshow(clean_melspec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    
    # Plot noisy spectrogram
    plt.subplot(2, 1, 2)
    plt.title("Noisy Audio Spectrogram")
    plt.imshow(noisy_melspec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(f"spectrograms_{title.replace(' ', '_')}.png")
    plt.close()

def visualize_embeddings(model, dataset, device='cuda', num_samples=500, title=None):
    """
    Visualize embeddings using t-SNE.
    """
    # Sample data
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i in indices:
            waveform, label, _ = dataset[i]
            melspec = extract_melspectrogram(waveform).unsqueeze(0).to(device)
            embedding = model(melspec)[0].cpu().numpy()
            embeddings.append(embedding)
            labels.append(label)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        idx = np.where(np.array(labels) == label)[0]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=[colors[i]], label=dataset.label_to_genre[label])
    
    plt.legend()
    plt.title(title or "t-SNE Visualization of Audio Embeddings")
    plt.savefig(f"embeddings_{title.replace(' ', '_')}.png")
    plt.close()


###########################
# Main Execution
###########################

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset parameters
    data_dir = "genres_original"
    sample_rate = 22050
    duration = 3  # seconds
    
    # Model parameters
    embedding_dim = 128
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # LSH parameters
    num_tables = 10
    hash_size = 16
    
    # Create dataset
    dataset = GTZANDataset(data_dir, sample_rate=sample_rate, duration=duration)
    print(f"Loaded {len(dataset)} audio files from {len(dataset.genres)} genres")
    
    # Split dataset
    train_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create triplets for training
    print("Creating triplets for training...")
    triplets = create_triplet_dataset(train_dataset)
    triplet_dataset = TripletDataset(dataset, triplets)
    
    # Create data loaders
    train_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Create validation triplets and loader
    val_triplets = create_triplet_dataset(test_dataset, same_class_pairs=500, different_class_pairs=500)
    val_triplet_dataset = TripletDataset(dataset, val_triplets)
    val_loader = DataLoader(val_triplet_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = AudioFingerprinter(embedding_dim=embedding_dim)
    
    # Train model
    print("Training model...")
    model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    
    # Save the trained model
    torch.save(model.state_dict(), "audio_fingerprinter.pth")
    print("Model saved to audio_fingerprinter.pth")
    
    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings(model, dataset, device=device, title="Audio Embeddings")
    
    # Generate embeddings for the test dataset
    print("Generating embeddings for test dataset...")
    embeddings = []
    file_paths = []
    
    model.eval()
    with torch.no_grad():
        for i in test_indices:
            waveform, _, file_path = dataset[i]
            melspec = extract_melspectrogram(waveform).unsqueeze(0).to(device)
            embedding = model(melspec)[0].cpu()
            embeddings.append(embedding)
            file_paths.append(file_path)
    
    # Initialize LSH
    print("Initializing LSH...")
    lsh = LSH(embedding_dim=embedding_dim, num_tables=num_tables, hash_size=hash_size)
    
    # Index embeddings
    print("Indexing embeddings...")
    lsh.index_embeddings(embeddings, file_paths)
    
    # Visualize some spectrograms
    print("Visualizing spectrograms...")
    for i, noise_type in enumerate(['white', 'pink', 'street']):
        waveform, _, _ = dataset[test_indices[i]]
        noisy_waveform = add_noise(waveform, noise_type, snr=10)
        visualize_spectrograms(waveform, noisy_waveform, title=f"{noise_type.capitalize()} Noise (10 dB SNR)")
    
    # Evaluate model
    print("Evaluating model...")
    test_subset = torch.utils.data.Subset(dataset, test_indices[:100])  # Use a subset for evaluation
    results = evaluate_model(model, test_subset, lsh, device=device)
    
    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write("Evaluation Results:\n")
        f.write(f"Clean audio - Top-1: {results['clean']['top1']:.2f}%, Top-5: {results['clean']['top5']:.2f}%\n")
        
        for noise_type in ['white', 'pink', 'street']:
            f.write(f"\n{noise_type.capitalize()} Noise:\n")
            for snr in [0, 5, 10, 15, 20]:
                key = f"{noise_type}_{snr}dB"
                f.write(f"  SNR {snr} dB - Top-1: {results[key]['top1']:.2f}%, Top-5: {results[key]['top5']:.2f}%\n")

if __name__ == "__main__":
    main()