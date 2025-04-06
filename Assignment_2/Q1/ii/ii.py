import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from peft import LoraConfig, get_peft_model, TaskType
import soundfile as sf
import matplotlib.pyplot as plt
import random
import math
import pickle
import warnings
warnings.filterwarnings("ignore")

# Constants and paths
BASE_DIR = '/DATA/rl_gaming/su_wav'
OUTPUT_DIR = '/DATA/rl_gaming/results'
VOX1_DIR = os.path.join(BASE_DIR, 'vox1')
VOX2_DIR = os.path.join(BASE_DIR, 'vox2')
TRIAL_PATH = os.path.join(VOX1_DIR, 'veri_test.txt')
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#################################################
# Utility Functions
#################################################

def load_audio(path, target_sr=16000):
    """Load audio file and convert to target sample rate"""
    try:
        # Check if it's a path from veri_test.txt that needs to be corrected
        if '/vox1_test_wav/wav/' not in path and 'id10' in path:
            # This is a VoxCeleb1 path from the trial list, fix it
            parts = path.split('/')
            if len(parts) >= 3:  # Should have at least id/session/file format
                speaker_id = parts[-3]
                session_id = parts[-2]
                file_name = parts[-1]
                corrected_path = os.path.join(VOX1_DIR, 'vox1_test_wav', 'wav', speaker_id, session_id, file_name)
                path = corrected_path
        
        # Try loading with torchaudio
        if os.path.exists(path):
            waveform, sample_rate = torchaudio.load(path)
            
            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            return waveform.squeeze(0).numpy()
        else:
            print(f"Warning: File not found: {path}")
            return np.zeros(target_sr)  # Return silence
    
    except Exception as e:
        try:
            # Use soundfile as a fallback
            audio, sr = sf.read(path)
            if sr != target_sr:
                # Resample if needed
                audio = sf.resample(audio, target_sr, sr)
            return audio
        except Exception as inner_e:
            print(f"Failed to load audio {path}: {inner_e}")
            # Return a short segment of silence
            return np.zeros(target_sr)

def compute_eer(scores, labels):
    """Compute Equal Error Rate (EER)"""
    # Sort scores and corresponding labels
    sorted_indexes = np.argsort(scores)
    labels = np.asarray(labels)[sorted_indexes]
    scores = np.asarray(scores)[sorted_indexes]
    
    # Count number of positive and negative samples
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    
    # Calculate FPR and FNR at different thresholds
    far = np.cumsum(labels) / n_pos  # False acceptance rate (1 - TPR)
    frr = 1 - np.cumsum(1 - labels) / n_neg  # False rejection rate (FNR)
    
    # Find the threshold where FAR = FRR (EER)
    idx = np.nanargmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2
    
    return eer * 100  # Return as percentage

def compute_tar_at_far(scores, labels, target_far=0.01):
    """Compute True Acceptance Rate at a specific False Acceptance Rate"""
    # Sort scores and corresponding labels
    sorted_indexes = np.argsort(scores)
    labels = np.asarray(labels)[sorted_indexes]
    scores = np.asarray(scores)[sorted_indexes]
    
    # Count number of positive and negative samples
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    
    # Calculate FPR and TPR at different thresholds
    fpr = np.cumsum(1 - labels) / n_neg
    tpr = np.cumsum(labels) / n_pos
    
    # Find the threshold where FPR is closest to the target FAR
    idx = np.nanargmin(np.abs(fpr - target_far))
    
    return tpr[idx]

def compute_ident_accuracy(embeddings, labels):
    """Compute speaker identification accuracy using cosine similarity"""
    # Convert to numpy arrays if they're not already
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    # Normalize embeddings
    embeddings = normalize(embeddings)
    
    # Use first embedding of each speaker as enrollment
    unique_labels = np.unique(labels)
    enrollment_embeddings = []
    enrollment_labels = []
    
    for label in unique_labels:
        idx = np.where(labels == label)[0][0]
        enrollment_embeddings.append(embeddings[idx])
        enrollment_labels.append(label)
    
    enrollment_embeddings = np.array(enrollment_embeddings)
    
    # Compute similarities for all test embeddings
    correct = 0
    total = 0
    
    for i, embedding in enumerate(embeddings):
        # Skip if this is an enrollment embedding
        if i in np.where(labels == labels[i])[0][:1]:
            continue
        
        # Compute similarities
        similarities = np.dot(enrollment_embeddings, embedding)
        pred_idx = np.argmax(similarities)
        pred_label = enrollment_labels[pred_idx]
        
        if pred_label == labels[i]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

#################################################
# Data Preparation & Datasets
#################################################

def load_trial_pairs(trial_path):
    """Load trial pairs from verification file"""
    trial_pairs = []
    with open(trial_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Format: label wav1_path wav2_path
            label = int(parts[0])
            
            # Construct proper paths with vox1_test_wav/wav in the path
            wav1_path = os.path.join(VOX1_DIR, 'vox1_test_wav', 'wav', parts[1])
            wav2_path = os.path.join(VOX1_DIR, 'vox1_test_wav', 'wav', parts[2])
            
            trial_pairs.append((wav1_path, wav2_path, label))
    return trial_pairs

class VoxCeleb2Dataset(Dataset):
    """Dataset for VoxCeleb2"""
    def __init__(self, root_dir, speaker_ids, feature_extractor, max_length=160000):
        self.root_dir = root_dir
        self.speaker_ids = speaker_ids
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for speaker_id in tqdm(self.speaker_ids, desc="Loading dataset"):
            speaker_dir = os.path.join(self.root_dir, 'vox2_test_aac/aac', speaker_id)
            
            if not os.path.exists(speaker_dir):
                print(f"Warning: Directory not found: {speaker_dir}")
                continue
                
            for session_dir in os.listdir(speaker_dir):
                session_path = os.path.join(speaker_dir, session_dir)
                
                if not os.path.isdir(session_path):
                    continue
                    
                for audio_file in os.listdir(session_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(session_path, audio_file)
                        samples.append((audio_path, self.speaker_ids.index(speaker_id)))
        
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        audio_path, speaker_idx = self.samples[idx]
        
        # Load and preprocess audio
        waveform = load_audio(audio_path)
        
        # Ensure the waveform is the right length
        if len(waveform) > self.max_length:
            # Randomly select a segment
            start = random.randint(0, len(waveform) - self.max_length)
            waveform = waveform[start:start + self.max_length]
        else:
            # Pad with zeros
            waveform = np.pad(waveform, (0, max(0, self.max_length - len(waveform))))
        
        # Prepare inputs for the model
        inputs = self.feature_extractor(
            waveform, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.max_length
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'speaker_id': speaker_idx,
            'path': audio_path
        }

def get_speaker_ids(root_dir, n_train=100, n_test=18):
    """Get sorted speaker IDs for training and testing"""
    vox2_dir = os.path.join(root_dir, 'vox2_test_aac/aac')
    all_speaker_ids = sorted([d for d in os.listdir(vox2_dir) if os.path.isdir(os.path.join(vox2_dir, d))])
    
    train_ids = all_speaker_ids[:n_train]
    test_ids = all_speaker_ids[n_train:n_train + n_test]
    
    return train_ids, test_ids

#################################################
# ArcFace Loss
#################################################

class ArcFaceLoss(nn.Module):
    """ArcFace loss for speaker verification"""
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Normalize weights
        weights_norm = F.normalize(self.weight, dim=1)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        
        # Calculate cosine similarity
        cos_theta = F.linear(embeddings_norm, weights_norm)
        
        # Clip to prevent numerical instability
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
        
        # Calculate arccos
        theta = torch.acos(cos_theta)
        
        # Add margin to target classes
        target_mask = torch.zeros_like(cos_theta)
        target_mask.scatter_(1, labels.view(-1, 1), 1.0)
        theta += self.margin * target_mask
        
        # Convert back to cosine
        cos_theta_m = torch.cos(theta)
        
        # Scale by s
        scaled_cos_theta = self.scale * cos_theta_m
        
        # Cross-entropy loss
        loss = F.cross_entropy(scaled_cos_theta, labels)
        
        return loss

#################################################
# Model Handling
#################################################

class SpeakerVerificationModel(nn.Module):
    """Speaker verification model based on WavLM with optional LoRA fine-tuning"""
    def __init__(self, num_classes, pretrained_model_name="microsoft/wavlm-base-plus", use_lora=False):
        super(SpeakerVerificationModel, self).__init__()
        
        # Load pre-trained model
        self.wavlm = WavLMModel.from_pretrained(pretrained_model_name)
        self.use_lora = use_lora
        
        # Apply LoRA if requested
        if use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                r=8,  # Rank of the update matrices
                lora_alpha=16,  # Scaling factor
                target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],  # Apply LoRA to attention modules
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            # Apply LoRA to the model
            self.wavlm = get_peft_model(self.wavlm, lora_config)
            self.wavlm.print_trainable_parameters()
            
            # Store the original base model for direct access
            self.base_wavlm = self.wavlm.base_model.model
            
        else:
            # Freeze all parameters if not using LoRA
            for param in self.wavlm.parameters():
                param.requires_grad = False
            
            # No base model in this case
            self.base_wavlm = None
        
        # Speaker embedding layers
        hidden_size = self.wavlm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Classification layer for training
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, input_values, extract_embedding=False):
        # Get WavLM outputs, bypassing PEFT wrapper if LoRA is used
        if self.use_lora:
            # Access the base model directly to avoid parameter name conversion
            outputs = self.base_wavlm(input_values).last_hidden_state
        else:
            # For regular models without LoRA
            outputs = self.wavlm(input_values).last_hidden_state
        
        # Pool across time dimension (mean pooling)
        embeddings = torch.mean(outputs, dim=1)
        
        # Project to speaker embedding space
        embeddings = self.projector(embeddings)
        
        if extract_embedding:
            return embeddings
        
        # Classification for training
        logits = self.classifier(embeddings)
        return logits, embeddings

def evaluate_verification(model, trial_pairs, feature_extractor, device):
    """Evaluate the model on verification pairs"""
    model.eval()
    scores = []
    labels = []
    
    # Process pairs in smaller batches to avoid memory issues
    batch_size = 16
    total_pairs = len(trial_pairs)
    
    for batch_idx in tqdm(range(0, total_pairs, batch_size), desc="Evaluating verification"):
        batch_pairs = trial_pairs[batch_idx:min(batch_idx + batch_size, total_pairs)]
        batch_scores = []
        batch_labels = []
        
        # Process each pair
        for wav1_path, wav2_path, label in batch_pairs:
            try:
                # Load and preprocess audio files
                waveform1 = load_audio(wav1_path)
                waveform2 = load_audio(wav2_path)
                
                # Skip if either audio failed to load (returned silence)
                if np.all(waveform1 == 0) or np.all(waveform2 == 0):
                    continue
                
                with torch.no_grad():
                    # Convert to model inputs
                    inputs1 = feature_extractor(
                        waveform1, 
                        sampling_rate=16000, 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=160000
                    ).input_values.to(device)
                    
                    inputs2 = feature_extractor(
                        waveform2, 
                        sampling_rate=16000, 
                        return_tensors="pt", 
                        padding="max_length", 
                        max_length=160000
                    ).input_values.to(device)
                    
                    # Extract embeddings
                    embedding1 = model(inputs1, extract_embedding=True)
                    embedding2 = model(inputs2, extract_embedding=True)
                    
                    # Compute cosine similarity
                    similarity = F.cosine_similarity(embedding1, embedding2).item()
                    
                    batch_scores.append(similarity)
                    batch_labels.append(label)
            except Exception as e:
                print(f"Error processing pair ({wav1_path}, {wav2_path}): {e}")
                continue
        
        # Add batch results to overall results
        scores.extend(batch_scores)
        labels.extend(batch_labels)
        
        # Save intermediate results periodically
        if batch_idx % 10 == 0 and len(scores) > 0:
            # This helps recover in case of process interruption
            temp_results = {
                "scores": scores,
                "labels": labels,
                "progress": f"{batch_idx}/{total_pairs} batches"
            }
            with open(os.path.join(OUTPUT_DIR, 'verification_progress.pkl'), 'wb') as f:
                pickle.dump(temp_results, f)
    
    if len(scores) == 0:
        print("Warning: No valid verification pairs processed")
        return 50.0, 0.0, [], []  # Return default values
    
    # Calculate metrics
    eer = compute_eer(scores, labels)
    tar_1_far = compute_tar_at_far(scores, labels, target_far=0.01)
    
    return eer, tar_1_far, scores, labels

def evaluate_identification(model, dataset, device):
    """Evaluate the model on identification task"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_values = batch['input_values'].to(device)
            speaker_ids = batch['speaker_id'].numpy()
            
            # Extract embeddings
            embeddings = model(input_values, extract_embedding=True)
            
            all_embeddings.extend(embeddings.cpu().numpy())
            all_labels.extend(speaker_ids)
    
    # Calculate identification accuracy
    accuracy = compute_ident_accuracy(all_embeddings, all_labels)
    
    return accuracy

#################################################
# Main Training and Evaluation Functions
#################################################

def fine_tune_model(model, train_dataset, test_dataset, num_epochs=10, batch_size=16, learning_rate=5e-5):
    """Fine-tune the model using LoRA and ArcFace loss"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define loss function
    criterion = ArcFaceLoss(embedding_size=256, num_classes=len(train_dataset.speaker_ids))
    criterion = criterion.to(DEVICE)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            input_values = batch['input_values'].to(DEVICE)
            speaker_ids = batch['speaker_id'].to(DEVICE)
            
            # Forward pass
            logits, embeddings = model(input_values)
            
            # Calculate loss
            loss = criterion(embeddings, speaker_ids)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Validating epoch {epoch+1}/{num_epochs}"):
                # Move data to device
                input_values = batch['input_values'].to(DEVICE)
                speaker_ids = batch['speaker_id'].to(DEVICE)
                
                # Forward pass
                logits, embeddings = model(input_values)
                
                # Calculate loss
                loss = criterion(embeddings, speaker_ids)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += speaker_ids.size(0)
                correct += (predicted == speaker_ids).sum().item()
        
        val_loss /= len(test_loader)
        accuracy = correct / total
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
            print(f"Saved best model with accuracy {best_accuracy:.4f}")
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    return model

def main():
    print(f"Using device: {DEVICE}")
    
    # Load trial pairs for verification
    trial_pairs = load_trial_pairs(TRIAL_PATH)
    print(f"Loaded {len(trial_pairs)} trial pairs for verification")
    
    # Get speaker IDs for training and testing
    train_speaker_ids, test_speaker_ids = get_speaker_ids(VOX2_DIR, n_train=100, n_test=18)
    print(f"Selected {len(train_speaker_ids)} speakers for training and {len(test_speaker_ids)} speakers for testing")
    
    # Initialize the feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Create datasets
    train_dataset = VoxCeleb2Dataset(VOX2_DIR, train_speaker_ids, feature_extractor)
    test_dataset = VoxCeleb2Dataset(VOX2_DIR, test_speaker_ids, feature_extractor)
    
    print(f"Created training dataset with {len(train_dataset)} samples and testing dataset with {len(test_dataset)} samples")
    
    # Save feature extractor for later use in task III
    with open(os.path.join(OUTPUT_DIR, 'feature_extractor.pkl'), 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Step 1: Evaluate pre-trained model
    if os.path.exists(os.path.join(OUTPUT_DIR, 'pretrained_results.pkl')):
        # Load cached results if available
        print("Loading cached pre-trained model results...")
        with open(os.path.join(OUTPUT_DIR, 'pretrained_results.pkl'), 'rb') as f:
            pretrained_results = pickle.load(f)
        pretrained_eer = pretrained_results['eer']
        pretrained_tar_1_far = pretrained_results['tar_1_far']
        pretrained_scores = pretrained_results['scores']
        pretrained_labels = pretrained_results['labels']
        pretrained_ident_acc = pretrained_results['ident_acc']
    else:
        print("Evaluating pre-trained model...")
        pretrained_model = SpeakerVerificationModel(
            num_classes=len(train_speaker_ids), 
            pretrained_model_name="microsoft/wavlm-base-plus",
            use_lora=False
        ).to(DEVICE)
        
        # Save the pre-trained model for later use in task III
        torch.save(pretrained_model.state_dict(), os.path.join(OUTPUT_DIR, 'pretrained_model.pt'))
        
        # Evaluate pre-trained model on verification task
        pretrained_eer, pretrained_tar_1_far, pretrained_scores, pretrained_labels = evaluate_verification(
            pretrained_model, trial_pairs, feature_extractor, DEVICE
        )
        
        # Evaluate pre-trained model on identification task
        pretrained_ident_acc = evaluate_identification(pretrained_model, test_dataset, DEVICE)
        
        # Cache results
        pretrained_results = {
            'eer': pretrained_eer,
            'tar_1_far': pretrained_tar_1_far,
            'scores': pretrained_scores,
            'labels': pretrained_labels,
            'ident_acc': pretrained_ident_acc
        }
        with open(os.path.join(OUTPUT_DIR, 'pretrained_results.pkl'), 'wb') as f:
            pickle.dump(pretrained_results, f)
    
    print(f"Pre-trained model metrics:")
    print(f"  EER: {pretrained_eer:.2f}%")
    print(f"  TAR@1%FAR: {pretrained_tar_1_far:.4f}")
    print(f"  Identification Accuracy: {pretrained_ident_acc:.4f}")
    
    # Step 2: Fine-tune model with LoRA and ArcFace loss
    print("Fine-tuning model with LoRA and ArcFace loss...")
    finetuned_model = SpeakerVerificationModel(
        num_classes=len(train_speaker_ids), 
        pretrained_model_name="microsoft/wavlm-base-plus",
        use_lora=True
    ).to(DEVICE)
    
    # Check if we have a previously fine-tuned model
    if os.path.exists(os.path.join(OUTPUT_DIR, 'best_model.pt')):
        print("Loading previously fine-tuned model...")
        finetuned_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
    else:
        finetuned_model = fine_tune_model(
            finetuned_model, 
            train_dataset, 
            test_dataset,
            num_epochs=3,
            batch_size=12,
            learning_rate=1e-4
        )
    
    # Save the fine-tuned model for later use in task III
    torch.save(finetuned_model.state_dict(), os.path.join(OUTPUT_DIR, 'finetuned_model.pt'))
    
    if os.path.exists(os.path.join(OUTPUT_DIR, 'finetuned_results.pkl')):
        # Load cached results if available
        print("Loading cached fine-tuned model results...")
        with open(os.path.join(OUTPUT_DIR, 'finetuned_results.pkl'), 'rb') as f:
            finetuned_results = pickle.load(f)
        finetuned_eer = finetuned_results['eer']
        finetuned_tar_1_far = finetuned_results['tar_1_far']
        finetuned_scores = finetuned_results['scores']
        finetuned_labels = finetuned_results['labels']
        finetuned_ident_acc = finetuned_results['ident_acc']
    else:
        # Evaluate fine-tuned model on verification task
        finetuned_eer, finetuned_tar_1_far, finetuned_scores, finetuned_labels = evaluate_verification(
            finetuned_model, trial_pairs, feature_extractor, DEVICE
        )
        
        # Evaluate fine-tuned model on identification task
        finetuned_ident_acc = evaluate_identification(finetuned_model, test_dataset, DEVICE)
        
        # Cache results
        finetuned_results = {
            'eer': finetuned_eer,
            'tar_1_far': finetuned_tar_1_far,
            'scores': finetuned_scores,
            'labels': finetuned_labels,
            'ident_acc': finetuned_ident_acc
        }
        with open(os.path.join(OUTPUT_DIR, 'finetuned_results.pkl'), 'wb') as f:
            pickle.dump(finetuned_results, f)
    
    print(f"Fine-tuned model metrics:")
    print(f"  EER: {finetuned_eer:.2f}%")
    print(f"  TAR@1%FAR: {finetuned_tar_1_far:.4f}")
    print(f"  Identification Accuracy: {finetuned_ident_acc:.4f}")
    
    # Step 3: Plot ROC curves for comparison
    plt.figure(figsize=(10, 8))
    
    # Calculate ROC curve for pre-trained model
    fpr_pretrained = []
    tpr_pretrained = []
    thresholds = np.linspace(-1, 1, 100)
    for threshold in thresholds:
        fp = sum(1 for i, score in enumerate(pretrained_scores) if score >= threshold and pretrained_labels[i] == 0)
        tp = sum(1 for i, score in enumerate(pretrained_scores) if score >= threshold and pretrained_labels[i] == 1)
        fn = sum(1 for i, score in enumerate(pretrained_scores) if score < threshold and pretrained_labels[i] == 1)
        tn = sum(1 for i, score in enumerate(pretrained_scores) if score < threshold and pretrained_labels[i] == 0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fpr_pretrained.append(fpr)
        tpr_pretrained.append(tpr)
    
    # Calculate ROC curve for fine-tuned model
    fpr_finetuned = []
    tpr_finetuned = []
    for threshold in thresholds:
        fp = sum(1 for i, score in enumerate(finetuned_scores) if score >= threshold and finetuned_labels[i] == 0)
        tp = sum(1 for i, score in enumerate(finetuned_scores) if score >= threshold and finetuned_labels[i] == 1)
        fn = sum(1 for i, score in enumerate(finetuned_scores) if score < threshold and finetuned_labels[i] == 1)
        tn = sum(1 for i, score in enumerate(finetuned_scores) if score < threshold and finetuned_labels[i] == 0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fpr_finetuned.append(fpr)
        tpr_finetuned.append(tpr)
    
    # Plot ROC curves
    plt.plot(fpr_pretrained, tpr_pretrained, label=f'Pre-trained (EER: {pretrained_eer:.2f}%)')
    plt.plot(fpr_finetuned, tpr_finetuned, label=f'Fine-tuned (EER: {finetuned_eer:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('True Acceptance Rate')
    plt.title('ROC Curves for Speaker Verification')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
    
    # Save results to CSV
    results = {
        'Model': ['Pre-trained', 'Fine-tuned'],
        'EER (%)': [pretrained_eer, finetuned_eer],
        'TAR@1%FAR': [pretrained_tar_1_far, finetuned_tar_1_far],
        'Identification Accuracy': [pretrained_ident_acc, finetuned_ident_acc]
    }
    
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'verification_results.csv'), index=False)
    
    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'verification_results.csv')}")
    print(f"ROC curves saved to {os.path.join(OUTPUT_DIR, 'roc_curves.png')}")

if __name__ == "__main__":
    main()