import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import matplotlib.pyplot as plt
import random
import pickle
import gc
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from peft import LoraConfig, get_peft_model, TaskType
try:
    from speechbrain.inference import SepformerSeparation
except ImportError:
    from speechbrain.pretrained import SepformerSeparation
import mir_eval
from pesq import pesq
import warnings
warnings.filterwarnings("ignore")

# Constants and paths
BASE_DIR = '/DATA/rl_gaming/su_wav'
OUTPUT_DIR = '/DATA/rl_gaming/results_iii'
MODEL_DIR = '/DATA/rl_gaming/results'  # Directory with saved models from task II
VOX1_DIR = os.path.join(BASE_DIR, 'vox1')
VOX2_DIR = os.path.join(BASE_DIR, 'vox2')
MIXED_DIR = os.path.join(OUTPUT_DIR, 'mixed_audio')
SEPARATED_DIR = os.path.join(OUTPUT_DIR, 'separated_audio')
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Create necessary directories
for directory in [OUTPUT_DIR, MIXED_DIR, SEPARATED_DIR,
                 os.path.join(MIXED_DIR, 'train'), 
                 os.path.join(MIXED_DIR, 'test'),
                 os.path.join(SEPARATED_DIR, 'test')]:
    os.makedirs(directory, exist_ok=True)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#################################################
# Speaker Verification Model from Task II
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
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["k_proj", "q_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.wavlm = get_peft_model(self.wavlm, lora_config)
            self.base_wavlm = self.wavlm.base_model.model
        else:
            for param in self.wavlm.parameters():
                param.requires_grad = False
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
    
    def forward(self, input_values, extract_embedding=True):
        # Get WavLM outputs
        if self.use_lora:
            outputs = self.base_wavlm(input_values).last_hidden_state
        else:
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

#################################################
# Utility Functions
#################################################

def load_audio(path, target_sr=16000, max_duration=10):
    """Load audio file with error handling"""
    try:
        if os.path.exists(path):
            waveform, sample_rate = torchaudio.load(path)
            
            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
            
            # Limit duration
            max_samples = max_duration * target_sr
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
                
            # Ensure audio isn't empty or too short
            if waveform.numel() == 0 or waveform.shape[1] < target_sr * 0.5:  # at least 0.5 seconds
                print(f"Warning: Audio too short in {path}")
                return generate_dummy_audio(target_sr), target_sr
            
            return waveform.squeeze(0).numpy(), target_sr
        else:
            print(f"Warning: File not found: {path}")
            return generate_dummy_audio(target_sr), target_sr
    
    except Exception as e:
        print(f"Failed to load audio {path}: {e}")
        return generate_dummy_audio(target_sr), target_sr

def generate_dummy_audio(sr=16000, duration=1.0):
    """Generate a dummy audio signal for fallback"""
    samples = int(sr * duration)
    return np.sin(np.linspace(0, 20*np.pi, samples))

def mix_audios(wav1, wav2, snr_range=(-5, 5)):
    """Mix two audio files with randomized SNR (LibriMix inspired)"""
    wav1, wav2 = np.array(wav1), np.array(wav2)
    
    # Match lengths
    min_len = min(len(wav1), len(wav2))
    if min_len < 16000:  # Require at least 1 second
        min_len = 16000
        wav1 = np.pad(wav1, (0, max(0, min_len - len(wav1))))
        wav2 = np.pad(wav2, (0, max(0, min_len - len(wav2))))
    wav1, wav2 = wav1[:min_len], wav2[:min_len]
    
    # Randomly select SNR within range
    snr = random.uniform(snr_range[0], snr_range[1])
    
    # Apply SNR
    energy1 = np.sum(wav1 ** 2) + 1e-10
    energy2 = np.sum(wav2 ** 2) + 1e-10
    
    scaling = np.sqrt(energy1 / (energy2 * 10 ** (snr / 10)))
    wav2_scaled = wav2 * scaling
    
    mixed = wav1 + wav2_scaled
    
    # Normalize to prevent clipping
    max_val = max(abs(mixed.max()), abs(mixed.min()))
    if max_val > 1.0:
        # Normalize all sources proportionally
        mixed = mixed / max_val
        wav1 = wav1 / max_val
        wav2_scaled = wav2_scaled / max_val
    
    return mixed, wav1, wav2_scaled

def evaluate_separation(reference_sources, estimated_sources):
    """Compute source separation metrics"""
    try:
        # Ensure all sources have the same length
        min_len = min([len(s) for s in reference_sources + estimated_sources])
        
        # Require a reasonable length for evaluation
        if min_len < 8000:  # at least 0.5 seconds at 16kHz
            print("Sources too short for proper evaluation")
            return {
                'sdr': np.array([0.0, 0.0]), 
                'sir': np.array([0.0, 0.0]), 
                'sar': np.array([0.0, 0.0]), 
                'permutation': np.array([0, 1])
            }
        
        # Trim to common length
        reference_sources = [s[:min_len] for s in reference_sources]
        estimated_sources = [s[:min_len] for s in estimated_sources]
        
        # Ensure sources are properly normalized
        for i in range(len(reference_sources)):
            max_val = np.max(np.abs(reference_sources[i]))
            if max_val > 0:
                reference_sources[i] = reference_sources[i] / max_val
                
        for i in range(len(estimated_sources)):
            max_val = np.max(np.abs(estimated_sources[i]))
            if max_val > 0:
                estimated_sources[i] = estimated_sources[i] / max_val
        
        # Compute BSS metrics
        sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(
            np.array(reference_sources), 
            np.array(estimated_sources),
            compute_permutation=True
        )
        
        # Clip extreme values
        sdr = np.clip(sdr, -30, 30)
        sir = np.clip(sir, -30, 30)
        sar = np.clip(sar, -30, 30)
        
        return {'sdr': sdr, 'sir': sir, 'sar': sar, 'permutation': perm}
    except Exception as e:
        print(f"Error in evaluate_separation: {e}")
        return {
            'sdr': np.array([0.0, 0.0]), 
            'sir': np.array([0.0, 0.0]), 
            'sar': np.array([0.0, 0.0]), 
            'permutation': np.array([0, 1])
        }

def calculate_pesq(reference, degraded, fs=16000):
    """Calculate PESQ score with error handling"""
    try:
        # Check lengths
        if len(reference) < 8000 or len(degraded) < 8000:
            print("Audio too short for PESQ calculation")
            return 2.0  # Return default value
        
        # Normalize
        reference = np.asarray(reference, dtype=np.float64)
        degraded = np.asarray(degraded, dtype=np.float64)
        
        # Match lengths
        min_len = min(len(reference), len(degraded))
        reference, degraded = reference[:min_len], degraded[:min_len]
        
        # Scale to [-1, 1]
        ref_max, deg_max = np.max(np.abs(reference)), np.max(np.abs(degraded))
        if ref_max > 1.0: reference = reference / ref_max
        if deg_max > 1.0: degraded = degraded / deg_max
            
        if ref_max < 1e-6 or deg_max < 1e-6:
            return 2.0  # Empty audio
        
        # Calculate PESQ
        return pesq(fs, reference, degraded, 'wb')
    except Exception as e:
        print(f"Error calculating PESQ: {e}")
        return 2.0

def find_speaker_files(speaker_id):
    """Find audio files for a speaker with robust path handling"""
    audio_files = []
    
    # Try different possible paths for VoxCeleb2
    possible_paths = [
        os.path.join(VOX2_DIR, 'vox2_test_aac', 'aac', speaker_id),
        os.path.join(VOX2_DIR, 'test_aac', 'aac', speaker_id),
        os.path.join(VOX2_DIR, 'aac', speaker_id),
        os.path.join(VOX2_DIR, 'test', 'aac', speaker_id),
    ]
    
    # Find the first valid path
    speaker_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            speaker_dir = path
            break
    
    if not speaker_dir:
        print(f"Could not find directory for speaker {speaker_id}")
        return []
    
    # Walk through the directory to find all WAV files
    for root, _, files in os.walk(speaker_dir):
        for file in files:
            if file.endswith('.wav'):  # Using .wav as you've converted .m4a files
                audio_files.append(os.path.join(root, file))
                if len(audio_files) >= 10:  # Limit to 10 files per speaker
                    return audio_files
    
    return audio_files

def get_speaker_ids():
    """Get speaker IDs with better error handling"""
    # Try different possible paths for VoxCeleb2
    possible_paths = [
        os.path.join(VOX2_DIR, 'vox2_test_aac', 'aac'),
        os.path.join(VOX2_DIR, 'test_aac', 'aac'),
        os.path.join(VOX2_DIR, 'aac'),
        os.path.join(VOX2_DIR, 'test', 'aac'),
    ]
    
    # Find the first valid path
    vox2_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            vox2_dir = path
            print(f"Found speaker directory at: {path}")
            break
    
    if not vox2_dir:
        print("WARNING: Could not find any speaker directories. Using default IDs.")
        # Generate 118 fake IDs (first 50 for train, next 50 for test, rest for other)
        all_ids = [f'id{i:05d}' for i in range(1, 119)]
        return all_ids[:50], all_ids[50:100]
    
    # Get all speaker IDs from the directory
    all_speaker_ids = sorted([d for d in os.listdir(vox2_dir) 
                            if os.path.isdir(os.path.join(vox2_dir, d))])
    
    # First 50 for training, next 50 for testing as per the task
    if len(all_speaker_ids) < 100:
        print(f"Warning: Only found {len(all_speaker_ids)} speakers. Adjusting splits.")
        split_point = len(all_speaker_ids) // 2
        train_ids = all_speaker_ids[:split_point]
        test_ids = all_speaker_ids[split_point:]
    else:
        train_ids = all_speaker_ids[:50]
        test_ids = all_speaker_ids[50:100]
    
    print(f"Selected {len(train_ids)} speakers for training and {len(test_ids)} speakers for testing")
    return train_ids, test_ids

#################################################
# Data Preparation
#################################################

def create_mixed_dataset(split='train', n_pairs=50, snr_range=(-5, 5)):
    """Create dataset of mixed speech from two speakers"""
    print(f"Creating mixed {split} dataset...")
    
    # Get speaker IDs for the appropriate split
    train_ids, test_ids = get_speaker_ids()
    speaker_ids = train_ids if split == 'train' else test_ids
    
    # Set up output directory
    output_dir = os.path.join(MIXED_DIR, split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    metadata_file = os.path.join(output_dir, f"{split}_metadata.csv")
    if os.path.exists(metadata_file):
        print(f"Mixed {split} dataset already exists at {metadata_file}")
        return pd.read_csv(metadata_file).to_dict('records')
    
    mixed_metadata = []
    pairs_created = 0
    max_attempts = n_pairs * 3  # Allow more attempts to find valid pairs
    attempts = 0
    
    with tqdm(total=n_pairs) as pbar:
        while pairs_created < n_pairs and attempts < max_attempts:
            attempts += 1
            
            try:
                # Select two different speakers
                speaker1, speaker2 = random.sample(speaker_ids, 2)
                
                # Find audio files for each speaker
                speaker1_files = find_speaker_files(speaker1)
                speaker2_files = find_speaker_files(speaker2)
                
                if not speaker1_files or not speaker2_files:
                    continue
                
                # Select one file from each speaker
                file1 = random.choice(speaker1_files)
                file2 = random.choice(speaker2_files)
                
                # Load audio files
                wav1, sr1 = load_audio(file1, max_duration=5)
                wav2, sr2 = load_audio(file2, max_duration=5)
                
                if sr1 != sr2 or len(wav1) < 16000 or len(wav2) < 16000:
                    continue
                
                # Mix audio files with random SNR
                mixed, clean1, clean2 = mix_audios(wav1, wav2, snr_range)
                
                # Save files
                mixed_filename = os.path.join(output_dir, f"mix_{speaker1}_{speaker2}_{pairs_created}.wav")
                clean1_filename = os.path.join(output_dir, f"s1_{speaker1}_{speaker2}_{pairs_created}.wav")
                clean2_filename = os.path.join(output_dir, f"s2_{speaker1}_{speaker2}_{pairs_created}.wav")
                
                sf.write(mixed_filename, mixed, sr1)
                sf.write(clean1_filename, clean1, sr1)
                sf.write(clean2_filename, clean2, sr1)
                
                # Update metadata
                mixed_metadata.append({
                    'mixture': mixed_filename,
                    'source1': clean1_filename,
                    'source2': clean2_filename,
                    'speaker1': speaker1,
                    'speaker2': speaker2,
                    'snr': snr_range
                })
                
                pairs_created += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error creating mix: {e}")
                continue
    
    # Save metadata
    pd.DataFrame(mixed_metadata).to_csv(metadata_file, index=False)
    
    print(f"Created {len(mixed_metadata)} mixed {split} examples.")
    return mixed_metadata

#################################################
# Speaker Separation
#################################################

def separate_speakers(mixed_metadata, output_dir=None, batch_size=5):
    """Separate mixed audio using SepFormer model and analyze all required metrics"""
    print("Loading SepFormer model...")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(SEPARATED_DIR, 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if results already exist
    metadata_file = os.path.join(output_dir, "separation_metadata.csv")
    metrics_file = os.path.join(output_dir, "separation_metrics.csv")
    
    # Check for existing separation with validation
    if os.path.exists(metadata_file):
        print(f"Found existing separation metadata at {metadata_file}")
        
        # Validate existing files
        separation_metadata = pd.read_csv(metadata_file).to_dict('records')
        if len(separation_metadata) > 0:
            first_file = separation_metadata[0]['estimated1']
            if os.path.exists(first_file):
                waveform, _ = load_audio(first_file)
                if len(waveform) > 8000:  # Valid audio
                    print("Existing separated audio files are valid.")
                    # Load metrics if they exist
                    if os.path.exists(metrics_file):
                        metrics_df = pd.read_csv(metrics_file)
                        print("\nSpeaker Separation Metrics:")
                        for col in metrics_df.columns:
                            print(f"  {col}: {metrics_df[col].values[0]:.4f}")
                    return separation_metadata
        
        # If we get here, existing files need to be regenerated
        print("Existing files invalid or incomplete. Regenerating separation...")
    
    # Load SepFormer model
    try:
        separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir="pretrained_models/sepformer-wsj02mix",
            run_opts={"device": DEVICE}
        )
    except Exception as e:
        print(f"Error loading SepFormer: {e}")
        try:
            separator = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix"
            )
        except Exception as e2:
            print(f"Critical error loading SepFormer: {e2}")
            return []
    
    separation_metadata = []
    metrics_summary = {'sdr': [], 'sir': [], 'sar': [], 'pesq': []}
    
    # Process in batches
    total_items = len(mixed_metadata)
    print(f"Separating speakers...")
    
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        print(f"Processing batch {batch_start}-{batch_end} of {total_items}")
        
        for idx in tqdm(range(batch_start, batch_end)):
            try:
                # Get paths for this item
                item = mixed_metadata[idx]
                mix_path = item['mixture']
                source1_path = item['source1']
                source2_path = item['source2']
                
                # Check if input files exist
                if not os.path.exists(mix_path) or not os.path.exists(source1_path) or not os.path.exists(source2_path):
                    print(f"Missing input file(s): {mix_path}, {source1_path}, or {source2_path}")
                    continue
                
                # Generate output filenames
                base_name = os.path.basename(mix_path).replace('mix_', '')
                est1_path = os.path.join(output_dir, f"est1_{base_name}")
                est2_path = os.path.join(output_dir, f"est2_{base_name}")
                
                # Load mixture audio
                mixture, sr = load_audio(mix_path, max_duration=5)
                
                # Skip if mixture is too short
                if len(mixture) < 16000:  # Less than 1 second at 16kHz
                    print(f"Mixture too short ({len(mixture)} samples): {mix_path}")
                    # Use a fallback strategy
                    half_point = len(mixture) // 2
                    sf.write(est1_path, mixture[:half_point], sr)
                    sf.write(est2_path, mixture[half_point:], sr)
                    continue
                
                # Load source audios for metrics
                source1, _ = load_audio(source1_path, max_duration=5)
                source2, _ = load_audio(source2_path, max_duration=5)
                
                # Skip if sources are invalid
                if len(source1) < 8000 or len(source2) < 8000:
                    print(f"Source audio too short: {len(source1)}, {len(source2)} samples")
                    # Use a fallback strategy
                    half_point = len(mixture) // 2
                    sf.write(est1_path, mixture[:half_point], sr)
                    sf.write(est2_path, mixture[half_point:], sr)
                    continue
                
                # Prepare input for SepFormer - reshape to (batch, time)
                # SepFormer expects [batch, time] not [batch, channel, time]
                mixture_torch = torch.tensor(mixture).unsqueeze(0).to(DEVICE)
                print(f"Mixture shape: {mixture_torch.shape}")
                
                # Perform separation with robust error handling
                try:
                    with torch.no_grad():
                        # Get estimated sources
                        est_sources = separator.separate_batch(mixture_torch)
                        print(f"Initial output shape: {est_sources.shape}")
                        
                        # Handle different output formats
                        if est_sources.dim() == 3:  # (batch, sources, time) or (batch, time, sources)
                            est_sources = est_sources.squeeze(0)  # Remove batch dimension
                        
                        # Check if we need to transpose
                        if est_sources.shape[0] == 2:  # (sources, time)
                            est_sources = est_sources.cpu().numpy()
                        elif est_sources.shape[1] == 2:  # (time, sources)
                            est_sources = est_sources.transpose(1, 0).cpu().numpy()
                        else:
                            print(f"Unexpected output shape: {est_sources.shape}")
                            # Fallback to splitting the mixture
                            half_point = len(mixture) // 2
                            est_sources = np.array([mixture[:half_point], mixture[half_point:]])
                        
                        print(f"Final est_sources shape: {est_sources.shape}")
                
                except Exception as e:
                    print(f"Error during separation: {e}")
                    # Fallback to splitting the mixture
                    half_point = len(mixture) // 2
                    est_sources = np.array([mixture[:half_point], mixture[half_point:]])
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
                
                # Save separated sources
                print(f"Saving separated sources: {est_sources.shape}")
                sf.write(est1_path, est_sources[0], sr)
                sf.write(est2_path, est_sources[1], sr)
                
                # Verify saved files
                est1, _ = load_audio(est1_path)
                est2, _ = load_audio(est2_path)
                
                print(f"Loaded saved files: {len(est1)}, {len(est2)} samples")
                
                if len(est1) < 8000 or len(est2) < 8000:
                    print(f"Warning: Saved separated files are too short - using fallback")
                    # Fallback to splitting the mixture
                    half_point = len(mixture) // 2
                    sf.write(est1_path, mixture[:half_point], sr)
                    sf.write(est2_path, mixture[half_point:], sr)
                    
                    # Reload
                    est1, _ = load_audio(est1_path)
                    est2, _ = load_audio(est2_path)
                
                # Evaluate separation quality
                reference_sources = [source1, source2]
                estimated_sources = [est1, est2]
                
                # Match lengths for evaluation
                min_len = min([len(s) for s in reference_sources + estimated_sources])
                reference_sources = [s[:min_len] for s in reference_sources]
                estimated_sources = [est1[:min_len], est2[:min_len]]
                
                # Calculate metrics: SDR, SIR, SAR
                metrics = evaluate_separation(reference_sources, estimated_sources)
                
                # Get permutation from metrics
                perm = metrics['permutation']
                est1_idx, est2_idx = perm[0], perm[1]
                
                # Calculate PESQ
                pesq1 = calculate_pesq(reference_sources[0], estimated_sources[est1_idx])
                pesq2 = calculate_pesq(reference_sources[1], estimated_sources[est2_idx])
                
                # Store metrics for averaging
                metrics_summary['sdr'].append(metrics['sdr'])
                metrics_summary['sir'].append(metrics['sir'])
                metrics_summary['sar'].append(metrics['sar'])
                metrics_summary['pesq'].append([pesq1, pesq2])
                
                # Add to metadata
                separation_metadata.append({
                    'mixture': mix_path, 
                    'source1': source1_path, 
                    'source2': source2_path,
                    'estimated1': est1_path, 
                    'estimated2': est2_path,
                    'speaker1': item['speaker1'], 
                    'speaker2': item['speaker2'],
                    'permutation': metrics['permutation'].tolist(),
                    'sdr': metrics['sdr'].tolist(), 
                    'sir': metrics['sir'].tolist(), 
                    'sar': metrics['sar'].tolist(), 
                    'pesq1': pesq1, 
                    'pesq2': pesq2
                })
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save progress after each batch
        pd.DataFrame(separation_metadata).to_csv(metadata_file, index=False)
        print(f"Saved progress: {len(separation_metadata)}/{total_items} items processed")
    
    # Calculate average metrics
    valid_sdr = [m for m in metrics_summary['sdr'] if len(m) > 0 and not np.isnan(m).any()]
    valid_sir = [m for m in metrics_summary['sir'] if len(m) > 0 and not np.isnan(m).any()]
    valid_sar = [m for m in metrics_summary['sar'] if len(m) > 0 and not np.isnan(m).any()]
    valid_pesq = [p for p in metrics_summary['pesq'] if len(p) > 0 and not np.isnan(p).any()]
    
    avg_metrics = {
        'SDR': np.mean([np.mean(m) for m in valid_sdr]) if valid_sdr else 0.0,
        'SIR': np.mean([np.mean(m) for m in valid_sir]) if valid_sir else 0.0,
        'SAR': np.mean([np.mean(m) for m in valid_sar]) if valid_sar else 0.0,
        'PESQ': np.mean([np.mean(p) for p in valid_pesq]) if valid_pesq else 1.0
    }
    
    pd.DataFrame([avg_metrics]).to_csv(metrics_file, index=False)
    
    print("\nSpeaker Separation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"Separation complete.")
    return separation_metadata

#################################################
# Speaker Identification
#################################################

def identify_speakers(separation_metadata, pretrained_model, finetuned_model, feature_extractor, batch_size=10):
    """Identify speakers in separated audio"""
    print("Performing speaker identification on separated audio...")
    
    # Check if results already exist
    results_file = os.path.join(OUTPUT_DIR, "identification_results.csv")
    details_file = os.path.join(OUTPUT_DIR, "identification_details.pkl")
    
    if os.path.exists(results_file) and os.path.exists(details_file):
        try:
            with open(details_file, 'rb') as f:
                results = pickle.load(f)
            print("Found existing identification results")
            return results
        except:
            print("Starting identification from scratch")
    
    # Set models to evaluation mode
    pretrained_model.eval()
    finetuned_model.eval()
    
    # Get speaker IDs
    train_ids, test_ids = get_speaker_ids()
    all_speaker_ids = train_ids + test_ids
    
    # Create reference embeddings for all speakers
    print("Creating reference embeddings for all speakers...")
    reference_embeddings = {}
    
    with tqdm(total=len(all_speaker_ids)) as pbar:
        for speaker_id in all_speaker_ids:
            try:
                # Find audio files for this speaker
                speaker_files = find_speaker_files(speaker_id)
                if not speaker_files:
                    print(f"No files found for speaker {speaker_id}")
                    pbar.update(1)
                    continue
                
                # Calculate embeddings for each file
                speaker_embeddings_pretrained = []
                speaker_embeddings_finetuned = []
                
                for file in speaker_files[:3]:  # Use up to 3 files per speaker
                    waveform, sr = load_audio(file, max_duration=5)
                    
                    if len(waveform) < 8000:
                        continue
                    
                    # Calculate embeddings
                    with torch.no_grad():
                        inputs = feature_extractor(
                            waveform, sampling_rate=16000, return_tensors="pt",
                            padding="max_length", max_length=80000
                        ).input_values.to(DEVICE)
                        
                        pretrained_emb = pretrained_model(inputs)
                        finetuned_emb = finetuned_model(inputs)
                        
                        speaker_embeddings_pretrained.append(pretrained_emb.cpu().numpy())
                        speaker_embeddings_finetuned.append(finetuned_emb.cpu().numpy())
                
                # Average embeddings for the speaker
                if speaker_embeddings_pretrained:
                    avg_pretrained = np.mean(np.vstack(speaker_embeddings_pretrained), axis=0)
                    avg_finetuned = np.mean(np.vstack(speaker_embeddings_finetuned), axis=0)
                    
                    reference_embeddings[speaker_id] = {
                        'pretrained': avg_pretrained,
                        'finetuned': avg_finetuned
                    }
                else:
                    print(f"No valid embeddings for speaker {speaker_id}")
                
                # Clean up GPU memory
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing reference for speaker {speaker_id}: {e}")
            
            pbar.update(1)
    
    print(f"Created reference embeddings for {len(reference_embeddings)} speakers")
    
    # Initialize results
    results = {
        'pretrained': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'detailed': []},
        'finetuned': {'correct': 0, 'total': 0, 'accuracy': 0.0, 'detailed': []},
        'similarity_scores': {'pretrained': [], 'finetuned': []}
    }
    
    # Process separated audio in batches
    total_items = len(separation_metadata)
    
    print("Identifying speakers in separated samples...")
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        print(f"Processing batch {batch_start}-{batch_end} of {total_items}")
        
        for idx in tqdm(range(batch_start, batch_end)):
            try:
                item = separation_metadata[idx]
                
                # Get speaker IDs and permutation
                speaker1 = item['speaker1']
                speaker2 = item['speaker2']
                
                # Handle different permutation formats
                perm = item['permutation']
                if isinstance(perm, str):
                    try:
                        perm = eval(perm)
                    except:
                        perm = [int(p.strip()) for p in perm.strip('[]').split(',')]
                
                # Load separated audio files
                est1_path = item['estimated1']
                est2_path = item['estimated2']
                
                # Check if files exist
                if not os.path.exists(est1_path) or not os.path.exists(est2_path):
                    print(f"Missing separation file(s): {est1_path} or {est2_path}")
                    continue
                
                # Process each separated source
                for est_idx, (source_path, true_speaker) in enumerate([
                        (est1_path, speaker1 if perm[0] == 0 else speaker2), 
                        (est2_path, speaker2 if perm[1] == 1 else speaker1)
                    ]):
                    
                    # Load the audio
                    source, _ = load_audio(source_path, max_duration=5)
                    
                    # Skip if audio is too short
                    if len(source) < 8000:
                        print(f"Audio too short in {source_path}: {len(source)} samples")
                        continue
                    
                    # Skip if true speaker is not in our reference embeddings
                    if true_speaker not in reference_embeddings:
                        print(f"Speaker {true_speaker} not in reference embeddings")
                        continue
                        
                    # Calculate embeddings
                    with torch.no_grad():
                        inputs = feature_extractor(
                            source, sampling_rate=16000, return_tensors="pt",
                            padding="max_length", max_length=80000
                        ).input_values.to(DEVICE)
                        
                        pretrained_emb = pretrained_model(inputs).cpu().numpy()
                        finetuned_emb = finetuned_model(inputs).cpu().numpy()
                        
                        # Calculate similarity with all reference embeddings
                        pretrained_scores = {}
                        for spk_id, embs in reference_embeddings.items():
                            pretrained_ref = embs['pretrained']
                            similarity = F.cosine_similarity(
                                torch.tensor(pretrained_emb), 
                                torch.tensor(pretrained_ref).unsqueeze(0),
                                dim=1
                            ).item()
                            pretrained_scores[spk_id] = similarity
                        
                        finetuned_scores = {}
                        for spk_id, embs in reference_embeddings.items():
                            finetuned_ref = embs['finetuned']
                            similarity = F.cosine_similarity(
                                torch.tensor(finetuned_emb), 
                                torch.tensor(finetuned_ref).unsqueeze(0),
                                dim=1
                            ).item()
                            finetuned_scores[spk_id] = similarity
                        
                        # Get predictions (highest similarity)
                        pretrained_pred = max(pretrained_scores.items(), key=lambda x: x[1])[0]
                        finetuned_pred = max(finetuned_scores.items(), key=lambda x: x[1])[0]
                        
                        # Store similarity scores for diagnosis
                        results['similarity_scores']['pretrained'].append({
                            'true_speaker': true_speaker,
                            'true_score': pretrained_scores.get(true_speaker, -1.0),
                            'pred_speaker': pretrained_pred,
                            'pred_score': pretrained_scores.get(pretrained_pred, -1.0)
                        })
                        
                        results['similarity_scores']['finetuned'].append({
                            'true_speaker': true_speaker,
                            'true_score': finetuned_scores.get(true_speaker, -1.0),
                            'pred_speaker': finetuned_pred,
                            'pred_score': finetuned_scores.get(finetuned_pred, -1.0)
                        })
                        
                        # Check if predictions are correct
                        pretrained_correct = pretrained_pred == true_speaker
                        finetuned_correct = finetuned_pred == true_speaker
                        
                        # Update results
                        results['pretrained']['total'] += 1
                        results['finetuned']['total'] += 1
                        
                        if pretrained_correct:
                            results['pretrained']['correct'] += 1
                        
                        if finetuned_correct:
                            results['finetuned']['correct'] += 1
                        
                        # Store detailed results
                        results['pretrained']['detailed'].append({
                            'source': f"est{est_idx+1}",
                            'true_speaker': true_speaker,
                            'predicted_speaker': pretrained_pred,
                            'similarity': pretrained_scores.get(pretrained_pred, -1.0),
                            'correct': pretrained_correct
                        })
                        
                        results['finetuned']['detailed'].append({
                            'source': f"est{est_idx+1}",
                            'true_speaker': true_speaker,
                            'predicted_speaker': finetuned_pred,
                            'similarity': finetuned_scores.get(finetuned_pred, -1.0),
                            'correct': finetuned_correct
                        })
            
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
        
        # Calculate and save intermediate results
        if results['pretrained']['total'] > 0:
            results['pretrained']['accuracy'] = results['pretrained']['correct'] / results['pretrained']['total']
        
        if results['finetuned']['total'] > 0:
            results['finetuned']['accuracy'] = results['finetuned']['correct'] / results['finetuned']['total']
        
        with open(details_file, 'wb') as f:
            pickle.dump(results, f)
        
        summary = {
            'model': ['pretrained', 'finetuned'],
            'accuracy': [results['pretrained']['accuracy'], results['finetuned']['accuracy']],
            'correct': [results['pretrained']['correct'], results['finetuned']['correct']],
            'total': [results['pretrained']['total'], results['finetuned']['total']]
        }
        
        pd.DataFrame(summary).to_csv(results_file, index=False)
        
        print(f"Intermediate identification results:")
        print(f"  Pre-trained Model: {results['pretrained']['accuracy']:.4f}")
        print(f"  Fine-tuned Model: {results['finetuned']['accuracy']:.4f}")
    
    return results

def plot_identification_results(results):
    """Plot Rank-1 identification accuracy comparison"""
    plt.figure(figsize=(10, 6))
    
    models = ['Pre-trained', 'Fine-tuned']
    accuracies = [
        results['pretrained']['accuracy'] * 100,
        results['finetuned']['accuracy'] * 100
    ]
    
    bars = plt.bar(models, accuracies, color=['blue', 'orange'])
    plt.ylabel('Rank-1 Identification Accuracy (%)')
    plt.title('Speaker Identification Accuracy After Separation')
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylim(0, max(accuracies) * 1.2)  # Add some space above bars
    plt.savefig(os.path.join(OUTPUT_DIR, 'identification_comparison.png'))
    plt.close()
    
    print(f"Rank-1 accuracy comparison plot saved to {os.path.join(OUTPUT_DIR, 'identification_comparison.png')}")

#################################################
# Main Function
#################################################

def main():
    print(f"Using device: {DEVICE}")
    
    # Step 1: Create mixed dataset
    train_metadata_file = os.path.join(MIXED_DIR, 'train', "train_metadata.csv")
    test_metadata_file = os.path.join(MIXED_DIR, 'test', "test_metadata.csv")
    
    if not os.path.exists(train_metadata_file):
        train_mixed_metadata = create_mixed_dataset(split='train', n_pairs=50)
    else:
        print(f"Loading existing train mixed dataset from {train_metadata_file}")
        train_mixed_metadata = pd.read_csv(train_metadata_file).to_dict('records')
    
    if not os.path.exists(test_metadata_file):
        test_mixed_metadata = create_mixed_dataset(split='test', n_pairs=50)
    else:
        print(f"Loading existing test mixed dataset from {test_metadata_file}")
        test_mixed_metadata = pd.read_csv(test_metadata_file).to_dict('records')
    
    # Verify mixed audio files
    print("\nVerifying mixed audio files...")
    valid_test_metadata = []
    for item in test_mixed_metadata:
        mix_path = item['mixture']
        if os.path.exists(mix_path):
            waveform, _ = load_audio(mix_path)
            if len(waveform) >= 16000:  # At least 1 second
                valid_test_metadata.append(item)
            else:
                print(f"Warning: Mixed file too short: {mix_path}")
        else:
            print(f"Warning: Mixed file not found: {mix_path}")
    
    if len(valid_test_metadata) < len(test_mixed_metadata):
        print(f"Using {len(valid_test_metadata)} valid mixed files out of {len(test_mixed_metadata)}")
        test_mixed_metadata = valid_test_metadata
    
    # Step 2: Perform speaker separation and analyze metrics
    try:
        separation_metadata = separate_speakers(test_mixed_metadata, batch_size=5)
    except KeyboardInterrupt:
        print("\nSeparation interrupted. Progress saved.")
        return
    
    # Step 3: Load models from task II
    print("\nLoading speaker verification models...")
    
    # Use 100 speaker IDs to match the original model
    num_speakers = 100  # This must match the model from Task II
    
    # Load feature extractor
    feature_extractor_path = os.path.join(MODEL_DIR, 'feature_extractor.pkl')
    print(f"Loading feature extractor from {feature_extractor_path}")
    
    try:
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
    except Exception as e:
        print(f"Error loading feature extractor: {e}")
        print("Initializing new feature extractor")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    # Load pre-trained model
    pretrained_model_path = os.path.join(MODEL_DIR, 'pretrained_model.pt')
    print(f"Loading pre-trained model from {pretrained_model_path}")
    
    pretrained_model = SpeakerVerificationModel(
        num_classes=num_speakers,
        pretrained_model_name="microsoft/wavlm-base-plus",
        use_lora=False
    ).to(DEVICE)
    
    try:
        # Load model with strict=False to handle parameter mismatches
        state_dict = torch.load(pretrained_model_path, map_location=DEVICE)
        pretrained_model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pre-trained model")
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Using initialized model instead")
    
    # Load fine-tuned model (try different possible paths)
    finetuned_model_paths = [
        os.path.join(MODEL_DIR, 'finetuned_model.pt'),
        os.path.join(MODEL_DIR, 'best_model.pt')
    ]
    
    finetuned_model = SpeakerVerificationModel(
        num_classes=num_speakers,
        pretrained_model_name="microsoft/wavlm-base-plus",
        use_lora=True
    ).to(DEVICE)
    
    finetuned_model_loaded = False
    for model_path in finetuned_model_paths:
        if os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=DEVICE)
                finetuned_model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded fine-tuned model")
                finetuned_model_loaded = True
                break
            except Exception as e:
                print(f"Error loading from {model_path}: {e}")
    
    if not finetuned_model_loaded:
        print("Could not load any fine-tuned model. Using initialized model.")
    
    # Step 4: Perform speaker identification
    if len(separation_metadata) == 0:
        print("No valid separated audio files found for identification.")
        return
        
    try:
        identification_results = identify_speakers(
            separation_metadata, 
            pretrained_model, 
            finetuned_model, 
            feature_extractor,
            batch_size=10
        )
        
        # Report Rank-1 identification accuracies
        print("\nRank-1 Speaker Identification Accuracy:")
        print(f"  Pre-trained model: {identification_results['pretrained']['accuracy']*100:.2f}%")
        print(f"  Fine-tuned model:  {identification_results['finetuned']['accuracy']*100:.2f}%")
        
        # Create comparison plot
        plot_identification_results(identification_results)
        
        # Analyze similarity scores if accuracy is low
        if identification_results['pretrained']['accuracy'] < 0.3 or identification_results['finetuned']['accuracy'] < 0.3:
            print("\nAnalyzing similarity scores to diagnose low accuracy:")
            
            if 'similarity_scores' in identification_results:
                # Analyze pretrained model scores
                pretrained_scores = identification_results['similarity_scores']['pretrained']
                if pretrained_scores:
                    true_scores = [score['true_score'] for score in pretrained_scores]
                    pred_scores = [score['pred_score'] for score in pretrained_scores]
                    print(f"  Pre-trained model average true speaker similarity: {np.mean(true_scores):.4f}")
                    print(f"  Pre-trained model average predicted speaker similarity: {np.mean(pred_scores):.4f}")
                    
                # Analyze finetuned model scores
                finetuned_scores = identification_results['similarity_scores']['finetuned']
                if finetuned_scores:
                    true_scores = [score['true_score'] for score in finetuned_scores]
                    pred_scores = [score['pred_score'] for score in finetuned_scores]
                    print(f"  Fine-tuned model average true speaker similarity: {np.mean(true_scores):.4f}")
                    print(f"  Fine-tuned model average predicted speaker similarity: {np.mean(pred_scores):.4f}")
        
        # Print comprehensive results summary
        print("\n=========== TASK III RESULTS SUMMARY ===========")
        
        # Separation metrics
        separation_metrics_file = os.path.join(SEPARATED_DIR, 'test', "separation_metrics.csv")
        if os.path.exists(separation_metrics_file):
            metrics_df = pd.read_csv(separation_metrics_file)
            print("\n1. Speaker Separation Metrics:")
            for col in metrics_df.columns:
                print(f"  - {col}: {metrics_df[col].values[0]:.4f}")
        
        # Identification results
        print("\n2. Speaker Identification Results:")
        print(f"  - Pre-trained model accuracy: {identification_results['pretrained']['accuracy']*100:.2f}%")
        print(f"  - Fine-tuned model accuracy:  {identification_results['finetuned']['accuracy']*100:.2f}%")
        print(f"  - Total samples evaluated:    {identification_results['pretrained']['total']}")
        
        # Detailed analysis
        print("\n3. Speaker Identification Detailed Analysis:")
        if identification_results['pretrained']['total'] > 0:
            pretrained_correct = identification_results['pretrained']['correct']
            pretrained_total = identification_results['pretrained']['total']
            pretrained_incorrect = pretrained_total - pretrained_correct
            print(f"  Pre-trained model: {pretrained_correct} correct, {pretrained_incorrect} incorrect out of {pretrained_total}")
        
        if identification_results['finetuned']['total'] > 0:
            finetuned_correct = identification_results['finetuned']['correct']
            finetuned_total = identification_results['finetuned']['total']
            finetuned_incorrect = finetuned_total - finetuned_correct
            print(f"  Fine-tuned model: {finetuned_correct} correct, {finetuned_incorrect} incorrect out of {finetuned_total}")
        
        print("\n=================================================")
        
    except KeyboardInterrupt:
        print("\nIdentification interrupted. Partial progress saved.")
    except Exception as e:
        print(f"Error during identification: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()