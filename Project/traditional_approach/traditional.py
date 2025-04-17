import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, wiener
from collections import defaultdict
import hashlib
import sqlite3
import pywt
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import random
import csv
from tqdm import tqdm

# Configuration
DATASET_PATH = "genres_original"  # UPDATE THIS PATH TO YOUR DATASET
SAMPLE_RATE = 22050
WINDOW_SIZE = 4096
HOP_LENGTH = 256
FAN_VALUE = 15
TIME_WINDOW = 200

# Parameters for different fingerprinting techniques
MFCC_N = 13  # Number of MFCCs to extract
WAVELET_TYPE = 'db4'  # Wavelet type for transform
WAVELET_LEVEL = 5  # Decomposition level for wavelet transform

# Database setup
conn = sqlite3.connect('fingerprints.db', timeout=30)
c = conn.cursor()

# Create separate tables for each fingerprinting method
c.execute('''CREATE TABLE IF NOT EXISTS spectrogram_fingerprints
             (hash TEXT, song_id TEXT, offset INTEGER)''')
c.execute('''CREATE INDEX IF NOT EXISTS spectrogram_hash_index
             ON spectrogram_fingerprints (hash)''')

c.execute('''CREATE TABLE IF NOT EXISTS mfcc_fingerprints
             (hash TEXT, song_id TEXT, offset INTEGER)''')
c.execute('''CREATE INDEX IF NOT EXISTS mfcc_hash_index
             ON mfcc_fingerprints (hash)''')

c.execute('''CREATE TABLE IF NOT EXISTS wavelet_fingerprints
             (hash TEXT, song_id TEXT, offset INTEGER)''')
c.execute('''CREATE INDEX IF NOT EXISTS wavelet_hash_index
             ON wavelet_fingerprints (hash)''')

# Create a songs metadata table
c.execute('''CREATE TABLE IF NOT EXISTS songs
             (song_id TEXT PRIMARY KEY, genre TEXT, filename TEXT)''')

# Create a results table for evaluation
c.execute('''CREATE TABLE IF NOT EXISTS evaluation_results
             (method TEXT, noise_type TEXT, snr INTEGER, accuracy REAL, 
              precision REAL, recall REAL, f1 REAL, time_ms INTEGER)''')

def verify_database():
    """Verify database contents for all fingerprinting methods"""
    for method in ["spectrogram", "mfcc", "wavelet"]:
        table = f"{method}_fingerprints"
        c.execute(f"SELECT COUNT(*) FROM {table}")
        count = c.fetchone()[0]
        print(f"{method.capitalize()} database contains {count} hashes")
        
        c.execute(f"SELECT hash, song_id FROM {table} LIMIT 3")
        print(f"Sample entries for {method}:", c.fetchall())
    
    print("\nSongs in database:")
    c.execute("SELECT song_id, genre FROM songs LIMIT 5")
    print(c.fetchall())

def preprocess_audio(file_path):
    """Load and preprocess audio file with advanced noise reduction"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Basic normalization
    y = librosa.util.normalize(y)
    
    # Advanced noise reduction (using Wiener filter)
    y_denoised = wiener(y)
    
    return y_denoised

def apply_spectral_subtraction(y, n_fft=2048, hop_length=512, noise_clip=0.1):
    """Apply spectral subtraction for noise reduction"""
    # Estimate noise from first segment (assuming it's mostly noise)
    noise_sample = y[:int(len(y) * noise_clip)]
    noise_spec = np.abs(librosa.stft(noise_sample, n_fft=n_fft, hop_length=hop_length))
    noise_power = np.mean(noise_spec**2, axis=1, keepdims=True)
    
    # Compute STFT of the signal
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spec_mag = np.abs(spec)
    spec_phase = np.angle(spec)
    
    # Subtract noise power from signal power
    power = spec_mag**2
    power_reduced = np.maximum(power - noise_power, 0.0)
    mag_reduced = np.sqrt(power_reduced)
    
    # Reconstruct signal with reduced noise
    spec_reduced = mag_reduced * np.exp(1j * spec_phase)
    y_reduced = librosa.istft(spec_reduced, hop_length=hop_length)
    
    return y_reduced

def create_spectrogram(y):
    """Create spectrogram with logarithmic amplitude"""
    S = librosa.stft(y, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    S_log = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_log

def extract_mfccs(y, sr=SAMPLE_RATE):
    """Extract MFCCs as mentioned in the proposal"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    # Delta and delta-delta features for better representation
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    # Combine features
    mfcc_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    return mfcc_features

def apply_wavelet_transform(y, wavelet=WAVELET_TYPE, level=WAVELET_LEVEL):
    """Apply wavelet transformation as mentioned in the proposal"""
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(y, wavelet, level=level)
    
    # Process the coefficients for feature extraction
    # We'll use approximation coefficients and detail coefficients
    features = []
    
    # Add approximation coefficients
    features.append(coeffs[0])
    
    # Add detail coefficients
    for i in range(1, len(coeffs)):
        features.append(coeffs[i])
    
    # Convert features to a 2D array
    max_len = max([len(f) for f in features])
    padded_features = []
    
    for f in features:
        # Pad with zeros to make all features the same length
        padded = np.pad(f, (0, max_len - len(f)))
        padded_features.append(padded)
    
    return np.array(padded_features)

def find_spectral_peaks(spectrogram):
    """Find peaks in spectrogram"""
    # Keep only top 20 peaks per time slice
    peaks = []
    for time_idx in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, time_idx]
        freq_peaks = np.argsort(spectrum)[-20:]  # Top 20 peaks
        peaks.extend((freq_idx, time_idx) for freq_idx in freq_peaks)
    return peaks

def find_mfcc_peaks(mfcc_features):
    """Find peaks in MFCC features"""
    peaks = []
    for time_idx in range(mfcc_features.shape[1]):
        mfcc_frame = mfcc_features[:, time_idx]
        # Find local maxima in the MFCC frame
        peak_indices, _ = find_peaks(mfcc_frame)
        if len(peak_indices) == 0:  # If no peaks found, take top values
            peak_indices = np.argsort(mfcc_frame)[-5:]
        peaks.extend((freq_idx, time_idx) for freq_idx in peak_indices)
    return peaks

def find_wavelet_peaks(wavelet_features):
    """Find peaks in wavelet features"""
    peaks = []
    for time_idx in range(wavelet_features.shape[1]):
        wavelet_frame = wavelet_features[:, time_idx]
        # Find local maxima in the wavelet frame
        peak_indices, _ = find_peaks(wavelet_frame)
        if len(peak_indices) == 0:  # If no peaks found, take top values
            peak_indices = np.argsort(wavelet_frame)[-5:]
        peaks.extend((freq_idx, time_idx) for freq_idx in peak_indices)
    return peaks

def generate_hashes(peaks, song_id):
    """Generate hashes from peak pairs using combinatorial approach"""
    hashes = []
    for i in range(len(peaks)):
        for j in range(1, FAN_VALUE):
            if (i + j) < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                delta_time = time2 - time1

                if delta_time <= 0 or delta_time > TIME_WINDOW:
                    continue

                # Create hash from frequency pair and time difference
                hash_str = f"{freq1}|{freq2}|{delta_time}"
                hash_hex = hashlib.sha1(hash_str.encode()).hexdigest()[:20]
                hashes.append((hash_hex, song_id, time1))
    return hashes

def store_fingerprints(hashes, method="spectrogram"):
    """Store hashes in SQLite database for the specified method"""
    table = f"{method}_fingerprints"
    c.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", hashes)
    conn.commit()

def process_file(file_path, song_id, genre):
    """Process a single audio file using all fingerprinting methods"""
    print(f"Processing: {song_id}")
    
    # Basic preprocessing
    y = preprocess_audio(file_path)
    
    # Apply advanced noise reduction
    y_denoised = apply_spectral_subtraction(y)
    
    # Store song metadata
    c.execute("INSERT OR REPLACE INTO songs VALUES (?, ?, ?)", 
              (song_id, genre, os.path.basename(file_path)))
    
    # Method 1: Spectrogram-based fingerprinting
    spectrogram = create_spectrogram(y_denoised)
    spec_peaks = find_spectral_peaks(spectrogram)
    spec_hashes = generate_hashes(spec_peaks, song_id)
    store_fingerprints(spec_hashes, "spectrogram")
    
    # Method 2: MFCC-based fingerprinting
    mfcc_features = extract_mfccs(y_denoised)
    mfcc_peaks = find_mfcc_peaks(mfcc_features)
    mfcc_hashes = generate_hashes(mfcc_peaks, song_id)
    store_fingerprints(mfcc_hashes, "mfcc")
    
    # Method 3: Wavelet-based fingerprinting
    wavelet_features = apply_wavelet_transform(y_denoised)
    # Ensure 2D format for peak finding
    if len(wavelet_features.shape) == 1:
        wavelet_features = wavelet_features.reshape(-1, 1)
    wavelet_peaks = find_wavelet_peaks(wavelet_features)
    wavelet_hashes = generate_hashes(wavelet_peaks, song_id)
    store_fingerprints(wavelet_hashes, "wavelet")

def process_directory(dataset_path):
    """Process entire dataset and populate database"""
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_path, file)
                    song_id = f"{genre}_{file}"
                    process_file(file_path, song_id, genre)

def match_audio(query_path, method="spectrogram"):
    """Match query audio against database using specified method"""
    # Process query audio
    start_time = time.time()
    y = preprocess_audio(query_path)
    
    # Apply noise reduction
    y_denoised = apply_spectral_subtraction(y)
    
    # Process with the selected method
    if method == "spectrogram":
        features = create_spectrogram(y_denoised)
        peaks = find_spectral_peaks(features)
    elif method == "mfcc":
        features = extract_mfccs(y_denoised)
        peaks = find_mfcc_peaks(features)
    elif method == "wavelet":
        features = apply_wavelet_transform(y_denoised)
        # Ensure 2D format for peak finding
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        peaks = find_wavelet_peaks(features)
    else:
        raise ValueError(f"Unknown fingerprinting method: {method}")
    
    query_hashes = generate_hashes(peaks, "query")
    
    # Find matches
    matches = defaultdict(list)
    table = f"{method}_fingerprints"
    
    for hash_hex, _, offset in query_hashes:
        c.execute(f"SELECT song_id, offset FROM {table} WHERE hash=?", (hash_hex,))
        results = c.fetchall()
        for song_id, db_offset in results:
            matches[song_id].append(db_offset - offset)
    
    # Improved histogram analysis
    best_match = None
    best_score = 0
    
    for song_id, offsets in matches.items():
        hist, _ = np.histogram(offsets, bins=range(-TIME_WINDOW, TIME_WINDOW))
        
        # Look for cluster of matches
        top_values = sorted(hist, reverse=True)[:3]
        current_score = sum(top_values)  # Use sum of top 3 bins
        
        if current_score > best_score:
            best_score = current_score
            best_match = song_id
    
    # Add minimum score threshold
    min_score = 15
    elapsed_time = (time.time() - start_time) * 1000  # ms
    
    if best_score >= min_score:
        return best_match, best_score, elapsed_time
    else:
        return None, 0, elapsed_time

def add_noise(audio, noise_type='gaussian', snr_db=10):
    """Add different types of noise to audio for testing robustness"""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    if noise_type == 'gaussian':
        # Gaussian white noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    
    elif noise_type == 'pink':
        # Pink noise (1/f spectrum)
        white_noise = np.random.normal(0, 1, len(audio))
        # Create pink noise by applying 1/f filter
        f = np.fft.rfftfreq(len(white_noise))
        f[0] = 1  # Prevent division by zero
        pink_filter = 1 / np.sqrt(f)
        pink_filter[0] = 0  # Remove DC component
        pink_noise_fft = np.fft.rfft(white_noise) * pink_filter
        pink_noise = np.fft.irfft(pink_noise_fft)
        # Normalize and scale
        pink_noise = pink_noise / np.std(pink_noise) * np.sqrt(noise_power)
        noise = pink_noise
    
    elif noise_type == 'environment':
        # Simulate environmental noise by mixing multiple frequencies
        t = np.arange(len(audio)) / SAMPLE_RATE
        # Mix of sine waves at different frequencies
        noise = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.25 * np.sin(2 * np.pi * 300 * t)
        noise = noise / np.std(noise) * np.sqrt(noise_power)
    
    elif noise_type == 'music':
        # Simulate interfering music by adding random chords
        t = np.arange(len(audio)) / SAMPLE_RATE
        frequencies = [261.63, 329.63, 392.00]  # C4, E4, G4 (C major chord)
        noise = np.zeros_like(audio)
        for freq in frequencies:
            noise += np.sin(2 * np.pi * freq * t)
        noise = noise / np.std(noise) * np.sqrt(noise_power)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise

def evaluate_method(method, test_files, noise_types, snr_values):
    """Evaluate a single fingerprinting method with different noise types and SNR levels"""
    results = []
    
    for noise_type in noise_types:
        for snr in snr_values:
            print(f"Evaluating {method} with {noise_type} noise at {snr} dB SNR...")
            
            correct = 0
            total = 0
            total_time = 0
            
            # Store true and predicted labels for precision/recall calculation
            y_true = []
            y_pred = []
            
            for file_path, true_song_id in test_files:
                # Add noise
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                noisy_audio = add_noise(y, noise_type=noise_type, snr_db=snr)
                
                # Save temporary file
                temp_file = f"temp_query_{noise_type}_{snr}.wav"
                sf.write(temp_file, noisy_audio, SAMPLE_RATE)
                
                # Match
                predicted_song_id, score, match_time = match_audio(temp_file, method=method)
                total_time += match_time
                
                # Record results
                total += 1
                y_true.append(true_song_id)
                y_pred.append(predicted_song_id if predicted_song_id else "unknown")
                
                if predicted_song_id == true_song_id:
                    correct += 1
            
            # Calculate metrics
            accuracy = correct / total if total > 0 else 0
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            avg_time = total_time / total if total > 0 else 0
            
            # Store result
            result = (method, noise_type, snr, accuracy, precision, recall, f1, avg_time)
            results.append(result)
            
            # Store in database
            c.execute('''INSERT INTO evaluation_results 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', result)
            conn.commit()
            
            print(f"  Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Time: {avg_time:.1f}ms")
    
    return results

def comprehensive_evaluation():
    """Run comprehensive evaluation on all methods with varied conditions"""
    # Get test files (sample of songs from database)
    c.execute("SELECT song_id, filename, genre FROM songs")
    all_songs = c.fetchall()
    
    # Sample test files (at least one from each genre)
    genres = set(song[2] for song in all_songs)
    test_files = []
    
    for genre in genres:
        genre_songs = [song for song in all_songs if song[2] == genre]
        # Use up to 2 songs per genre for testing
        for song in random.sample(genre_songs, min(2, len(genre_songs))):
            song_id, filename, _ = song
            file_path = os.path.join(DATASET_PATH, genre, filename)
            test_files.append((file_path, song_id))
    
    # Define noise types and SNR values for testing
    noise_types = ['gaussian', 'pink', 'environment', 'music']
    snr_values = [30, 20, 10, 5]  # SNR in dB
    
    # Evaluate each method
    methods = ['spectrogram', 'mfcc', 'wavelet']
    all_results = {}
    
    for method in methods:
        print(f"\nEvaluating {method} fingerprinting method...")
        results = evaluate_method(method, test_files, noise_types, snr_values)
        all_results[method] = results
    
    # Generate summary report
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Test files: {len(test_files)}")
    print(f"Noise types: {noise_types}")
    print(f"SNR values: {snr_values}")
    
    # Export results to CSV
    with open('fingerprinting_evaluation_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Noise Type', 'SNR (dB)', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time (ms)'])
        
        for method, results in all_results.items():
            for result in results:
                writer.writerow(result)
    
    print("\nResults exported to 'fingerprinting_evaluation_results.csv'")
    
    # Plot comparison graph
    plot_evaluation_results(all_results)
    
    return all_results

def plot_evaluation_results(results):
    """Plot evaluation results for visualization"""
    # Create a figure for accuracy comparison across methods
    plt.figure(figsize=(12, 8))
    
    methods = list(results.keys())
    noise_types = ['gaussian', 'pink', 'environment', 'music']
    snr_values = [30, 20, 10, 5]
    
    # Plot a line for each method and noise type
    for method in methods:
        for noise_type in noise_types:
            # Extract data for this method and noise type
            method_data = [r for r in results[method] if r[1] == noise_type]
            
            # Sort by SNR
            method_data.sort(key=lambda x: x[2])
            
            # Extract SNR and accuracy values
            snrs = [r[2] for r in method_data]
            accuracies = [r[3] for r in method_data]
            
            # Plot line
            plt.plot(snrs, accuracies, marker='o', label=f"{method} - {noise_type}")
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('Fingerprinting Method Accuracy vs. Noise Level')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xticks(snr_values)
    plt.savefig('fingerprinting_comparison.png')
    plt.close()
    
    # Create a figure for time comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for bar chart
    avg_times = []
    method_labels = []
    
    for method in methods:
        times = [r[7] for r in results[method]]
        avg_time = np.mean(times)
        avg_times.append(avg_time)
        method_labels.append(method)
    
    plt.bar(method_labels, avg_times)
    plt.xlabel('Fingerprinting Method')
    plt.ylabel('Average Matching Time (ms)')
    plt.title('Average Processing Time by Method')
    plt.savefig('fingerprinting_timing.png')
    plt.close()

def visualize_fingerprints(file_path):
    """Visualize different fingerprinting methods for a sample file"""
    print(f"Visualizing fingerprints for {file_path}...")
    
    # Load and preprocess the audio
    y = preprocess_audio(file_path)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    # Spectrogram visualization
    spectrogram = create_spectrogram(y)
    img = librosa.display.specshow(spectrogram, x_axis='time', y_axis='log', 
                                  sr=SAMPLE_RATE, hop_length=HOP_LENGTH, ax=axs[0])
    peaks = find_spectral_peaks(spectrogram)
    peak_freqs = [p[0] for p in peaks]
    peak_times = [p[1] * HOP_LENGTH / SAMPLE_RATE for p in peaks]
    axs[0].scatter(peak_times, peak_freqs, color='r', s=5, alpha=0.5)
    axs[0].set_title('Spectrogram with Peaks')
    
    # MFCC visualization
    mfcc_features = extract_mfccs(y)
    librosa.display.specshow(mfcc_features, x_axis='time', ax=axs[1])
    mfcc_peaks = find_mfcc_peaks(mfcc_features)
    mfcc_peak_freqs = [p[0] for p in mfcc_peaks]
    mfcc_peak_times = [p[1] * HOP_LENGTH / SAMPLE_RATE for p in mfcc_peaks]
    axs[1].scatter(mfcc_peak_times, mfcc_peak_freqs, color='r', s=5, alpha=0.5)
    axs[1].set_title('MFCC Features with Peaks')
    
    # Wavelet visualization
    wavelet_features = apply_wavelet_transform(y)
    if len(wavelet_features.shape) == 1:
        wavelet_features = wavelet_features.reshape(-1, 1)
    librosa.display.specshow(wavelet_features, x_axis='time', ax=axs[2])
    wavelet_peaks = find_wavelet_peaks(wavelet_features)
    wavelet_peak_freqs = [p[0] for p in wavelet_peaks]
    wavelet_peak_times = [p[1] * HOP_LENGTH / SAMPLE_RATE for p in wavelet_peaks]
    axs[2].scatter(wavelet_peak_times, wavelet_peak_freqs, color='r', s=5, alpha=0.5)
    axs[2].set_title('Wavelet Features with Peaks')
    
    plt.tight_layout()
    plt.savefig('fingerprint_visualization.png')
    plt.close()
    
    print("Visualization saved as 'fingerprint_visualization.png'")

def clear_database():
    """Clear all database tables"""
    tables = [
        "spectrogram_fingerprints",
        "mfcc_fingerprints",
        "wavelet_fingerprints",
        "songs",
        "evaluation_results"
    ]
    for table in tables:
        c.execute(f"DELETE FROM {table}")
    conn.commit()
    print("Database cleared successfully")

def main():
    # Clear database for fresh start (comment out if you want to keep existing data)
    clear_database()
    
    # Process the dataset
    print("Processing dataset...")
    process_directory(DATASET_PATH)
    
    # Verify database contents
    print("\nVerifying database...")
    verify_database()
    
    # Visualize fingerprints for a sample file
    sample_genre = os.listdir(DATASET_PATH)[0]
    sample_files = os.listdir(os.path.join(DATASET_PATH, sample_genre))
    sample_file = next((f for f in sample_files if f.endswith('.wav')), None)
    
    if sample_file:
        sample_path = os.path.join(DATASET_PATH, sample_genre, sample_file)
        visualize_fingerprints(sample_path)
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluation_results = comprehensive_evaluation()
    
    # Close database connection
    conn.close()
    
    print("\nAudio fingerprinting system evaluation completed.")
    print("Results are available in 'fingerprinting_evaluation_results.csv'")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()