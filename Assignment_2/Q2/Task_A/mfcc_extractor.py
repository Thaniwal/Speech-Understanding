import os
import numpy as np
import librosa
import glob
from tqdm import tqdm
import pickle

def extract_mfcc_features(audio_path, sr=22050, n_mfcc=13, target_len=None):
    """
    Extract MFCC features from an audio file
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    sr : int
        Sampling rate for audio loading
    n_mfcc : int
        Number of MFCC coefficients to extract
    target_len : int, optional
        Target length for padding/truncating the MFCCs
        
    Returns:
    --------
    mfccs : ndarray
        MFCC features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Standardize length if target_len is provided
        if target_len is not None:
            if mfccs.shape[1] > target_len:
                # Truncate
                mfccs = mfccs[:, :target_len]
            elif mfccs.shape[1] < target_len:
                # Pad with zeros
                padding = np.zeros((mfccs.shape[0], target_len - mfccs.shape[1]))
                mfccs = np.hstack((mfccs, padding))
        
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCCs from {audio_path}: {e}")
        return None

def collect_audio_files(languages, base_dir):
    """
    Collect audio files for selected languages
    
    Parameters:
    -----------
    languages : list
        List of language names to collect audio files for
    base_dir : str
        Base directory containing language folders
        
    Returns:
    --------
    file_dict : dict
        Dictionary mapping languages to lists of audio file paths
    """
    file_dict = {}
    
    # Print the absolute path we're searching in
    abs_base_dir = os.path.abspath(base_dir)
    print(f"Searching for audio files in: {abs_base_dir}")
    
    # List all directories in the base directory
    try:
        available_dirs = [d for d in os.listdir(abs_base_dir) 
                         if os.path.isdir(os.path.join(abs_base_dir, d))]
        print(f"Available language directories: {', '.join(available_dirs)}")
    except FileNotFoundError:
        print(f"ERROR: Base directory not found: {abs_base_dir}")
        return file_dict
    
    for lang in languages:
        lang_dir = os.path.join(abs_base_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Directory for {lang} not found at {lang_dir}")
            continue
            
        # Try multiple audio extensions
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
            audio_files.extend(glob.glob(os.path.join(lang_dir, ext)))
        
        if not audio_files:
            print(f"No audio files found for {lang}")
            continue
            
        file_dict[lang] = audio_files
        print(f"Found {len(audio_files)} audio files for {lang}")
    
    return file_dict

def process_audio_files(file_dict, n_samples=None, n_mfcc=13):
    """
    Process audio files to extract MFCC features
    
    Parameters:
    -----------
    file_dict : dict
        Dictionary mapping languages to lists of audio file paths
    n_samples : int, optional
        Number of samples to process per language
    n_mfcc : int
        Number of MFCC coefficients to extract
        
    Returns:
    --------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
    """
    mfcc_dict = {}
    
    # First pass to determine the average MFCC length
    print("Determining appropriate MFCC standardization length...")
    lengths = []
    for lang, files in file_dict.items():
        # Sample a few files to determine average length
        sample_files = files[:min(3, len(files))]
        for file_path in sample_files:
            try:
                y, sr = librosa.load(file_path)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                lengths.append(mfccs.shape[1])
            except Exception:
                pass
    
    # Calculate target length (median of observed lengths)
    if lengths:
        target_len = int(np.median(lengths))
        print(f"Using target MFCC length of {target_len} frames")
    else:
        target_len = 100  # Default if we couldn't determine
        print(f"Unable to determine average length, using default of {target_len} frames")
    
    # Second pass to extract standardized MFCCs
    for lang, files in file_dict.items():
        print(f"Processing {lang} audio files...")
        
        # Limit samples if specified
        if n_samples and n_samples < len(files):
            files = files[:n_samples]
        
        # Extract MFCCs for each file
        mfccs_list = []
        for file_path in tqdm(files):
            mfccs = extract_mfcc_features(file_path, n_mfcc=n_mfcc, target_len=target_len)
            if mfccs is not None:
                mfccs_list.append(mfccs)
        
        if mfccs_list:
            mfcc_dict[lang] = mfccs_list
            print(f"Extracted MFCCs from {len(mfccs_list)} audio files for {lang}")
        else:
            print(f"No MFCCs extracted for {lang}")
    
    return mfcc_dict

def save_features(mfcc_dict, output_file):
    """
    Save extracted MFCC features to a file
    
    Parameters:
    -----------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
    output_file : str
        Path to save the features
    """
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(mfcc_dict, f)
        print(f"Features saved to {output_file}")
    except Exception as e:
        print(f"Error saving features: {e}")

def main():
    # Configuration
    base_dir = os.path.expanduser("~/Downloads/archive/Language_Detection_Dataset")
    selected_languages = ['Hindi', 'Malayalam', 'Urdu', 'Bengali']
    n_mfcc = 13  # Number of MFCC coefficients
    n_samples_per_language = 20  # Number of samples to process per language
    output_file = "mfcc_features.pkl"
    
    # Step 1: Collect audio files for selected languages
    print("\n======= MFCC Feature Extraction =======\n")
    print("Collecting audio files...")
    file_dict = collect_audio_files(selected_languages, base_dir)
    
    # Check if we found any audio files
    total_files = sum(len(files) for files in file_dict.values())
    if total_files == 0:
        print("\nERROR: No audio files found! Please check the dataset path.")
        print(f"Current path: {os.path.abspath(base_dir)}")
        return None
    
    # Step 2: Process audio files to extract MFCC features
    print("\nExtracting MFCC features...")
    mfcc_dict = process_audio_files(file_dict, 
                                   n_samples=n_samples_per_language, 
                                   n_mfcc=n_mfcc)
    
    # Step 3: Save features
    save_features(mfcc_dict, output_file)
    
    print("\nFeature extraction complete!")
    return mfcc_dict

if __name__ == "__main__":
    mfcc_features = main()