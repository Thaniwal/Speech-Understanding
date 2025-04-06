import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from tqdm import tqdm
import pickle
from mfcc_extractor import collect_audio_files, process_audio_files, extract_mfcc_features

def visualize_mfcc_grid(mfcc_dict, n_samples_per_language=3):
    """
    Visualize MFCC spectrograms in a grid layout
    
    Parameters:
    -----------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
    n_samples_per_language : int
        Number of samples to visualize per language
    """
    languages = list(mfcc_dict.keys())
    n_languages = len(languages)
    
    # Create a 4x3 grid (4 languages, 3 samples each)
    fig, axes = plt.subplots(n_languages, n_samples_per_language, 
                            figsize=(15, 4*n_languages))
    
    for i, lang in enumerate(languages):
        # Select a few representative samples
        if len(mfcc_dict[lang]) >= n_samples_per_language:
            # Evenly spaced indices for diverse selection
            indices = np.linspace(0, len(mfcc_dict[lang])-1, n_samples_per_language, dtype=int)
            
            for j, idx in enumerate(indices):
                mfccs = mfcc_dict[lang][idx]
                
                # Plot on the appropriate subplot
                if n_languages > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                
                img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
                ax.set_title(f"{lang} - Sample {j+1}")
                if j == 0:
                    ax.set_ylabel(lang)
                else:
                    ax.set_ylabel("")
    
    plt.tight_layout()
    plt.savefig("mfcc_spectrograms_grid.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_statistics(mfcc_dict):
    """
    Calculate mean and standard deviation of MFCC coefficients for each language
    
    Parameters:
    -----------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
        
    Returns:
    --------
    stats_df : DataFrame
        DataFrame containing mean and std statistics for each language
    """
    stats = {}
    
    for lang, mfccs_list in mfcc_dict.items():
        # Stack all MFCCs for this language
        all_mfccs = np.array(mfccs_list)
        
        # Calculate mean and std for each coefficient
        # First average over time dimension (axis=2), then over samples (axis=0)
        mean_values = np.mean(np.mean(all_mfccs, axis=2), axis=0)
        std_values = np.std(np.mean(all_mfccs, axis=2), axis=0)
        
        stats[lang] = {
            'mean': mean_values,
            'std': std_values
        }
    
    # Create DataFrames for easier visualization
    mean_df = pd.DataFrame({lang: stats[lang]['mean'] for lang in stats.keys()})
    std_df = pd.DataFrame({lang: stats[lang]['std'] for lang in stats.keys()})
    
    # Add index names for coefficients
    mean_df.index = [f'MFCC_{i+1}' for i in range(mean_df.shape[0])]
    std_df.index = [f'MFCC_{i+1}' for i in range(std_df.shape[0])]
    
    return {'mean': mean_df, 'std': std_df}

def visualize_statistics(stats_dict):
    """
    Visualize mean and standard deviation of MFCC coefficients
    
    Parameters:
    -----------
    stats_dict : dict
        Dictionary containing DataFrames with mean and std statistics
    """
    # Plot means
    plt.figure(figsize=(12, 6))
    stats_dict['mean'].plot(kind='bar')
    plt.title('Mean MFCC Coefficients Across Languages')
    plt.ylabel('Mean Value')
    plt.xlabel('MFCC Coefficient')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("mfcc_mean_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot standard deviations
    plt.figure(figsize=(12, 6))
    stats_dict['std'].plot(kind='bar')
    plt.title('Standard Deviation of MFCC Coefficients Across Languages')
    plt.ylabel('Standard Deviation')
    plt.xlabel('MFCC Coefficient')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("mfcc_std_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create heatmap for mean values
    plt.figure(figsize=(10, 8))
    sns.heatmap(stats_dict['mean'], annot=True, cmap='viridis', fmt='.2f')
    plt.title('Heatmap of Mean MFCC Coefficients')
    plt.tight_layout()
    plt.savefig("mfcc_mean_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

def statistical_significance_tests(mfcc_dict):
    """
    Perform statistical tests to quantify differences between languages
    
    Parameters:
    -----------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    languages = list(mfcc_dict.keys())
    n_langs = len(languages)
    n_mfcc = mfcc_dict[languages[0]][0].shape[0]  # Number of MFCC coefficients
    
    # Prepare data for tests
    # Average MFCCs over time for each sample
    avg_mfccs = {}
    for lang, mfccs_list in mfcc_dict.items():
        avg_mfccs[lang] = np.array([np.mean(mfcc, axis=1) for mfcc in mfccs_list])
    
    # Perform ANOVA for each coefficient
    print("\nANOVA Results (Testing if at least one language has different means):")
    anova_results = {}
    for i in range(n_mfcc):
        groups = [avg_mfccs[lang][:, i] for lang in languages]
        f_val, p_val = f_oneway(*groups)
        anova_results[f'MFCC_{i+1}'] = {
            'F-value': f_val,
            'p-value': p_val,
            'significant': p_val < 0.05
        }
        print(f"MFCC_{i+1}: F={f_val:.4f}, p={p_val:.4f}, Significant: {p_val < 0.05}")
    
    # Perform t-tests between each pair of languages
    print("\nPairwise t-test Results:")
    p_values = pd.DataFrame(index=languages, columns=languages)
    
    # Initialize with NaN values
    for i in range(n_langs):
        for j in range(n_langs):
            p_values.iloc[i, j] = np.nan
    
    # Perform t-tests for each pair of languages
    for i in range(n_langs):
        for j in range(i+1, n_langs):
            lang1 = languages[i]
            lang2 = languages[j]
            
            # Calculate mean p-value across all coefficients
            p_vals = []
            for coef in range(n_mfcc):
                _, p_val = ttest_ind(
                    avg_mfccs[lang1][:, coef], 
                    avg_mfccs[lang2][:, coef], 
                    equal_var=False  # Welch's t-test for unequal variances
                )
                p_vals.append(p_val)
            
            mean_p = np.mean(p_vals)
            p_values.loc[lang1, lang2] = mean_p
            p_values.loc[lang2, lang1] = mean_p
            
            print(f"{lang1} vs {lang2}: Mean p-value = {mean_p:.6f}, Significant: {mean_p < 0.05}")
    
    # Fill diagonal with 1.0 (no difference with itself)
    for lang in languages:
        p_values.loc[lang, lang] = 1.0
    
    # Visualize p-values as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_values.astype(float), annot=True, cmap='coolwarm_r', vmin=0, vmax=0.05, fmt='.5f')
    plt.title('Mean p-values from t-tests between language pairs')
    plt.tight_layout()
    plt.savefig("language_pairwise_tests.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {'anova': anova_results, 'pairwise': p_values}

def main():
    # Configuration
    base_dir = os.path.expanduser("~/Downloads/archive/Language_Detection_Dataset")
    selected_languages = ['Hindi', 'Malayalam', 'Urdu', 'Bengali']
    n_mfcc = 13
    n_samples_per_language = 4000
    feature_file = "mfcc_features_4k.pkl"
    
    print("MFCC Feature Extraction and Analysis")
    print("====================================")
    
    # Step 1: Collect audio files for selected languages
    print("\nCollecting audio files...")
    file_dict = collect_audio_files(selected_languages, base_dir)
    
    # Check if we found any audio files
    total_files = sum(len(files) for files in file_dict.values())
    if total_files == 0:
        print("\nERROR: No audio files found! Please check the dataset path.")
        return
    
    # Step 2: Extract MFCC features
    print(f"\nExtracting MFCC features ({n_samples_per_language} samples per language)...")
    mfcc_dict = process_audio_files(file_dict, n_samples=n_samples_per_language, n_mfcc=n_mfcc)
    
    # Step 3: Save features for Task B
    print(f"\nSaving MFCC features to {feature_file}...")
    try:
        with open(feature_file, 'wb') as f:
            pickle.dump(mfcc_dict, f)
        print("Features saved successfully.")
    except Exception as e:
        print(f"Error saving features: {e}")
    
    # Step 4: Visualize MFCC spectrograms for representative samples
    print("\nVisualizing MFCC spectrograms...")
    visualize_mfcc_grid(mfcc_dict, n_samples_per_language=3)
    
    # Step 5: Calculate and visualize statistics
    print("\nCalculating MFCC statistics...")
    stats_dict = calculate_statistics(mfcc_dict)
    visualize_statistics(stats_dict)
    
    # Step 6: Perform statistical significance tests
    print("\nPerforming statistical significance tests...")
    test_results = statistical_significance_tests(mfcc_dict)
    
    print("\nAnalysis complete! MFCC features saved for use in Task B.")
    print(f"Feature file: {os.path.abspath(feature_file)}")

if __name__ == "__main__":
    main()