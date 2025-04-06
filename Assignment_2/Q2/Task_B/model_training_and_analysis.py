import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_mfcc_features(feature_file):
    """
    Load MFCC features saved from Task A
    
    Parameters:
    -----------
    feature_file : str
        Path to the saved features file
        
    Returns:
    --------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
    """
    try:
        with open(feature_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

def prepare_data(mfcc_dict):
    """
    Prepare data for classification
    
    Parameters:
    -----------
    mfcc_dict : dict
        Dictionary mapping languages to lists of MFCC features
        
    Returns:
    --------
    X : ndarray
        Feature matrix
    y : ndarray
        Target labels
    """
    X = []
    y = []
    
    # Process each language
    for lang, mfccs_list in mfcc_dict.items():
        for mfccs in mfccs_list:
            # Extract features from MFCC
            # Use mean of each coefficient across time as features
            features = np.mean(mfccs, axis=1)
            X.append(features)
            y.append(lang)
    
    return np.array(X), np.array(y)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models
    
    Parameters:
    -----------
    X_train, X_test : ndarray
        Training and testing feature matrices (already scaled)
    y_train, y_test : ndarray
        Training and testing target labels
        
    Returns:
    --------
    results : dict
        Dictionary containing model results
    """
    # Define models to evaluate
    models = {
        'Support Vector Machine': SVC(kernel='rbf', gamma='scale'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        print(f"\nClassification Report for {name}:\n{report}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    return results

def plot_confusion_matrices(results, class_names):
    """
    Plot confusion matrices for all models in a 1x3 grid
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    class_names : list
        List of class names
    """
    # Create a figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Get the model names
    model_names = list(results.keys())
    
    # Plot confusion matrix for each model
    for i, name in enumerate(model_names):
        cm = results[name]['confusion_matrix']
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        
        if i == 0:
            axes[i].set_ylabel('True Label')
        else:
            axes[i].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_comparison(results):
    """
    Plot accuracy comparison between models
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    """
    # Extract model names and accuracies
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    task_a_dir = r"C:\Users\haris\Downloads\su_as2\Q2\Task_A"
    feature_file = os.path.join(task_a_dir, "mfcc_features_4k.pkl")
    
    print("Language Classification Model Training and Evaluation")
    print("====================================================")
    
    # Step 1: Load MFCC features from Task A
    print(f"\nLoading MFCC features from {feature_file}...")
    mfcc_dict = load_mfcc_features(feature_file)
    
    if mfcc_dict is None:
        print("ERROR: Could not load MFCC features. Make sure Task A has been completed.")
        return
    
    languages = list(mfcc_dict.keys())
    print(f"Loaded features for {len(languages)} languages: {', '.join(languages)}")
    
    # Step 2: Prepare data for classification
    print("\nPreparing data for classification...")
    X, y = prepare_data(mfcc_dict)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target labels shape: {y.shape}")
    
    # Step 3: Split data into training and testing sets
    print("\nSplitting data into training and testing sets (75:25)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 4: Scale the data once for all models
    print("\nScaling the feature data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaling complete")
    
    # Step 5: Train and evaluate models
    print("\nTraining and evaluating classification models...")
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Visualize results
    print("\nVisualizing results...")
    plot_confusion_matrices(results, languages)
    plot_accuracy_comparison(results)
    
    print("\nModel training and evaluation complete!")
    
    # Print out best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

if __name__ == "__main__":
    main()