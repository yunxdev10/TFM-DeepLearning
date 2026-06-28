import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from typing import Dict, List, Any
from torch.utils.data import DataLoader

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes classification metrics for evaluation.
    
    Args:
        y_true: Ground truth labels (1D array)
        y_pred: Predicted labels (1D array)
        y_prob: Predicted probabilities for AUC-ROC (2D array: [n_samples, n_classes])
        
    Returns:
        Dictionary containing calculated metrics.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Macro and Weighted F1
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if y_prob is not None:
        try:
            # Multi-class AUC-ROC using One-vs-Rest strategy
            metrics['auc_roc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            metrics['auc_roc_macro'] = None
            
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['classification_report'] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    return metrics

def predict_classification(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    """
    Runs inference over a classification dataloader and returns labels,
    predicted classes, and class probabilities.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    return {
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'y_prob': np.array(y_prob),
    }
