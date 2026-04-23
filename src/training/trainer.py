import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from typing import Dict, Any, Optional

class Trainer:
    """
    Generic training loop for PyTorch classification models with early stopping.
    """
    def __init__(self, 
                 model: nn.Module, 
                 criterion: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 device: str,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 early_stopping_patience: int = 5):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = early_stopping_patience
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def _calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data)
        return corrects.double().item() / labels.size(0)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return {'loss': epoch_loss, 'acc': epoch_acc}

    def val_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc="Validation")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return {'loss': epoch_loss, 'acc': epoch_acc}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> nn.Module:
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            print(f"\\nEpoch {epoch+1}/{epochs}")
            print("-" * 10)
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.val_epoch(val_loader)
            
            if self.scheduler:
                # Assuming ReduceLROnPlateau based on val_loss
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f}")
            
            # Early stopping and model saving
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                print(">> Saving new best model.")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(">> Early stopping triggered.")
                    break
                    
        print(f"\\nTraining complete. Best val_loss: {best_loss:.4f}")
        self.model.load_state_dict(best_model_wts)
        return self.model
