"""
NeuroTopology - Day 3: Training Utilities, Metrics, Model Save/Load
"""

import numpy as np
import json
from typing import Dict, List, Optional, Callable
from .tensor import Tensor
from .nn import Module


class Trainer:
    """Training loop with progress tracking."""
    
    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer,
        metrics: Optional[List[str]] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def fit(
        self,
        train_loader,
        val_loader = None,
        epochs: int = 10,
        verbose: int = 1,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        callbacks = callbacks or []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            train_correct = 0
            train_total = 0
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                x_tensor = Tensor(X_batch)
                logits = self.model(x_tensor)
                loss = self.loss_fn(logits, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                
                if 'accuracy' in self.metrics:
                    preds = np.argmax(logits.data, axis=1)
                    train_correct += np.sum(preds == y_batch)
                    train_total += len(y_batch)
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            if 'accuracy' in self.metrics:
                train_acc = train_correct / train_total
                self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                if 'accuracy' in self.metrics:
                    self.history['val_acc'].append(val_acc)
            
            for cb in callbacks:
                cb(self, epoch)
            
            if verbose > 0:
                msg = f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f}"
                if 'accuracy' in self.metrics:
                    msg += f" | Train Acc: {train_acc:.4f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                    if 'accuracy' in self.metrics:
                        msg += f" | Val Acc: {val_acc:.4f}"
                print(msg)
        
        return self.history
    
    def evaluate(self, data_loader):
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        
        for X_batch, y_batch in data_loader:
            x_tensor = Tensor(X_batch)
            logits = self.model(x_tensor)
            loss = self.loss_fn(logits, y_batch)
            losses.append(loss.item())
            
            if 'accuracy' in self.metrics:
                preds = np.argmax(logits.data, axis=1)
                correct += np.sum(preds == y_batch)
                total += len(y_batch)
        
        avg_loss = np.mean(losses)
        acc = correct / total if total > 0 else 0
        return avg_loss, acc
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(Tensor(X))
        return np.argmax(logits.data, axis=1)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, trainer, epoch):
        current_loss = trainer.history['val_loss'][-1] if trainer.history['val_loss'] else float('inf')
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = self._get_weights(trainer)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            if self.restore_best and self.best_weights:
                self._set_weights(trainer, self.best_weights)
            return True
        return False
    
    def _get_weights(self, trainer):
        return [p.data.copy() for p in trainer.model.parameters()]
    
    def _set_weights(self, trainer, weights):
        for p, w in zip(trainer.model.parameters(), weights):
            p.data = w


class ModelCheckpoint:
    def __init__(self, filepath: str = 'best_model.npz', monitor: str = 'val_loss', mode: str = 'min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, trainer, epoch):
        current = trainer.history[self.monitor][-1] if trainer.history[self.monitor] else None
        if current is None:
            return
        
        improved = (self.mode == 'min' and current < self.best_value) or \
                   (self.mode == 'max' and current > self.best_value)
        
        if improved:
            self.best_value = current
            save_model(trainer.model, self.filepath)
            print(f"  -> Saved best model (epoch {epoch + 1})")


def save_model(model: Module, filepath: str):
    weights = {}
    for i, p in enumerate(model.parameters()):
        weights[f'param_{i}'] = p.data
    np.savez(filepath, **weights)
    print(f"Model saved to {filepath}")


def load_model(model: Module, filepath: str):
    weights = np.load(filepath)
    for i, p in enumerate(model.parameters()):
        p.data = weights[f'param_{i}'].copy()
    print(f"Model loaded from {filepath}")
    return model


def save_history(history: Dict, filepath: str = 'history.json'):
    with open(filepath, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    print(f"History saved to {filepath}")


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred == y_true)


def precision_recall_f1(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> Dict:
    metrics = {}
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[c] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    macro_f1 = np.mean([m['f1'] for m in metrics.values()])
    metrics['macro_f1'] = macro_f1
    return metrics