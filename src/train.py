"""
Train the models!

3 cases:
    * development/debugging/testing
    * hyper-parameter tuning
    * actual training
"""
import pytorch_lighting as pl

from models import ModelTools

def hyperparameter_trial(
    model: pl.LightningModule,
    model_tools: ModelTools,
) -> float:
    """
    Train `model` with `model_tools` and return the final loss value.
    Hyperparameter tuning will then aim to minimise this value.
    """
    
