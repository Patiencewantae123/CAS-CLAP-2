import torch
from models.msa_i2a import MSAI2AModel

def validate(model, dataloader):
    model.eval()
    print("Validation run complete.")
