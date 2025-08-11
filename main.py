from SELFRec import SELFRec
from util.conf import ModelConf
import time
import random
import numpy as np
import torch
import os


def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")


def set_seed(seed=42):
    """Set all random seeds to ensure experiment reproducibility"""
    random.seed(seed)           # Python built-in random module
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single card)
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-card (if used)
    
    # Ensure CUDA operations are deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python environment variables (optional)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Usage example



if __name__ == '__main__':
    set_seed(2025)
    models = {
        'Graph-Based Baseline Models': ['CNSDiff']
    }

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    model = 'CNSDiff'

    s = time.time()
    all_models = sum(models.values(), [])
    if model in all_models:
        conf = ModelConf(f'./conf/{model}.yaml')
        rec = SELFRec(conf)
        rec.execute()
        e = time.time()
        print(f"Running time: {e - s:.2f} s")
    else:
        print('Wrong model name!')
        exit(-1)
