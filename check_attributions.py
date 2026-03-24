#!/usr/bin/env python3

import numpy as np
import torch
from pathlib import Path

# Check the actual attribution file shape for adult dataset
ROOT = Path('src')
attr_file = ROOT / 'explanations' / 'results_medoid' / 'adult_attributions.npy'
if attr_file.exists():
    print(f'Found: {attr_file}')
    attr = np.load(attr_file, allow_pickle=True)
    print(f'Attribution file shape: {attr.shape}')
    print(f'Attribution file type: {type(attr)}')
    if hasattr(attr, 'item'):
        print(f'Can call .item(): {hasattr(attr, "item")}')
        try:
            item = attr.item()
            print(f'Item type: {type(item)}')
            if isinstance(item, dict):
                print(f'Dict keys: {list(item.keys())}')
        except:
            print('Cannot call .item()')
else:
    print(f'Attribution file not found: {attr_file}')

# Check the model-specific one
attr_file2 = ROOT / 'explanations' / 'results_medoid' / 'adult_model1' / 'test' / 'attributions.npy'
if attr_file2.exists():
    print(f'Found model-specific: {attr_file2}')
    attr2 = np.load(attr_file2, allow_pickle=True)
    print(f'Model-specific attribution shape: {attr2.shape}')
    print(f'Model-specific attribution type: {type(attr2)}')
    if hasattr(attr2, 'item'):
        try:
            item2 = attr2.item()
            print(f'Model-specific item type: {type(item2)}')
            if isinstance(item2, dict):
                print(f'Model-specific dict keys: {list(item2.keys())}')
                for k, v in item2.items():
                    print(f'  {k}: {type(v)} shape={getattr(v, "shape", "N/A")}')
        except Exception as e:
            print(f'Cannot call .item() on model-specific: {e}')

# Also check the test data shape for comparison
X_test = torch.load(ROOT / 'data' / 'adult' / 'Xtest.pt', map_location='cpu')
y_test = torch.load(ROOT / 'data' / 'adult' / 'ytest.pt', map_location='cpu')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')