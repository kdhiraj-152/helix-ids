from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from helix_ids.utils.callbacks import ModelCheckpoint

# Prepare a tiny model
model = nn.Linear(10, 2)
# include recommended optimizer hyperparameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Use artifacts directory in repo
out_dir = Path('artifacts/test_smoke')
out_dir.mkdir(parents=True, exist_ok=True)

ckpt_path = out_dir / 'helix_smoke_{epoch:03d}_{threat_weighted_f1:.4f}.pt'
cb = ModelCheckpoint(filepath=ckpt_path, monitor='threat_weighted_f1', save_best_only=False, save_weights_only=False, verbose=True)
cb.set_model(model)
cb.set_optimizer(optimizer)
cb.on_train_begin()

# Simulate end of epoch save
cb.on_epoch_end(0, {'threat_weighted_f1': 0.5123})

print('Files written:')
for p in out_dir.glob('*'):
    print(' -', p)
