import torch
ckpt = torch.load('checkpoints/symbolic/full_best.pt', map_location='cpu', weights_only=False)
ut_sd = ckpt['ut']
for k in ['embed.coordinates', 'embed.scale', 'decoder.linear.weight', 'decoder.linear.bias', 'decoder.group_classifier.weight', 'decoder.group_classifier.bias']:
    if k in ut_sd:
        print(f'{k}: shape={ut_sd[k].shape}')
print(f'step: {ckpt.get("step", "?")}')
print('keys:', list(ckpt.keys())[:5])
V = ut_sd['embed.coordinates'].shape[0]
D = ut_sd['embed.coordinates'].shape[1]
print(f'Vocab size: {V}, Coord dim: {D}')
