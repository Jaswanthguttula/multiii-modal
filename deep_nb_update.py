import json
import os

notebook_path = r'd:\\multimodel-surveillence\\multimodal_fusion.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # 1. Update model strings
            line = line.replace("'video_autoencoder.h5'", "'video_transfer_model.h5'")
            line = line.replace('"video_autoencoder.h5"', '"video_transfer_model.h5"')
            
            # 2. Update IMG_SIZE
            line = line.replace("IMG_SIZE = 128", "IMG_SIZE = 96")
            
            # 3. Update Thresholds (specific patterns)
            line = line.replace("threshold=0.5", "threshold=0.2")
            line = line.replace("'threshold': 0.5", "'threshold': 0.2")
            line = line.replace('"threshold": 0.5', '"threshold": 0.2')
            line = line.replace("score > 0.5", "score > 0.2")
            line = line.replace("axvline(x=0.5", "axvline(x=0.2")
            
            new_source.append(line)
        cell['source'] = new_source
        
        # CLEAR OUTPUTS to force the user to re-run and see the NEW values
        cell['outputs'] = []

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully applied deep updates to multimodal_fusion.ipynb and cleared outputs.')
