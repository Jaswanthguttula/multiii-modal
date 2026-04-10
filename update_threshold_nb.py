import json
import os

notebook_path = r'd:\multimodel-surveillence\multimodal_fusion.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        
        # 1. Update MultimodalFusion class default threshold
        if "MultimodalFusion" in source_str and "threshold=0.5" in source_str:
            cell['source'] = [s.replace('threshold=0.5', 'threshold=0.2') for s in cell['source']]
            
        # 2. Update threshold markers in plots (axvline and threshold variables)
        if "axvline(x=0.5" in source_str:
            cell['source'] = [s.replace('axvline(x=0.5', 'axvline(x=0.2') for s in cell['source']]
        
        if "pred = 'ANOMALY' if score > 0.5" in source_str:
            cell['source'] = [s.replace('score > 0.5', 'score > 0.2') for s in cell['source']]
            
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated thresholds in multimodal_fusion.ipynb to 0.2')
