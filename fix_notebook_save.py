import json
import os

notebook_path = r'd:\multimodel-surveillence\multimodal_fusion.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if this cell is meant to be the SAVE CONFIG cell based on typical labels or past outputs
        is_save_cell = False
        if "outputs" in cell:
            for out in cell['outputs']:
                if "text" in out:
                    text_all = "".join(out['text'])
                    if "Fusion config saved!" in text_all:
                        is_save_cell = True
                        break
        
        if is_save_cell:
            new_source = []
            new_source.append("import json\n")
            new_source.append("config = {\n")
            new_source.append("    'audio_weight': 0.4,\n")
            new_source.append("    'video_weight': 0.6,\n")
            new_source.append("    'threshold': 0.2,\n")
            new_source.append("    'img_size': 96,\n")
            new_source.append("    'audio_model': 'audio_isolation_forest.pkl',\n")
            new_source.append("    'audio_scaler': 'audio_scaler.pkl',\n")
            new_source.append("    'video_model': 'video_transfer_model.h5'\n")
            new_source.append("}\n")
            new_source.append("\n")
            new_source.append("with open(os.path.join(BASE, 'fusion_config.json'), 'w') as f:\n")
            new_source.append("    json.dump(config, f, indent=2)\n")
            new_source.append("\n")
            new_source.append("print('Fusion config saved with threshold 0.2!')\n")
            new_source.append("print(json.dumps(config, indent=2))")
            cell['source'] = new_source
            # Clear old output to avoid confusion
            cell['outputs'] = []

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully restored and updated Save Config cell in multimodal_fusion.ipynb')
