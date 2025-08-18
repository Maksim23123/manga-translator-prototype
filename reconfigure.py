import json
import os

def prompt_update(field, current_value):
    new_value = input(f"Current {field}: {current_value}\nEnter new value (leave blank to keep current): ")
    return new_value or current_value

config_path = 'config.json'
if not os.path.exists(config_path):
    print('config.json not found. Please run setup.py first.')
    exit(1)

with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Update API keys (nested under 'API_KEYS')
if 'API_KEYS' in config and isinstance(config['API_KEYS'], dict):
    print('\n--- API Keys ---')
    for key in config['API_KEYS']:
        config['API_KEYS'][key] = prompt_update(f"API key '{key}'", config['API_KEYS'][key])
else:
    print('No API_KEYS found in config. Skipping.')

# Update input/output folders and input file name (nested under 'IMPORT_PARAMS')
if 'IMPORT_PARAMS' in config and isinstance(config['IMPORT_PARAMS'], dict):
    print('\n--- Import Params ---')
    config['IMPORT_PARAMS']['input_folder'] = prompt_update('input folder', config['IMPORT_PARAMS'].get('input_folder', 'inputs'))
    config['IMPORT_PARAMS']['output_folder'] = prompt_update('output folder', config['IMPORT_PARAMS'].get('output_folder', 'outputs'))
    config['IMPORT_PARAMS']['input_file_name'] = prompt_update('input file name', config['IMPORT_PARAMS'].get('input_file_name', ''))
else:
    print('No IMPORT_PARAMS found in config. Skipping.')

# Optionally, remove duplicated top-level keys if present
for key in ['input_folder', 'output_folder', 'input_file']:
    if key in config:
        del config[key]

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('Config updated successfully.')
