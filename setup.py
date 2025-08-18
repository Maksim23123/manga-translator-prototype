import os
import shutil
import subprocess
import sys

# Step 1: Create virtual environment
venv_dir = 'venv'
if not os.path.exists(venv_dir):
    print('Creating virtual environment...')
    subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
else:
    print('Virtual environment already exists.')

# Step 2: Install requirements
pip_path = os.path.join(venv_dir, 'Scripts', 'pip.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'pip')
if os.path.exists('requirements.txt'):
    print('Installing dependencies from requirements.txt...')
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
else:
    print('requirements.txt not found.')

# Step 3: Copy config.template.json to config.json if not present
if os.path.exists('config.template.json'):
    if not os.path.exists('config.json'):
        print('Copying config.template.json to config.json...')
        shutil.copy('config.template.json', 'config.json')
    else:
        print('config.json already exists.')
else:
    print('config.template.json not found.')

# Step 4: Ensure inputs and outputs folders exist
for folder in ['inputs', 'outputs']:
    if not os.path.exists(folder):
        print(f'Creating folder: {folder}')
        os.makedirs(folder)
    else:
        print(f'Folder already exists: {folder}')

print('Setup complete.')
