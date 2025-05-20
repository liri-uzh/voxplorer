# Installer for voxplorer on Intel x86 Macs

# Conda environment install
conda env create -f x86OSX_requirements/voxplorer_env_x86OSX.yml
# eval "$(conda shell.bash hook)"
conda activate voxplorer

# Install pip dependendencies
# SYSTEM_VERSION_COMPART=0 pip install tensorflow tensorflow-metal
# pip install -r x86OSX_pip_requirements.txt
SYSTEM_VERSION_COMPAT=0 pip install -r x86OSX_requirements/x86OSX_pip_requirements.txt
pip install soundfile --force-reinstall # ensures that libsndfile is installed correctly