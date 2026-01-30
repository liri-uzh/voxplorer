
# Voxplorer
Voxplorer is a web-app dashboard tailored to the analysis of voices.  
Voxplorer allows users to upload their pre–computed data in the form of 
a `CSV`, `TSV`, or `XLSX` table, or alternatively upload audio files (`WAV`, `MP3`, or `FLAC`) 
and extract either MFCCs or speaker embeddings (thanks to pre–trained 
[speechbrain](https://huggingface.co/speechbrain) models). 
  
The daashboard supports several dimensionality reduction algorithms to allow 
users to visualise in 2 or 3 dimensions their data in an interactive plot.  
This is paired with an interactive table of the full feature set, which allows 
users to filter data by values or logic statements, isolating particular data points 
in the visualisation. 
  
Finally, voxplorer supports the download of the full dataset or only the selected 
observations (both full feature set and reduced space when available). 
In parallel a `JSON` log of each processing setting (dimensionality reduction and 
feature extraction when used) will be downloaded, allowing the users to reproduce 
their visualisations and reduced dimension space. 
The figure also supports download as a `PNG` image.  
  
![Example plot from voxplorer](./images/newplot.png)

# Table of contents
| Contents |
| ------------- |
|  [Installation](#installation) |
| [Usage](#usage) |
| [Secure voxploer](#working-on-voxplorer-securely-(without-internet-connection)) |
| [Cite](#cite-this-work) |


# Installation
## Cloning the repository
First, move into the desired directory in which you would like 
to store voxplorer.  
Clone this repository:
```sh
git clone https://github.com/liri-uzh/voxplorer.git
```

If you would like to stay up to date with the most recent updates before an 
official release, clone the "develop" branch.
```sh
git clone --branch develop https://github.com/liri-uzh/voxplorer.git
```

## Install the dependencies using uv (recommended)
If you are using [`uv`](https://docs.astral.sh/uv/) as a package manager you don't need to do manually install the 
dependencies in a virtual environment; you can run the app by simply running the 
following `uv` command from the voxplorer directory.
```sh
cd voxplorer
uv run app.py
```

This will also directly start the app.  
You can access it by copying the provided link in a browser, clicking on the 
link directly from the terminal emulator (if supported by terminal emulator), 
or going to `http://127.0.0.1:8050` from a browser.

## Installing dependencies using pip
If you do not have `uv` installed or prefer to create your own Python virtual environment, 
you can install the requirements via `pip`:
```sh
cd voxplorer
pip install -r requirements.txt
```

# Usage
## Workflow
The general workflow is described in the following diagram; 
![Workflow diagram](./images/block_diagram.png)

## Running voxplorer
To run voxplorer, you can move the local repository and run
```sh
uv run app.py
```
or without `uv`
```sh
python3 app.py
```

Output should look something like:
```sh
INFO:dash.dash:Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'app'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:8050
INFO:werkzeug:Press CTRL+C to quit
```
In the output you will see where the address at which the app is running locally; 
copy it in your chosen browser address. 
*Normally the app will run at 127.0.0.1:8050*

### Creating an alias for voxplorer 
If you would like to be able to always run voxplorer from anywhere 
in your terminal, simply add the following function to your shell rc file (usually 
`~/.zshrc` on MacOS and Linux systems using `zsh` as their shell or `~/.bashrc`).  
Remember to update the path to voxplorer to where you cloned the repository (update the line after `# run voxplorer`)
```sh
voxplorer () {
  cur_dir = $pwd

  # run voxplorer
  cd /path/to/voxplorer/
  uv run app.py &

  # Get PID
  local pid=$!

  # open browser
  sleep 5
  open http://127.0.0.1:8050/

  # Wait
  wait "$pid"
  
  # Return home
  cd "$cur_dir"
}
```
Now from anywhere in your terminal you can run `voxplorer` and the app will start.
If you are on linux, please change the line:
```sh
  open http://127.0.0.1:8050/
```
with:
```sh
  xdg-open http://127.0.0.1:8050/
```

## Output
voxplorer has 3 types of outputs:
1. tables (`CSV`)
2. figures (`PNG`)
3. logs (`JSON`)  
Tables are always the full features table (either uploaded or computed within the 
dashboard) and the reduced dimensions table (if dimensionlity reduction was ran). 
These can be either all observations or only selected observations (two different 
download buttons).  
Logs contain all the settings used to process the data for either or both 
dimensionality reduction and feature extraction. Logs are downloaded automatically 
when downloading tables.  
Figures can be downloaded using the _camera_ button in the interactive figure and 
are basically a screenshot of the figure as seen currently. 

# Working on voxplorer securely (without internet connection)
To ensure the highest degree of security when using voxplorer, we have implemented it so that it can be used without internet connection.  
If you are intending to use the speaker embedding feature extraction, you should download the Speechbrain model from the Speechbrain project on Hugging Face ([https://huggingface.co/speechbrain](https://huggingface.co/speechbrain)) in advance.  
There are two ways of downloading models from Hugging Face:  
1. Cloning the repository directly:
(_from the Hugging Face docs_)
```sh
curl -LsSf https://hf.co/cli/install.sh | bash

hf download <model-id>
```
or using git
```sh
brew tap huggingface/tap  # on MacOs example
brew install git-xet
git xet install

git clone <model-url>
```

2. Manually downloading the needed files from the repository: 
Files required:
- `classifier.ckpt`
- `embedding_model.ckpt`
- `hyperparams.yaml`
- `label_encoder.ckpt`
- `mean_var_norm_emb.ckpt`

Of course, any Speechbrain model trained by you will work as well.  
**Remember to use the directory where the model files are saved as the `model_id` parameter in the feature extraction via Speechbrain embedding models in voxploer.**


# API
[./lib](./lib/) contains the 4 main backend components of `voxplorer`:
1. the [data loader](./lib/data_loader.py)
2. the [feature extractor](./lib/feature_extraction.py)
3. the [dimensionality reduction](./lib/dimensionality_reduction.py)
4. the [plotter](./lib/plotting.py)  
Each function and class in these files is well documented within their docstring, 
but more in-detail documentation should be coming soon.

# Supported OSs
- MacOS ✅
- Linux ⭕️ (not tested yet, but should work)
- Windows ✅ (installation checked using `uv`)

# Cite this work
Cite our Interspeech 2025 Show&Tell paper:
```bib
@inproceedings{deluca25_interspeech,
  title     = {{Voxplorer: Voice data exploration and projection in an interactive dashboard}},
  author    = {Alessandro {De Luca} and Srikanth Madikeri and Volker Dellwo},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {296--297},
  issn      = {2958-1796},
}
```
or plain text:
> De Luca, A., Madikeri, S., Dellwo, V. (2025) Voxplorer: Voice data exploration and projection in an interactive dashboard. Proc. Interspeech 2025, 296-297
