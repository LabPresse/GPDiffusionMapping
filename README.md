# GPDiffusionMapping

This is the repository for our Diffusion Mapping project from trajectory data, BioArXiV link. For ease of use It is organized so that any user only needs to interact with the main.py file. There is also a sample dataset provided in the /data dirctory that correlates to a flat Diffusion Map at 0.05 $\frac{\mu m^2}{s}$, where the data set has ~5000 localizations.

## Installation

### Download

To download the package, simply clone this repository using `git` or click the green <>Code button above and select the download option. This will download the package as a Zip file to your computer. Unzip the file and open the directory in your favorite python editor.

#### Setup environment

In order to run the code you will need to set up a virtual environment and install the required packages. This can be done by running the following commands in the terminal or command prompt.

##### Mac/Linux Users

In the terminal, `cd` to the project directory and run the follwing:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

##### Windows Users

In the command prompt, `cd` to the project directory and run the follwing:
```bash
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

## Usage

To run the code, simply run the main.py file in your favorite python editor. Change `dataPath` to the path of your data file.

The data file should be a csv file with 3 columns: `particle #`, `xPos`, and `yPos`, where `particle #` is the particle number, `xPos` is the x position of the particle, and `yPos` is the y position of the particle. Position is measured in nanometers. Time steps are assumed to be taken at 30 Hz.

The code will output an image and save it to the `output/` directory.

The code is set to run with default hyperparameters, but the user can change the hyperparameters as needed by including them as keyword arguments in the `analyze` function. A dictionary of the hyperparameters can be found in `model.py`.

## Assumed Units

* Time: Seconds
* Length: Nanometers
