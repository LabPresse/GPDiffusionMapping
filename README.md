# GPDiffusionMapping
This is the repository for our Diffusion Mapping project from trajectory data, BioArXiV link. For ease of use It is organized so that any user only needs to interact with the main.py file. There is also a sample dataset provided in the /data dirctory that correlates to a flat Diffusion Map at 0.05 $\frac{\mu m^2}{s}$, where the data set has ~5000 localizations.

## Installation

To install the package any user only needs python, though git will make installation easier it is not necassary. A user who has git can simply run the following command in their terminal in the directory they would like to download and run all files:

```bash
git clone https://github.com/LabPresse/GPDiffusionMapping.git
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Usage
Any user of this package should only need to run the main.py file, and make any possible edits there. All hyper parameters are directly tunable in the main.py file. The data file is assumed to be a .csv file that has 3 columns, organized as (Trajectory Label, Xposition, Yposition) and the first row is a header.

## Assumed Units:

* Time: Seconds
* Length: Nanometers