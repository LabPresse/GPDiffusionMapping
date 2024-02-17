# GPDiffusionMapping
Code for our diffusion coefficient mapping project.

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

## Output
Once done sampling the MAP Estimate Surface is automatically plotted and saved for the user in the output directory. In the subdirectory named samples, the package also saves all MCMC samples in a file called Samples.csv along with a file containing the probability associated with each MCMC sample called LogPosterior.csv.