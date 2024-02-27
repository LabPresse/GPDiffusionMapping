# GPDiffusionMapping
This is the repository for our Diffusion Mapping project from trajectory data, BioArXiV link. For ease of use It is organized so that any user only needs to interact with the main.py file. There is also a sample dataset provided in the /data dirctory that correlates to a flat Diffusion Map at 0.05 $\frac{\mu m^2}{s}$, where the data set has ~5000 localizations.

## Installation

#### Retrieve files locally:

To install the package any user only needs python, though git will make installation easier it is not necassary. A user who has git can simply run the following command in their terminal to retrieve all files locally:

```bash
git clone https://github.com/LabPresse/GPDiffusionMapping.git
```
For those who do not have git, simply download the package as a Zip file via the green <>Code button above. Then unzip the file on your computer and open the directory in your favorite python editor.

##### Setup environment

Note that the process to setup the virtual environment with required python packages differes based on OS. Follow the one below that applies to you.

###### Mac/Linux Users

In the terminal, with the /GPDiffusionMapping directory open, run the follwing:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

###### Windows Users

In the Command Prompt/Terminal, with the /GPDiffusionMapping directory open, run the follwing:
```bash
python -m venv .env
.env\Scripts\activate
pip install -r requirements.txt
```

Now you have all necassary packages loaded into a virual environment and are ready to use the code by simply running the main.py file. Additional description provided below.

## Usage
Any user of this package should only need to run the main.py file, and make any possible edits there. For analysis based on automated hyperparameters, one simply needs to change the dataPath to be the path of the data they would like analyzed. The data file needs to be a .csv file that has 3 columns, organized as (Trajectory Label, Xposition, Yposition) with the first row as a header. For those so inclined to have more control over the inference task, all hyperparameters are directly tunable in the main.py file. Simply find the name of the hyperparameter you would like to edit in the PARAMETERS object of the model.py file and pass it as a keyword arguement in the model.analyze() command of the main.py file. In this same function you can also pass the number of samples/iterations you would like to perform.

## Assumed Units:

* Time: Seconds
* Length: Nanometers