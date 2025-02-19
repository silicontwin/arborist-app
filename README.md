# Installation
- Create venv: `python3 -m venv venv`
- Activate venv: `source venv/bin/activate`
- Select the correct Python intepreter (so that you install packages in the venv)
- Install requirements: `pip install -r requirements.txt`
- Check the installation: `pip list`

# Usage
- Activate venv: `source venv/bin/activate`
- Change to the directory of the project: `cd arborist`
- Run the program using briefcase: `briefcase dev`

# Development
- Activate venv: `source venv/bin/activate`
- Change to the directory of the project: `arborist/src/arborist`
- Build the layouts:
  -  `pyside6-uic load.ui -o load.py`:
  -  `pyside6-uic train.ui -o train.py`:
  -  `pyside6-uic predict.ui -o predict.py`: