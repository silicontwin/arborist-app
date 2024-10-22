# Installation
- Create venv: `python3 -m venv venv`
- Activate venv: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

# Usage
- Activate venv: `source venv/bin/activate`
- Change to the directory of the project: `cd arborist`
- Run the program using briefcase: `briefcase dev`

# Development
- Activate venv: `source venv/bin/activate`
- CHange to the directory of the project: `arborist/src/arborist`
- Build the layouts:
  -  `pyside6-uic layout.ui -o layout.py`:
  -  `pyside6-uic browse.ui -o browse.py`:
  -  `pyside6-uic analyze.ui -o analyze.py`: