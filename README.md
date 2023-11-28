# Development





# Building the App for Distribution
- [ ] Update the version number in `package.json`
- [ ] Package the Python API code:
  - [ ] `python3 -m venv venv` # Create a virtual environment on your local machine
  - [ ] `source venv/bin/activate` # Activate the virtual environment
  - [ ] Copy the `requirements.txt` file from the `src/resources` folder in this repo to the directory where the venv was created
  - [ ] `pip install -r requirements.txt` # Install the dependencies listed in the `requirements.txt` file
  - [ ] `pyinstaller --onefile --add-data "<PATH_TO_PYTHON_LIBRARIES_IN_VENV>:./lib" <PATH_TO_SRC_FOLDER_IN_APP_REPO>/api.py` # Package the Python code
    - The `<PATH_TO_PYTHON_LIBRARIES_IN_VENV>` will be something like `venv/lib/pythonX.X/site-packages`
    - The `<PATH_TO_SRC_FOLDER_IN_APP_REPO>` is simply the path to the `src` folder in this repo
    - The packaged python code will be called `api` and will be located in the `dist` folder
    - Move this `api` file to the `resources` folder in the `src` directory in the app repo