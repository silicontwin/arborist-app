
# Dev Roadmap

- [x] Add React
- [x] Enforce TypeScript in strict mode
- [ ] Remove menu bar
- [ ] Add custom header
- [ ] Add IPC communication between React, Electron, and Python
- [ ] Add drag-and-drop SPSS file upload recognition
- [ ] Add SPSS file parsing and sanity checking
- [ ] Add causal inference model training to API
- [ ] Add causal inference model prediction and export API
- [ ] Add visualizations
- [ ] Add visualizations export API
- [ ] Add NLP with input recognition/parsing and scoring
- [ ] Consider using an uncommon port for the API to avoid conflicts with other apps on the user's machine

---

# Overview
This is a cross platform desktop app that enables users to efficiently leverage stochastic tree ensembles (BART/XBART) for supervised learning and causal inference on a local dataset. The resulting predictions can then be exported to their local machine along with a variety of visualizations of the resulting data.

---

# System Architecture

## Electron
We are using Electron to create a cross platform desktop app.

## Baysian Causal Inference
We are currently using the BART/XBART Python models from the StochasticTree GitHub repository for our analyses. The model is trained on the user's local machine and the resulting predictions are returned to the Electron app. This enables us to perform any analyses locally without having to send the user's data to a remote server.

## FastAPI
We use FastAPI to create a REST API that can be called from within the Electron app. The API is written in Python and acts as an interface to our causal inference model. The API is packaged using PyInstaller and is included in the Electron app so that a user doesn't need to install Python on their machine to use the app. The FastAPI server is started when the Electron app is started and is stopped when the app is closed.

---

# Machine Requirements

---

# Development
- `yarn start`: Start the app in development mode

---

# Building the App for Distribution
- [ ] Update the version number in `package.json`
- [ ] Package the Python API code:
  - [ ] Navigate to a directory where you want to create a virtual environment
  - [ ] Copy the `requirements.txt` file from the `src/resources` folder in this repo to the directory where the venv was created
  - [ ] `python3 -m venv venv`: Create a virtual environment on your local machine
  - [ ] `source venv/bin/activate`: Activate the virtual environment
  - [ ] `pip install -r requirements.txt`: Install the dependencies listed in the `requirements.txt` file
  - [ ] `pyinstaller --onefile --add-data "<PATH_TO_PYTHON_LIBRARIES_IN_VENV>:./lib" <PATH_TO_SRC_FOLDER_IN_APP_REPO>/api.py`: Package the Python code
    - The `<PATH_TO_PYTHON_LIBRARIES_IN_VENV>` will be something like `venv/lib/pythonX.X/site-packages`
    - The `<PATH_TO_SRC_FOLDER_IN_APP_REPO>` is simply the path to the `src` folder in this repo
    - The packaged python code will be called `api` and will be located in the `dist` folder
    - Move this `api` file to the `resources` folder in the `src` directory in the app repo
  - [ ] `yarn build`: Build the app

---

# Dev Notes:
- If you experience `[Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use` it can be helpful to run `lsof -i :8000` (macOS) to see what process is using that port. You can then kill that process with `kill -9 <PID>` (macOS) to stop it.
