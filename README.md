
![Header Image](/github/readme_header.png)


# Status
- **Arborist** is currently in alpha development. We are actively working on adding core application functionality. The beta testing phase has not yet commenced, and the `Issues` tab for this repository will remain disabled until further progress is made.

# Dev Roadmap

- [x] Add React
- [x] Enforce TypeScript in strict mode
- [x] Add IPC communication between React, Electron, and Python
- [x] Remove menu bar
- [x] Add custom header
- [x] Add Tailwind
- [x] Identify/resolve issue with multiple API server spin-ups (this is expected behavior)
- [x] Check that the `api` executable is being shut down when quitting the dev app
- [x] Add drag-and-drop SPSS file upload recognition
- [x] Separate webpack.config into `main` and `renderer` configs
- [x] Add hot reloading
- [x] Add app icon
- [ ] Add CSV file parsing and sanity checking
- [ ] Add SPSS file parsing and sanity checking
- [ ] Add workspace directory with file listing
- [ ] Add causal inference model training to API
- [ ] Add causal inference model prediction and export API
- [ ] Add buttons for maximize, minimize, and close
- [ ] Ensure that spinning up th main exe is non-blocking (add to background) when loading the app
- [ ] Upload the large `main` API exe file to GitHub releases
- [ ] Add a GitHub action to automatically create a new release (and upload the main binary)
- [ ] Add GitHub page for download the app for each OS
- [ ] Add visualizations
- [ ] Add visualizations export API
- [ ] Add NLP with input recognition/parsing and scoring
- [ ] Consider using an uncommon port for the API to avoid conflicts with other apps on the user's machine
- [ ] Create OS-specific versions of the `api` executable named `api-macOS`, `api-windows`, and `api-linux` and include them in the `api` folder in the `src` directory in the app repo

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
  - [ ] Copy the `requirements.txt` file from the `src/api` folder in this repo to the directory where the venv was created
  - [ ] `python3 -m venv venv`: Create a virtual environment on your local machine
  - [ ] `source venv/bin/activate`: Activate the virtual environment
  - [ ] `pip install -r requirements.txt`: Install the dependencies listed in the `requirements.txt` file
  - [ ] `pyinstaller --onefile --add-data "<PATH_TO_PYTHON_LIBRARIES_IN_VENV>:./lib" <PATH_TO_SRC_FOLDER_IN_APP_REPO>/main.py`: Package the Python code
    - The `<PATH_TO_PYTHON_LIBRARIES_IN_VENV>` will be something like `venv/lib/pythonX.X/site-packages`
    - The `<PATH_TO_SRC_FOLDER_IN_APP_REPO>` is simply the path to the `src` folder in this repo
    - The packaged python code will be called `main` and will be located in the `dist` folder
    - Move this `main` file to the `api` folder in the `src` directory in the app repo
  - [ ] `yarn package`: Build the app

---

# Dev Notes:
- If you experience `[Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use` it can be helpful to run `lsof -i :8000` (macOS) to see what process is using that port. You can then kill that process with `kill -9 <PID>` (macOS) to stop it.
- Electron can only execute JavaScript files, not TypeScript. Therefore, we must compile the TypeScript files to JavaScript before running the app. This is done automatically when running `yarn start` or `yarn build`, and the resulting JavaScript file is stored as `main.js` in the `dist` folder.
