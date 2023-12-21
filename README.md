
![Header Image](/github/readme_header.png)


# Status
- **Arborist** is currently in alpha development. We're actively working on adding core application functionality. The beta testing phase has not yet commenced, and the `Issues` tab for this repository will remain disabled until the app reaches the appropriate level of usability/polish.

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

# Relevant Repos
- [arborist-api](https://github.com/silicontwin/arborist-api)
- [bcf](https://github.com/jaredsmurray/bcf)
- [StochasticTree](https://github.com/andrewherren/StochasticTree)

---

# Building Arborist for Distribution
- [ ] Update the version number in `package.json` (Arborist dev team only)
- [ ] Package the Python API code as an executable via `pyinstaller`:
  - [ ] Clone the `arborist-api` repositiory to your local machine and navigate to the `arborist-api` folder
  - [ ] Follow the `Instructions` in the included `README.md` file to create a virtual environment and install the dependencies
  - [ ] Ensure that the virtual environment is activated
  - [ ] Bundle the API as an executable: `pyinstaller --onefile --add-data "<PATH_TO_PYTHON_LIBRARIES_IN_VENV>:./lib" <PATH_TO_SRC_FOLDER_IN_APP_REPO>/main.py`
    - The variable `<PATH_TO_PYTHON_LIBRARIES_IN_VENV>` will be something like `env/lib/pythonX.Y/site-packages`
    - The variable `<PATH_TO_SRC_FOLDER_IN_APP_REPO>` is the path to the `src` folder in this repo
    - The packaged python code will be called `main` and will be located in the `dist` folder of the `arborist-api` repo
  - Move the newly created `main` executable file in `arborist-api/dist` to the `api` folder in the `src` directory in the app repo `arborist-app/src`
  - [ ] `yarn package`: Build the app

---

# Dev Notes:
- If you experience `[Errno 48] error while attempting to bind on address ('0.0.0.0', 8000): address already in use` it can be helpful to run `lsof -i :8000` (macOS) to see what process is using that port. You can then kill that process with `kill -9 <PID>` (macOS) to stop it.
- Electron can only execute JavaScript files, not TypeScript. Therefore, we must compile the TypeScript files to JavaScript before running the app. This is done automatically when running `yarn start` or `yarn build`, and the resulting JavaScript file is stored as `main.js` in the `dist` folder.
