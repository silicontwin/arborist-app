
![Header Image](/github/readme_header.png)


# Status
**Arborist** is currently an `alpha` stage prototype, and not yet ready for research (or real) data. We're actively working on adding core application functionality. The beta testing phase has not yet commenced, and the `Issues` tab for this repository will remain disabled until the app reaches the appropriate level of usability/polish.

- Note: The `main` standalone API executable that contains Python, the required dependencies, and the Arborist API is currently too large to upload to GitHub and will be uploaded to `Git Large File Storage` in the near future (and will be accessible via the `releases` tab). Therefore, the `main` executable is not yet included in this repository (but can be built on your local machine using the `Building Arborist for Distribution` instructions).

---

# Overview
**Arborist** is a cross-platform application designed for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including `bcf`, `BART`, and `XBART`. Arborist significantly lowers the barrier to entry for these types of analytics workflows by offering:

- A user-friendly drag-and-drop interface for data import.
- A streamlined process for model selection and analysis, accessible with just a few clicks.
- Sensible defaults for model parameters.
- The ability to export predictions and create polished plots suitable for publications and presentations.

The Arborist executable includes Python, the Arborist API/server, and various models, thereby eliminating the need for end users to:

- Install or configure Python and its dependencies.
- Use the command line.
- Possess extensive Python knowledge or experience.
- Run servers locally.
- Understand how to interface with APIs.
- Know techniques for chunking datasets efficiently for API processing.

All analyses are performed locally on the user's machine. No data is sent to remote servers, making Arborist an ideal tool for researchers handling sensitive data.

---


# Affiliated Repos
- [arborist-api](https://github.com/silicontwin/arborist-api)
- [bcf](https://github.com/jaredsmurray/bcf)
- [StochasticTree](https://github.com/andrewherren/StochasticTree)

---

# System Architecture

## Arborist API
Built using FastAPI, the Arborist API includes a Python server that acts as an interface to the bcf/BART/XBART models. The API is packaged using PyInstaller and is included in the Electron app so that a user doesn't need to install Python on their machine to use the app. The FastAPI server is started when the Electron app is started and is stopped when the app is closed.

## Electron
Arborist is built on top of the Electron framework. Electron is a cross-platform framework for building desktop applications using JavaScript, HTML, and CSS. Electron is used to create the user interface and to communicate with the Arborist API.

## Baysian Causal Inference
We are  using the bcf/BART/XBART Python models from the StochasticTree GitHub repository for our analyses. Models are trained on the user's local machine and the resulting predictions are returned to the Electron app. This enables us to perform any analyses locally without having to send the user's data to a remote server.

---

# Machine Requirements
Testing is currently being performed using MacBook Pro M1 Max and newer machines.

---

# Development
- Clone the repository to your local machine
- `yarn`: Install dependencies
- `yarn start`: Start the app in development mode

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
