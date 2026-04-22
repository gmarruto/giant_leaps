# GIANT LEAPS – Alternative Protein Screening Engine

## Project Overview
This development implements a machine learning–based similarity engine designed to identify alternative protein items that are most similar to a user-selected traditional protein item.
The core algorithm combines Principal Component Analysis (PCA) for dimensionality reduction with a k-Nearest Neighbours (k-NN) approach to compute similarity in a reduced feature space.
The system interacts with a graphical user interface, and for this purpose a REST API has been developed to enable communication between the frontend and the underlying machine learning algorithm.


## Technical Requirements

Python (minimum version 3.12)
Virtual environment support (venv)
Local execution environment (development mode)
All third-party dependencies are explicitly defined in the requirements.txt file.


## Environment Setup
### 1. Create a Python Virtual Environment
From the root directory of the repository, create a virtual environment:

    python -m venv ai_giantleaps_venv


Activate the environment:

*   Linux / macOS

        source ai_giantleaps_venv/bin/activate



*   Windows (PowerShell)

        ai_giantleaps_venv\Scripts\Activate.ps1


### 2. Install Dependencies
Once the environment is activated, install the required dependencies:

    pip install --upgrade pip
    pip install -r requirements.txt


This step ensures that all libraries required for the algorithm and API execution are available in the local environment.


## Running the Development Environment

This document describes the technical steps required to manually execute the GIANT LEAPS API from a command-line interface. The procedure is intended for Linux-based environments and assumes that the project and its Python virtual environment are already installed on the system.


#### Step 1 – Open a Command-Line Interface
Start a terminal session on the target machine where the GIANT LEAPS project is deployed.
All subsequent commands are executed from this terminal session.


#### Step 2 – Activate the Python Virtual Environment
Activate the Python virtual environment that contains all required dependencies for the project:

    source ai_giantleaps_env/bin/activate


#### Step 3 – Navigate to the Project Root Directory
Change the current working directory to the root of the GIANT LEAPS project

    cd GIANT_LEAPS/


This directory is expected to contain:

*   The API entry point (api_main.py)

*   The machine learning implementation

*   Auxiliary configuration and runtime files


#### Step 4 – Launch the API Service

Start the API service using Uvicorn, specifying the application entry point, network configuration, and SSL parameters:

*   In case SSL certificate is available:

        uvicorn api_main:app \
        --host 0.0.0.0 \
        --port 8008 \
        --ssl-keyfile /etc/ssl/certs/cert.key \
        --ssl-certfile /etc/ssl/certs/cert.pem

*   If there is no SSL certificate:

        uvicorn api_main:app \
        --host 0.0.0.0 \
        --port 8008 \


This command performs the following operations:

Application initialization
Loads the ASGI application object (app) defined in api_main.py.

Network exposure--host 0.0.0.0 enables the API to accept incoming connections from external clients.
--port 8008 sets the TCP port on which the service listens.
Secure communication (HTTPS)
SSL encryption is enabled using the provided key and certificate files, ensuring secure communication between the API and external clients such as a graphical user interface.


## Usage Notes

The virtual environment and execution scripts are intended exclusively for local development.
No credentials, sensitive configuration files, or local data artifacts should be committed to the repository.
Generated data, intermediate artifacts, trained models, and execution outputs are expected to remain outside version control.


## Contact and Maintenance
This repository is maintained as part of the GIANT LEAPS research activities.

