# Installation Tools and Requirements

This directory contains installation files and tools required for setting up the MT-AI trading environment.

## Contents

- `exness5setup.exe` - Exness MT5 Terminal installer
- `TA_Lib-0.4.18-cp38-cp38-win_amd64.whl` - Technical Analysis Library wheel file for Python 3.8
- `python_lib_installation.ipynb` - Jupyter notebook with installation instructions

## Installation Steps

1. Install Python Dependencies:
   ```bash
   pip install -r ../../requirements.txt
   ```

2. Install TA-Lib:
   ```bash
   pip install TA_Lib-0.4.18-cp38-cp38-win_amd64.whl
   ```

3. Install Exness MT5 Terminal:
   - Run `exness5setup.exe`
   - Follow the installation wizard instructions
   - Configure your trading account credentials in `.env` file

## Additional Setup

For detailed setup instructions and troubleshooting, refer to:
1. `python_lib_installation.ipynb` for Python package installation details
2. Project root's README.md for complete project setup
3. `.env.example` for required environment variables 