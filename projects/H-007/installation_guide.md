# Python, Anaconda, and VS Code Setup Guide

Follow this guide to install Python, Anaconda, and Visual Studio Code (VS Code) and to configure and test a virtual environment inside VS Code. Steps are organized by operating system where relevant.

## 1. Install Python

### Windows
1. Download the latest stable installer from the [official Python website](https://www.python.org/downloads/windows/).
2. Run the installer and check **Add Python to PATH** before clicking **Install Now**.
3. After installation, open **Command Prompt** and verify:
   ```bash
   python --version
   ```

### macOS
1. Install Homebrew if you do not already have it:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python via Homebrew:
   ```bash
   brew install python
   ```
3. Confirm the version:
   ```bash
   python3 --version
   ```

### Linux (Debian/Ubuntu)
1. Update package lists and install Python 3 and pip:
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-pip
   ```
2. Verify installation:
   ```bash
   python3 --version
   pip3 --version
   ```

## 2. Install Anaconda (Optional but Recommended)
Anaconda bundles Python, common data science libraries, and conda environment management.

### Download and Install
- **Windows/macOS**: Download the graphical installer from the [Anaconda Individual Edition](https://www.anaconda.com/products/distribution) page and follow the on-screen prompts. Allow the installer to add Anaconda to your PATH when asked.
- **Linux**: Download the shell installer and run:
  ```bash
  bash Anaconda3-*.sh
  ```

### Verify Installation
After installation, open a terminal (or Anaconda Prompt on Windows) and run:
```bash
conda --version
```

## 3. Install Visual Studio Code

### Windows/macOS
1. Download the installer from the [VS Code website](https://code.visualstudio.com/Download).
2. Run the installer using default options. On Windows, enable "Add to PATH" and "Register as default editor" if desired.

### Linux
1. Follow the distribution-specific instructions from the VS Code download page. For Debian/Ubuntu:
   ```bash
   sudo apt update
   sudo apt install -y wget gpg
   wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
   sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
   sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
   sudo apt update
   sudo apt install -y code
   ```

### Recommended VS Code Extensions
- **Python** (by Microsoft)
- **Pylance** (for improved language support)
- **Jupyter** (for notebook workflows)

Install extensions from the **Extensions** view (Ctrl+Shift+X or Cmd+Shift+X) by searching for their names.

## 4. Create and Activate a Virtual Environment in VS Code

### Using `venv` (built-in)
1. Open VS Code and open your project folder.
2. Open the integrated terminal (**View > Terminal** or ``Ctrl+` ``).
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the environment:
   - **Windows (Command Prompt)**:
     ```bash
     .venv\\Scripts\\activate
     ```
   - **PowerShell**:
     ```powershell
     .venv\\Scripts\\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
5. When prompted by VS Code, select **Yes** to trust the workspace and set the interpreter to `.venv`.

### Using `conda`
1. Open the integrated terminal in VS Code.
2. Create an environment (replace `myenv` with your preferred name and pick a Python version):
   ```bash
   conda create -n myenv python=3.12
   ```
3. Activate the environment:
   ```bash
   conda activate myenv
   ```
4. In VS Code, use **Command Palette** (Ctrl+Shift+P or Cmd+Shift+P) > **Python: Select Interpreter** and choose the interpreter from the activated environment.

## 5. Test the Virtual Environment in VS Code

### Install a Test Dependency
With the environment activated in the VS Code terminal, install a package (e.g., `requests`):
```bash
pip install requests
```

### Run a Simple Script
Create `test_env.py` in your project folder with the following content:
```python
import sys
import requests

print(f"Python executable: {sys.executable}")
print(f"requests version: {requests.__version__}")
```

Run the script from the integrated terminal:
```bash
python test_env.py
```
You should see the virtual environment's Python path and the installed `requests` version, confirming the environment is active.

## 6. Troubleshooting Tips
- If VS Code does not detect the environment, reopen the window (**Developer: Reload Window** from the Command Palette) and reselect the interpreter.
- Ensure the integrated terminal is using the correct shell (e.g., Command Prompt, PowerShell, bash) for activation commands.
- For permission errors on macOS/Linux, prepend commands with `sudo` only when necessary.
- On Windows, if scripts are blocked in PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in an elevated PowerShell and retry activation.

You are now ready to develop Python projects in an isolated virtual environment using VS Code!
