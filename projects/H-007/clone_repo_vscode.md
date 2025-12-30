# Cloning a GitHub Repository in VS Code

Follow these steps to clone any public or private GitHub repository directly from Visual Studio Code.

1. **Install VS Code & Git**  
   Ensure Visual Studio Code and Git are installed. On Windows, install Git from [git-scm.com](https://git-scm.com/downloads); on macOS/Linux use your package manager.

2. **Sign in to GitHub (optional but recommended)**  
   In VS Code, open the **Accounts** menu (profile icon) and sign in with GitHub to enable authenticated cloning of private repos.

3. **Open the Command Palette**  
   Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) to open the Command Palette.

4. **Run "Git: Clone"**  
   Type `Git: Clone` and select it. When prompted, paste the repository URL (e.g., `https://github.com/owner/repo.git`).

5. **Choose a local folder**  
   Select or create the destination folder where you want the repository to live.

6. **Open the cloned repository**  
   When VS Code offers to open the newly cloned repository, click **Open**. The folder will be loaded in VS Code with Git integration ready.

## Tips
- If you have SSH keys configured with GitHub, use the `git@github.com:owner/repo.git` URL to clone over SSH.
- You can also start from the **Source Control** sidebar: click **Clone Repository** and follow the same prompts.
- Confirm Git is available by running `git --version` in the integrated terminal (`Ctrl+``).
