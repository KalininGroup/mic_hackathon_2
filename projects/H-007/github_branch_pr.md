# Creating a Branch and Opening a Pull Request on GitHub

Use this guide to collaborate safely by creating topic branches and submitting pull requests.

1. **Sync your local `main` (or `master`) branch**  
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create and switch to a new branch**  
   Choose a short, descriptive name (e.g., `feature/login-form` or `bugfix/fix-typo`).
   ```bash
   git checkout -b feature/your-branch-name
   ```

3. **Make and stage your changes**  
   Edit files, then stage the changes when ready:
   ```bash
   git status               # see modified files
   git add <file1> <file2>  # or use git add . to stage everything
   ```

4. **Commit with a clear message**  
   ```bash
   git commit -m "Short, descriptive commit message"
   ```

5. **Push the branch to GitHub**  
   ```bash
   git push -u origin feature/your-branch-name
   ```
   The `-u` flag sets upstream tracking so future `git push`/`git pull` use this branch by default.

6. **Open a pull request (PR)**  
   - Go to the repository on GitHub. You should see a prompt to create a PR from your newly pushed branch. Click **Compare & pull request**.
   - Fill in the PR title and description, summarizing the changes and any testing performed.
   - Choose reviewers if needed, then click **Create pull request**.

7. **Address feedback and update the PR**  
   Make additional commits on the same branch and push again; GitHub updates the PR automatically.

8. **Merge the PR when approved**  
   After approvals and passing checks, click **Merge pull request** (or the chosen strategy). Delete the branch if desired.

## Tips
- Keep branches focused on a single change to simplify reviews.
- Run tests locally before pushing to reduce CI failures.
- If your repository protects `main`, you must open a PR—direct pushes will be blocked.
