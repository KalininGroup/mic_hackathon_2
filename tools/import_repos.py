#!/usr/bin/env python3
"""
Import hackathon project repos into a uniform, lightweight archive format.

Reads: repos_with_ids.csv (TSV or CSV) with columns: project_id, repo

For GitHub links:
  - shallow clones repo
  - copies only allowed, non-large files into:
      projects/<H-XXX>/
        README.md (if present upstream; otherwise created)
        ARCHIVE_NOTE.md   (public-friendly note + examples of skipped files)
        _SOURCE.txt       (original link)
        notebooks/        (.ipynb)
        docs/             (.pdf)
        assets/           (.png/.jpg/.svg/.gif)
        code/             (other allowed code/config files)
  - skips large or disallowed artifacts (models, big data, venv, etc.)

For non-GitHub links (Colab/Drive/Box/Other):
  - creates projects/<H-XXX>/README.md + ARCHIVE_NOTE.md + _SOURCE.txt
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse


# -------------------------
# Settings
# -------------------------
REPO_ROOT = os.getcwd()
CSV_PATH = os.path.join(REPO_ROOT, "repos_with_ids.csv")
DEST_ROOT = os.path.join(REPO_ROOT, "projects")

MAX_BYTES = 40 * 1024 * 1024  # 25 MB per file

# Keep this conservative: website/judging repo should stay lightweight.
ALLOW_EXT = {
    ".ipynb",
    ".md", ".txt",
    ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".py", ".m", ".r", ".jl",
    ".js", ".ts",
    ".json", ".yaml", ".yml", ".toml",
    ".csv", ".tsv",
    ".ini", ".cfg",
    ".sh", ".bat", ".ps1",
    ".html", ".css",
    ".ipynb",  # explicit
    ".rst",
    ".tex",
}

# Always exclude these dirs anywhere in repo
DENY_DIR_NAMES = {
    ".git", ".ipynb_checkpoints", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox",
    "venv", ".venv", "env", ".env",
    "node_modules", ".next", "dist", "build",
}

# Always exclude these extensions regardless of size
DENY_EXT = {
    ".pt", ".pth", ".ckpt", ".onnx",
    ".tif", ".tiff", ".svs", ".nd2", ".czi",
    ".dm3", ".dm4", ".ser", ".mrc", ".emd",
    ".h5", ".hdf5",
    ".npy", ".npz", ".pkl", ".joblib",
    ".zip", ".tar", ".gz", ".tgz", ".7z", ".rar",
    ".mp4", ".mov", ".avi", ".mkv",
}

# Some special filenames that are useful even without extension
ALLOW_NO_EXT = {"readme", "license"}


# -------------------------
# Helpers
# -------------------------
def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def is_github_link(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {"github.com", "www.github.com"}


def classify_link(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "colab.research.google.com" in host:
        return "colab"
    if "drive.google.com" in host:
        return "gdrive"
    if host.endswith("box.com"):
        return "box"
    if is_github_link(url):
        return "github"
    return "other"


def normalize_github_clone_url(url: str) -> str:
    """
    Convert GitHub browse URLs to a cloneable URL.

    Handles:
      - trailing '?' or query params
      - /tree/<branch>/... or /blob/<branch>/...
      - already endswith .git
    """
    url = url.strip()
    url = url.split("?")[0]
    url = url.rstrip("/")

    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)(/.*)?$", url, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Not a GitHub URL: {url}")
    owner, repo = m.group(1), m.group(2)
    repo = repo.replace(".git", "")
    return f"https://github.com/{owner}/{repo}.git"


def bucket_for_file(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".ipynb":
        return "notebooks"
    if ext == ".pdf":
        return "docs"
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "assets"
    # Keep README / text docs at root for quick GitHub viewing
    if ext in {".md", ".txt", ".rst"} or Path(filename).name.lower() in {"readme", "license"}:
        return ""
    return "code"


def write_source(dst_dir: str, url: str):
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(dst_dir, "_SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write(url.strip() + "\n")


def ensure_readme(dst_dir: str, project_id: str, source_url: str):
    """
    If a README already exists (copied from upstream), keep it.
    Otherwise create a minimal, uniform README.
    """
    for name in ["README.md", "readme.md", "README.MD", "Readme.md"]:
        if os.path.exists(os.path.join(dst_dir, name)):
            # Normalize to README.md if different casing
            if name != "README.md":
                os.replace(os.path.join(dst_dir, name), os.path.join(dst_dir, "README.md"))
            return

    with open(os.path.join(dst_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {project_id}\n\n")
        f.write("Project archived for the AI/ML Microscopy Hackathon.\n\n")
        f.write("## Original submission\n")
        f.write(f"{source_url}\n")


def write_archive_note(dst_dir: str, project_id: str, source_url: str, skipped: list[str], kind: str):
    """
    Public-friendly archive note. Shows examples (not full logs).
    """
    note_path = os.path.join(dst_dir, "ARCHIVE_NOTE.md")
    with open(note_path, "w", encoding="utf-8") as fp:
        fp.write("# Archive Note\n\n")
        fp.write(
            "This folder contains a **lightweight archive** of the project for the "
            "AI/ML Microscopy Hackathon website and judging.\n\n"
        )

        fp.write("## What is included here\n")
        fp.write("- Notebooks (`.ipynb`)\n")
        fp.write("- Documentation and reports\n")
        fp.write("- Source code and small assets\n\n")

        fp.write("## Full project\n")
        fp.write("The complete project is available at:\n\n")
        fp.write(f"- {source_url}\n\n")

        if kind != "github":
            fp.write("## Note about this link\n")
            fp.write(
                f"This submission is provided as a **{kind}** link. "
                "To archive additional files here, export/download them and place them in this folder.\n"
            )
            return

        if skipped:
            fp.write("## Large files not included in this archive\n")
            fp.write(
                "Some large datasets, image stacks, trained model files, or generated environments "
                "are not stored in this website repository to keep it lightweight and accessible.\n\n"
            )
            fp.write("Examples of files not included:\n")
            # Show up to 20 examples (only paths, not technical reasons)
            shown = 0
            for entry in skipped:
                # entry looks like "REASON\tpath"
                parts = entry.split("\t", 1)
                path = parts[-1].strip()
                if not path:
                    continue
                fp.write(f"- `{path}`\n")
                shown += 1
                if shown >= 20:
                    break
            if len(skipped) > 20:
                fp.write(f"- … and {len(skipped) - 20} more\n")


def copy_tree_filtered(src_dir: str, dst_dir: str) -> list[str]:
    """
    Copy a repo into a uniform structure and return a list of skipped items
    as "REASON\\tRELATIVE_PATH".
    """
    os.makedirs(dst_dir, exist_ok=True)
    skipped: list[str] = []

    for root, dirs, files in os.walk(src_dir):
        # prune denied dirs by name (anywhere)
        dirs[:] = [d for d in dirs if d not in DENY_DIR_NAMES]

        for fname in files:
            src_f = os.path.join(root, fname)

            rel = os.path.relpath(src_f, src_dir)

            # skip symlinks
            if os.path.islink(src_f):
                skipped.append(f"SYMLINK\t{rel}")
                continue

            ext = Path(fname).suffix.lower()
            base_no_ext = Path(fname).name.lower()

            # deny by extension
            if ext in DENY_EXT:
                skipped.append(f"DISALLOWED_TYPE\t{rel}")
                continue

            # allowlist by extension (or special no-extension names)
            if ext:
                if ext not in ALLOW_EXT:
                    skipped.append(f"NOT_ARCHIVED_TYPE\t{rel}")
                    continue
            else:
                if base_no_ext not in ALLOW_NO_EXT:
                    skipped.append(f"NO_EXTENSION\t{rel}")
                    continue

            # size limit
            try:
                size = os.path.getsize(src_f)
            except OSError:
                skipped.append(f"STAT_FAIL\t{rel}")
                continue

            if size > MAX_BYTES:
                skipped.append(f"TOO_LARGE\t{rel}")
                continue

            bucket = bucket_for_file(fname)
            out_dir = os.path.join(dst_dir, bucket) if bucket else dst_dir
            os.makedirs(out_dir, exist_ok=True)

            dst_f = os.path.join(out_dir, fname)

            # If filename collision occurs (rare), keep first and skip others
            if os.path.exists(dst_f):
                skipped.append(f"NAME_COLLISION\t{rel}")
                continue

            shutil.copy2(src_f, dst_f)

    return skipped


def write_placeholder_project(dst_dir: str, project_id: str, url: str, kind: str):
    os.makedirs(dst_dir, exist_ok=True)
    write_source(dst_dir, url)
    ensure_readme(dst_dir, project_id, url)
    write_archive_note(dst_dir, project_id, url, skipped=[], kind=kind)


def read_repos_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        sample = fp.read(4096)
        fp.seek(0)
        # Your pasted file looked tab-separated; auto-detect TSV vs CSV.
        dialect = csv.excel_tab if ("\t" in sample and "," not in sample) else csv.excel
        reader = csv.DictReader(fp, dialect=dialect)

        required = {"project_id", "repo"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"repos_with_ids.csv must have columns {sorted(required)}. Found: {reader.fieldnames}"
            )
        rows = []
        for r in reader:
            project_id = (r.get("project_id") or "").strip()
            repo = (r.get("repo") or "").strip()
            if project_id and repo:
                rows.append({"project_id": project_id, "repo": repo})
        return rows


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(DEST_ROOT, exist_ok=True)

    try:
        rows = read_repos_csv(CSV_PATH)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    summary = {"github_ok": 0, "github_failed": 0, "non_github": 0}

    for r in rows:
        project_id = r["project_id"]
        url = r["repo"]

        dst_dir = os.path.join(DEST_ROOT, project_id)
        kind = classify_link(url)

        # Always start fresh for a project folder (avoid mixing old content)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        if kind != "github":
            write_placeholder_project(dst_dir, project_id, url, kind)
            summary["non_github"] += 1
            print(f"[{project_id}] non-github ({kind}) -> placeholder archive")
            continue

        # GitHub repo: clone then copy filtered
        try:
            clone_url = normalize_github_clone_url(url)
        except Exception as e:
            write_placeholder_project(dst_dir, project_id, url, "github_link_unparseable")
            summary["github_failed"] += 1
            print(f"[{project_id}] GitHub URL parse failed -> placeholder\n  {e}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"{project_id}_") as tmp:
            repo_dir = os.path.join(tmp, "repo")
            try:
                run(["git", "clone", "--depth", "1", clone_url, repo_dir])

                os.makedirs(dst_dir, exist_ok=True)
                write_source(dst_dir, url)

                skipped = copy_tree_filtered(repo_dir, dst_dir)

                ensure_readme(dst_dir, project_id, url)
                write_archive_note(dst_dir, project_id, url, skipped=skipped, kind="github")

                summary["github_ok"] += 1
                print(f"[{project_id}] imported (filtered) -> {dst_dir}")
            except Exception as e:
                # If clone/copy fails, fall back to placeholder
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                write_placeholder_project(dst_dir, project_id, url, "github_clone_failed")
                summary["github_failed"] += 1
                print(f"[{project_id}] clone/copy failed -> placeholder\n  {e}")

    print("\n=== Summary ===")
    print(f"GitHub imported: {summary['github_ok']}")
    print(f"GitHub failed:   {summary['github_failed']}")
    print(f"Non-GitHub:      {summary['non_github']}")


if __name__ == "__main__":
    main()
