#!/usr/bin/env python3
"""
Download Google Docs / Drive files into project folders.

- Google Docs -> exported as PDF
- Google Drive files -> downloaded as-is
- Drive folders -> all files downloaded
- Unsupported links -> noted in ARCHIVE_NOTE.md
"""

import csv
import os
import subprocess
import sys
from urllib.parse import urlparse

ROOT = os.getcwd()
PROJECTS_DIR = os.path.join(ROOT, "projects")
CSV_PATH = os.path.join(ROOT, "project_docs.csv")

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    return p.stdout.strip()

def ensure_docs_dir(pid):
    d = os.path.join(PROJECTS_DIR, pid, "docs")
    os.makedirs(d, exist_ok=True)
    return d

def append_note(pid, text):
    note = os.path.join(PROJECTS_DIR, pid, "ARCHIVE_NOTE.md")
    with open(note, "a", encoding="utf-8") as f:
        f.write("\n\n" + text + "\n")

def is_google_doc(url):
    return "docs.google.com/document" in url

def is_drive_file(url):
    return "drive.google.com/file" in url

def is_drive_folder(url):
    return "drive.google.com/drive/folders" in url

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Missing {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["project_id"].strip()
            url = row["doc_link"].strip()

            if not pid or not url:
                continue

            docs_dir = ensure_docs_dir(pid)
            print(f"[{pid}] processing document")

            try:
                if is_google_doc(url):
                    # Export Google Doc as PDF
                    run([
                        "gdown",
                        "--format=pdf",
                        "--output", os.path.join(docs_dir, "report.pdf"),
                        url
                    ])
                elif is_drive_file(url) or is_drive_folder(url):
                    run([
                        "gdown",
                        "--folder" if is_drive_folder(url) else "",
                        "--output", docs_dir,
                        url
                    ])
                else:
                    append_note(pid, f"⚠️ Project document could not be auto-downloaded:\n{url}")
            except Exception as e:
                append_note(pid, f"⚠️ Failed to download document:\n{url}\nReason: {e}")

if __name__ == "__main__":
    main()
