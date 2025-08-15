# Microscopy Hackathon – Edition 2

This is a minimal, static site for GitHub Pages. Edit the JSON files in `data/` to fill the agenda, projects, results, and media lists. Replace text in `index.html` as needed.

## Quick start (GitHub Pages)

1. Create a new public repo (e.g., `mic-hackathon-2`).  
2. Upload the files from this folder (or drag‑and‑drop in GitHub web UI).  
3. Go to **Settings → Pages**.  
   - Source: **Deploy from a branch**  
   - Branch: **main** / root  
4. Open the URL GitHub shows (e.g., `https://<user-or-org>.github.io/mic-hackathon-2/`).

## Customize

- Put your logo at `assets/logo.png` (optional).  
- Update buttons in the hero section (`Register`, `Call for Projects`).  
- Fill `data/*.json` with your content (structure is self‑explanatory).  
- Tweak styles in `style.css`.

## Local preview

Open `index.html` in a browser. Since we load JSON via `fetch`, some browsers block file URLs. If so, run a tiny server:

```bash
python3 -m http.server 8080
# then visit http://localhost:8080
```

— Generated 2025-08-15
