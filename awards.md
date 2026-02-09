---
layout: page
title: Awards
menu_title: Awards
menu_icon: trophy
permalink: /awards/
published: true
sitemap: true
nav_exclude: False     # show in navigation
nav_order: 40          # adjust position in the menu (lower = earlier)
---

<style>
.section-card{
  background:#fafbfd;
  border:1px solid #e8ecf3;
  border-radius:14px;
  padding:26px 24px;
  margin:32px 0;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}
.section-card h2{
  font-size:1.3rem;
  color:#1d2a56;
  margin:0 0 10px 0;
  font-weight:600;
  border-left:4px solid #b4c8ff;
  padding-left:10px;
}
.grid-2{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
@media (max-width: 760px){ .grid-2{ grid-template-columns:1fr; } }
.table-soft{ width:100%; border-collapse:separate; border-spacing:0 6px; }
.table-soft th{ text-align:left; font-weight:700; font-size:.95rem; color:#344054; padding:10px 12px; }
.table-soft td{ background:#fff; border:1px solid #eef0f5; border-radius:10px; padding:12px; }
.logo-row{ display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; margin-top:10px; }
.logo-row img{ max-height:56px; width:auto; object-fit:contain; background:#fff; padding:.25rem .4rem; border-radius:10px; border:1px solid #eef0f5; }
.private-banner{
  background:#fff8e1; border:1px solid #ffe58f; border-radius:10px;
  padding:10px 14px; color:#7a5900; font-size:.9rem; margin-top:10px;
}
</style>

<div class="section-card">
  <h2>Awards & Winners</h2>

  <p>
    The following awards recognize outstanding hackathon projects, based on judging scores
    and sponsor selections. Winning teams and projects are listed below.
  </p>

<table class="table-soft">
  <thead>
    <tr>
      <th>Sponsor</th>
      <th>Prize</th>
      <th>Winner</th>
    </tr>
  </thead>

  <tbody>
    <!-- Renaissance Philanthropy (2 rows) -->
    <tr>
      <td rowspan="2" style="text-align:center; vertical-align:middle;">
        <a href="https://www.renaissancephilanthropy.org" target="_blank" rel="noopener">
          <img src="{{ '/assets/renaissance.png' | relative_url }}" alt="Renaissance Philanthropy" style="max-height:80px;">
        </a>
      </td>
      <td><strong>Overall Winner</strong> ($7,000)</td>
      <td><strong>H-051</strong> — Interpretable Digital Twins for Autonomous STEM Aberration Correction</td>
    </tr>
    <tr>
      <td><strong>2nd Overall Winner</strong> ($3,000)</td>
      <td><strong>H-065</strong> — Autonomous Identification of Metal Microstructural Features via Latent Space Mapping-Based Microscope Control</td>
    </tr>

    <!-- Covalent Metrology (2 rows) -->
    <tr>
      <td rowspan="2" style="text-align:center; vertical-align:middle;">
        <a href="https://covalentmetrology.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/covalent.png' | relative_url }}" alt="Covalent Metrology" style="max-height:55px;">
        </a>
      </td>
      <td><strong>2nd Place</strong> ($3,000)</td>
      <td><strong>H-068</strong> — TwinSpec: A Digital Twin Framework for GIWAXS Data, Geometry, and Physics-Aware ML</td>
    </tr>
    <tr>
      <td><strong>3rd Place</strong> ($2,000)</td>
      <td><strong>H-064</strong> — RONIN - Ronchigram based Optical Neural Inference for aberration detection</td>
    </tr>

    <!-- Theia Scientific -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://theiascientific.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/TheiaScientific.png' | relative_url }}" alt="Theia Scientific" style="max-height:60px;">
        </a>
      </td>
      <td>$1,000</td>
      <td><strong>H-043</strong> — Automated particle detection and quantitative analysis from electron microscopy images</td>
    </tr>

    <!-- Toyota Research Institute -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://www.tri.global" target="_blank" rel="noopener">
          <img src="{{ '/assets/TRI.png' | relative_url }}" alt="Toyota Research Institute" style="max-height:55px;">
        </a>
      </td>
      <td>$1,000 (Best Interactivity)</td>
      <td><strong>H-022</strong> — DeepScan Pro: Intelligent Microscopy Agent</td>
    </tr>
    

    <!-- Oxford Instruments - Asylum Research -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://afm.oxinst.com/" target="_blank" rel="noopener">
          <img src="{{ '/assets/asylum.png' | relative_url }}" alt="Oxford Instruments - Asylum Research" style="max-height:55px;">
        </a>
      </td>
      <td>$1,000 (Best AFM/SPM Project)</td>
      <td><strong>H-035</strong> — Angle-Dependent Morphologies of Ferroelectric Domain Walls</td>
    </tr>

    <!-- Polaron (2 rows) -->
    <tr>
      <td rowspan="2" style="text-align:center; vertical-align:middle;">
        <a href="https://polaron.ai" target="_blank" rel="noopener">
          <img src="{{ '/assets/polaron.png' | relative_url }}" alt="Polaron" style="max-height:55px;">
        </a>
      </td>
      <td><strong>Award Winner</strong> ($1,000 + platform access)</td>
      <td><strong>H-065</strong> — Autonomous Identification of Metal Microstructural Features via Latent Space Mapping-Based Microscope Control</td>
    </tr>
    <tr>
      <td><strong>Runner-up</strong> (platform access)</td>
      <td><strong>H-042</strong> — EM-Caddie</td>
    </tr>

    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://www.sciencedirect.com/journal/computational-materials-science" target="_blank" rel="noopener">
          <img src="{{ '/assets/commat.png' | relative_url }}" alt="COMMAT" style="max-height:55px;">
        </a>
      </td>
      <td>$500 (Canada Regional Winner)</td>
      <td><strong>H-014</strong> — VLRIMM - Vision-Language Retrieval for Identical Materials Morphology</td>
    </tr>

    <!-- JEOL (note: your winners list says $1,000; your old table said $500) -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://jeol.com/" target="_blank" rel="noopener">
          <img src="{{ '/assets/Jeol.png' | relative_url }}" alt="JEOL" style="max-height:55px;">
        </a>
      </td>
      <td>$1,000</td>
      <td><strong>H-041</strong> — Machine Learning Denoising of Reciprocal Space Maps for realistic center-of-mass evaluation</td>
    </tr>
  

    <!-- MSA Student Council -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://microscopy.org/the-student-council-stc/" target="_blank" rel="noopener">
          <img src="{{ '/assets/msa_stc.png' | relative_url }}" alt="MSA Student Council" style="max-height:55px;">
        </a>
      </td>
      <td>$500 + Student registration to MSA 2026 PMC (Milwaukee)</td>
      <td><strong>H-014</strong> — VLRIMM - Vision-Language Retrieval for Identical Materials Morphology</td>
    </tr>

    <!-- Waviks -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://www.waviks.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/waviks.png' | relative_url }}" alt="Waviks" style="max-height:50px;">
        </a>
      </td>
      <td>$500</td>
      <td><strong>H-042</strong> — EM-Caddie</td>
    </tr>

    <!-- Hugging Face -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://huggingface.co" target="_blank" rel="noopener">
          <img src="{{ '/assets/hf.png' | relative_url }}" alt="Hugging Face" style="max-height:50px;">
        </a>
      </td>
      <td>Merch + 2 months Hugging Face Pro</td>
      <td><strong>H-002</strong> — Contrastive Micrograph-Metadata Pre-Training</td>
    </tr>

    <!-- Mat3ra -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://mat3ra.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/mat3ra_logo.png' | relative_url }}" alt="Mat3ra" style="max-height:55px;">
        </a>
      </td>
      <td>Compute credits worth $5,000</td>
      <td><strong>H-051</strong> — Interpretable Digital Twins for Autonomous STEM Aberration Correction</td>
    </tr>

    <!-- DENS Solutions -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://denssolutions.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/DENS.png' | relative_url }}" alt="DENS Solutions" style="max-height:50px;">
        </a>
      </td>
      <td>Mystery Prize (Gin-Situ)</td>
      <td><strong>H-043</strong> — Automated particle detection and quantitative analysis from electron microscopy images</td>
    </tr>

    <!-- Hitachi High-Tech Canada (updated from your winners list) -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://www.hitachi-hightech.com/ca/en/" target="_blank" rel="noopener">
          <img src="{{ '/assets/hitachi.png' | relative_url }}" alt="Hitachi High-Tech Canada" style="max-height:55px;">
        </a>
      </td>
      <td>$300 (Canada Regional Runner-up)</td>
      <td><strong>H-062</strong> — ANCHOR: Registration by Alignment</td>
    </tr>

    <!-- MENA / Qatar Award (AISCIA) -->
    <tr>
      <td style="text-align:center; vertical-align:middle;">
        <a href="https://aiscia.com" target="_blank" rel="noopener">
          <img src="{{ '/assets/aiscia.png' | relative_url }}" alt="AISCIA" style="max-height:55px;">
        </a>
      </td>
      <td>MENA Regional Award</td>
      <td><strong>H-011</strong> — Digital Bubble Stability Tracking Microscope</td>
    </tr>

  </tbody>
</table>

</div>

<div class="section-card" style="text-align:center;">
  <em>These prizes celebrate creativity, collaboration, and the spirit of open science at the Hackathon.</em>
</div>
