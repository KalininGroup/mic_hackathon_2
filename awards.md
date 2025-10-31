---
layout: page
title: Awards
menu_title: Awards
menu_icon: trophy
permalink: /awards/
published: true          # Page will be built so you can see it
sitemap: false           # Not indexed in sitemap.xml
nav_exclude: true        # Hidden from navigation/menu
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

<div class="private-banner">
  🔒 <strong>Private Preview:</strong> This Awards page is hidden from the public navigation and search results.
  Only maintainers with the direct link can access it.
</div>

<div class="section-card">
  <h2>Prize Breakdown</h2>

  <p>
    Awards recognize outstanding work across categories such as <em>Best Overall Project</em>,
    <em>Best Open Science</em>, <em>Best Methods</em>, and <em>Best Visualization</em>.
    Final prize assignments will be announced during the hackathon.
  </p>

  <table class="table-soft">
    <thead><tr><th>Sponsor</th><th>Prize</th><th>Notes</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>DENS Solutions</strong></td>
        <td>Mystery Prize</td>
        <td>Awarded to the <strong>Overall Winner</strong> of the hackathon.</td>
      </tr>
      <tr>
        <td><strong>Theia Scientific</strong></td>
        <td>$1,000</td>
        <td>Sponsors one of the main award categories (to be announced at the final judging).</td>
      </tr>
      <tr>
        <td><strong>Thermo Fisher Scientific</strong></td>
        <td>$1,000</td>
        <td>Sponsors one of the main award categories (to be announced at the final judging).</td>
      </tr>
      <tr>
        <td><strong>Hugging Face</strong></td>
        <td>Merchandise</td>
        <td>May accompany another prize or serve as a special recognition award.</td>
      </tr>
    </tbody>
  </table>

  <p style="margin-top:8px; color:#475467;">
    <em>Exact prize-to-category mapping will be finalized based on submissions and judging outcomes.</em><br>
    All prizes will be presented during the <strong>Final Showcase on December 18, 2025</strong>.
  </p>
</div>

<div class="section-card">
  <h2>Award Categories</h2>
  <div class="grid-2">
    <div>
      <strong>Best Overall Project</strong><br>
      Highest-scoring project across all metrics — technical excellence, impact, and presentation.
    </div>
    <div>
      <strong>Category 1</strong><br>
      Description
    </div>
    <div>
      <strong>Category 2</strong><br>
      Description
    </div>
    <div>
      <strong>Student Awards</strong><br>
      Description
    </div>
  </div>
</div>

<div class="section-card">
  <h2>Sponsors</h2>
  <p>
    The 2025 Hackathon awards are made possible by the generous support of our sponsors and partners:
  </p>
  <div class="logo-row">
    <img src="{{ '/assets/TheiaScientific.png' | relative_url }}" alt="Theia Scientific logo">
    <img src="{{ '/assets/tf_logo.png' | relative_url }}" alt="Thermo Fisher Scientific logo">
    <img src="{{ '/assets/HuggingFace.png' | relative_url }}" alt="Hugging Face logo">
    <img src="{{ '/assets/DENS.png' | relative_url }}" alt="DENS Solutions logo">
  </div>
</div>

<div class="section-card" style="text-align:center;">
  <em>These prizes celebrate creativity, collaboration, and the spirit of open science at the Hackathon.</em>
</div>
