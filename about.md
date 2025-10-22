---
layout: page
title: About the Hackathon
menu_title: About
menu_icon: info-circle
permalink: /about/
published: true
---

<!-- reuse the same styles as Home so both pages match -->
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
  margin-top:0;
  margin-bottom:10px;
  font-weight:600;
  border-left:4px solid #b4c8ff;
  padding-left:10px;
}
.grid-2{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }
@media (max-width: 760px){ .grid-2{ grid-template-columns:1fr; } }
.table-soft{
  width:100%; border-collapse:separate; border-spacing:0 6px;
}
.table-soft th{ text-align:left; font-weight:700; font-size:.95rem; color:#344054; padding:10px 12px; }
.table-soft td{ background:#fff; border:1px solid #eef0f5; border-radius:10px; padding:12px; }

/* tiny helpers to mirror Home feel */
.badge{ display:inline-block; font-size:.8rem; padding:.2rem .5rem; border-radius:999px; background:#eef2ff; color:#334155; border:1px solid #c7d2fe; }
.logo-row{ display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; }
.logo-row img{ max-height:56px; width:auto; height:auto; object-fit:contain; background:#fff; padding:.25rem .4rem; border-radius:10px; border:1px solid #eef0f5; }
</style>

<div class="section-card">
  <h2>Our Mission</h2>
  <p>
    The <strong>Machine Learning for Microscopy Hackathon</strong> accelerates the use of AI in microscopy and materials research.
    By connecting microscopy and machine-learning communities, we foster collaboration, open science, and reproducible workflows
    for imaging, spectroscopy, and automated experimentation.
  </p>
</div>

<div class="section-card">
  <h2>Hackathon History</h2>
  <p>
    The first edition took place at the University of Tennessee, Knoxville, in 2024, with 250+ registrants and 80+ active participants
    from around the world. Building on that success, the <span class="badge">2025</span> edition expands into a <em>multi-site hybrid event</em>,
    enabling real-time collaboration across universities and research centers.
  </p>
</div>

<div class="section-card">
  <h2>Core Organizing Team</h2>
  <div class="grid-2">
    <div>
      <strong>Sergei V. Kalinin</strong><br>
      University of Tennessee, Knoxville<br>
      <em>Lead Organizer</em>
    </div>
    <div>
      <strong>Gerd Duscher</strong><br>
      University of Tennessee, Knoxville<br>
      <em>Education &amp; Outreach</em>
    </div>
    <div>
      <strong>Utkarsh Pratiush</strong><br>
      University of Tennessee, Knoxville<br>
      <em>Site Coordination</em>
    </div>
    <div>
      <strong>Rama Vasudevan</strong><br>
      Center for Nanophase Materials Sciences, ORNL<br>
      <em>Software &amp; Infrastructure</em>
    </div>
  </div>
</div>

<div class="section-card">
  <h2>How It Works</h2>
  <table class="table-soft">
    <thead><tr><th>Element</th><th>What to expect</th></tr></thead>
    <tbody>
      <tr><td>Teams</td><td>Interdisciplinary groups mixing microscopy and ML backgrounds; remote and on-site collaboration.</td></tr>
      <tr><td>Data & Tasks</td><td>Real microscopy datasets and challenges spanning imaging, spectroscopy, and automation.</td></tr>
      <tr><td>Mentorship</td><td>Guidance from domain experts and tool builders; cross-site office hours and Slack support.</td></tr>
      <tr><td>Outcomes</td><td>Working prototypes, analysis notebooks, and open-source contributions that persist post-event.</td></tr>
    </tbody>
  </table>
</div>

<div class="section-card">
  <h2>Partners &amp; Support</h2>

  <p>
    The hackathon is supported by the <strong>AI Tennessee Initiative</strong> and the
    <strong>Center for Advanced Materials &amp; Manufacturing (CAMM)</strong>, with participation from
    <strong>UTK, NCSU, NWU, UIC/ANL, ICN2, and IIT Delhi</strong>.
  </p>

  <p><strong>Sponsors</strong></p>
  <p>
    Our primary sponsors provide critical support for enabling open, collaborative AI-driven microscopy:
  </p>
  <div class="logo-row" style="margin-top:10px;">
    <img src="{{ '/assets/ONR.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/tf_logo.png' | relative_url }}"  style="max-height:64px;">
  </div>

  <p style="margin-top:20px;"><strong>Partners</strong></p>
  <p>
    Our partners empower open science and innovation in AI for materials research:
  </p>
  <div class="logo-row" style="margin-top:10px;">
    <img src="{{ '/assets/mat3ra_logo.png' | relative_url }}" alt="Mat3ra – Materials R&D Cloud logo">
    <img src="{{ '/assets/hf.png' | relative_url }}" alt="Hugging Face logo">
  </div>
</div>



<div class="section-card">
  <h2>Get Involved</h2>
  <div class="grid-2">
    <div>
      <strong>Join the Community</strong><br>
      Connect on Slack, meet collaborators, and find a team.
      <div style="margin-top:6px;"><a href="https://tiny.utk.edu/slack">Join the Slack workspace</a></div>
    </div>
    <div>
      <strong>Participate</strong><br>
      Register for the hackathon and choose your preferred site (or Online).
      <div style="margin-top:6px;"><a href="{{ '/registration/' | relative_url }}">Register</a></div>
    </div>
  </div>
</div>

<div class="section-card" style="text-align:center;">
  <em>We are building an open, collaborative future for AI-driven microscopy — join us.</em>
</div>
