---
layout: page
title: About the Hackathon
menu_title: About
menu_icon: info-circle
permalink: /about/
published: true
---

<style>
/* Same card look as Home */
.about-wrap, .about-wrap * { box-sizing: border-box; }
.section-card{
  background:#fff; border:1px solid #e6e6e6; border-radius:14px;
  padding:1rem 1.1rem; margin: 1rem 0 1.25rem;
  box-shadow:0 6px 18px rgba(0,0,0,.05);
}
.section-card h2{
  margin:.1rem 0 .6rem; font-size:1.15rem; line-height:1.3;
}
.section-card p{ margin:.4rem 0; }

.grid-2{ display:grid; gap:1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
.badge{ display:inline-block; font-size:.78rem; padding:.18rem .5rem; border-radius:999px; background:#eef2ff; color:#334155; border:1px solid #c7d2fe; }

/* Team */
.team-grid{ display:grid; gap:.9rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.team-card{
  border:1px solid #e9e9e9; border-radius:12px; padding:.8rem .9rem; background:#fafafa;
}
.team-card h3{ font-size:1rem; margin:0 0 .2rem; }
.team-card .affil{ color:#555; font-size:.95rem; margin:0 0 .3rem; }
.team-card .role{ color:#6b7280; font-style:italic; font-size:.92rem; }

/* Partners */
.logo-row{ display:flex; flex-wrap:wrap; gap:.8rem 1rem; align-items:center; }
.logo-row img{ max-height:56px; width:auto; height:auto; object-fit:contain; background:#fff; padding:.25rem .4rem; border-radius:10px; border:1px solid #eee; }

/* Buttons/links to match Home */
.btn{ display:inline-block; padding:.5rem .8rem; border-radius:10px; text-decoration:none; font-weight:600; border:1px solid #d6d6d6; background:#fafafa; }
.btn:hover{ border-color:#3a7bd5; box-shadow:0 4px 12px rgba(58,123,213,.18); }
.btn-primary{ border:1px solid #2e6bd6; background:#3a7bd5; color:#fff; }
</style>

<div class="about-wrap">

  <div class="section-card">
    <h2>Our Mission</h2>
    <p>
      The <strong>Machine Learning for Microscopy Hackathon</strong> accelerates the use of AI in microscopy and materials research.
      By connecting the microscopy and machine-learning communities, we foster collaboration, open science, and the development of
      reproducible AI workflows for imaging, spectroscopy, and automated experimentation.
    </p>
  </div>

  <div class="section-card">
    <h2>Hackathon History</h2>
    <p>
      The first edition took place at the University of Tennessee, Knoxville, in 2024, bringing together over 250 registrants and
      80+ active participants from around the world. Building on that success, the <span class="badge">2025</span> edition expands
      into a <em>multi-site hybrid event</em> across universities and research centers, enabling real-time collaboration and global participation.
    </p>
  </div>

  <div class="section-card">
    <h2>Core Organizing Team</h2>
    <div class="team-grid">

      <div class="team-card">
        <h3>Sergei V. Kalinin</h3>
        <div class="affil">University of Tennessee, Knoxville</div>
        <div class="role">Lead Organizer</div>
      </div>

      <div class="team-card">
        <h3>Gerd Duscher</h3>
        <div class="affil">University of Tennessee, Knoxville</div>
        <div class="role">Education &amp; Outreach</div>
      </div>

      <div class="team-card">
        <h3>Utkarsh Pratiush</h3>
        <div class="affil">University of Tennessee, Knoxville</div>
        <div class="role">Site Coordination</div>
      </div>

      <div class="team-card">
        <h3>Rama Vasudevan</h3>
        <div class="affil">Center for Nanophase Materials Sciences, Oak Ridge National Laboratory</div>
        <div class="role">Software &amp; Infrastructure</div>
      </div>

    </div>
  </div>

  <div class="section-card">
    <h2>Partners &amp; Support</h2>
    <p>
      Supported by the <strong>AI Tennessee Initiative</strong>, the <strong>Center for Advanced Materials &amp; Manufacturing (CAMM)</strong>,
      and collaborating institutions (UTK, NCSU, NWU, UIC/ANL, ICN2, and others).
    </p>
    <p class="logo-row">
      <!-- Replace with your actual paths; examples below -->
      <img src="{{ '/assets/ONR.png' | relative_url }}" alt="ONR">
      <img src="{{ '/assets/matora.png' | relative_url }}" alt="Matora">
      <!-- Add more logos as needed -->
    </p>
  </div>

  <div class="section-card">
    <h2>Get Involved</h2>
    <p>
      • Join the Slack group to connect with participants.<br>
      • Propose or join a project on the Submission page.<br>
      • Interested in sponsoring or hosting a site? Contact <a href="mailto:sergei2@utk.edu">sergei2@utk.edu</a>.
    </p>
    <p>
      <a class="btn" href="{{ '/submission/' | relative_url }}">Submission Page</a>
      <a class="btn btn-primary" href="{{ '/registration/' | relative_url }}">Register</a>
    </p>
  </div>

</div>
