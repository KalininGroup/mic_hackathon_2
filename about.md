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
/* --- Core Team cards --- */
.team-grid{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap:16px;
}
.team-card{
  display:flex; gap:12px; align-items:flex-start;
  background:#fff; border:1px solid #eef0f5; border-radius:14px;
  padding:14px;
}
.team-card img{
  width:64px; height:64px; border-radius:12px; object-fit:cover;
  border:1px solid #e8ecf3; background:#fff;
}
.team-meta .name{ font-weight:700; color:#1d2a56; }
.team-meta .affil{ font-size:.92rem; color:#64748b; }
.team-meta .role{ font-size:.92rem; color:#475569; }
.socials{ margin-top:8px; display:flex; flex-wrap:wrap; gap:8px; }
.socials a{
  display:inline-flex; align-items:center; gap:6px;
  font-size:.9rem; color:#1d4ed8; text-decoration:none;
  border:1px solid #e5e7eb; padding:4px 8px; border-radius:999px;
}
.socials a:hover{ background:#eef2ff; }
.socials svg{ width:16px; height:16px; }

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

  <div class="team-grid">

    <!-- TEMPLATE: copy one .team-card per person -->
    <!-- Put headshots in /assets/people/ and update filenames/links -->
    <div class="team-card">
      <img src="{{ '/assets/svk.png' | relative_url }}" alt="Sergei V. Kalinin">
      <div class="team-meta">
        <div class="name">Sergei V. Kalinin</div>
        <div class="affil">University of Tennessee, Knoxville, Pacific Northwest National Laboratory</div>

        <div class="socials">
          <a href="https://github.com/SergeiVKalinin" target="_blank" rel="noopener">
            <!-- GitHub icon -->
            <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0C16.8 5.5 17.8 5.8 17.8 5.8a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
            <span>GitHub</span>
          </a>
          <a href="[https://WEBSITE.URL](https://ae-spm.utk.edu/group-leader-pi/)" target="_blank" rel="noopener">
            <!-- Globe icon -->
            <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2Zm7.9 9h-3.2a14.5 14.5 0 0 0-1.1-5 8.03 8.03 0 0 1 4.3 5ZM12 4c.9 0 2.4 2.1 3 6h-6c.6-3.9 2.1-6 3-6Zm-3.6 1a14.5 14.5 0 0 0-1.1 5H4.1a8.03 8.03 0 0 1 4.3-5Zm-4.3 7h3.2c.1 1.8.5 3.5 1.1 5a8.03 8.03 0 0 1-4.3-5Zm7.9 8c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.6-1a14.5 14.5 0 0 0 1.1-5h3.2a8.03 8.03 0 0 1-4.3 5Z"/></svg>
            <span>Website</span>
          </a>
          <!-- Optional email button -->
          <a href="mailto:sergei2@utk.edu">
            <!-- Mail icon -->
            <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2Zm0 4-8 5L4 8V6l8 5 8-5v2Z"/></svg>
            <span>Email</span>
          </a>
        </div>
      </div>
    </div>

    <div class="team-card">
      <img src="{{ '/assets/GD.png' | relative_url }}" alt="Gerd Duscher">
      <div class="team-meta">
        <div class="name">Gerd Duscher</div>
        <div class="affil">University of Tennessee, Knoxville</div>
        
        <div class="socials">
          <a href="https://github.com/gduscher" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
            <span>GitHub</span>
          </a>
          <a href="[https://WEBSITE.URL](https://tickle.utk.edu/mse/faculty/gerd-duscher/)" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2Zm7.9 9h-3.2a14.5 14.5 0 0 0-1.1-5 8.03 8.03 0 0 1 4.3 5ZM12 4c.9 0 2.4 2.1 3 6h-6c.6-3.9 2.1-6 3-6Zm-3.6 1a14.5 14.5 0 0 0-1.1 5H4.1a8.03 8.03 0 0 1 4.3-5Zm-4.3 7h3.2c.1 1.8.5 3.5 1.1 5a8.03 8.03 0 0 1-4.3-5Zm7.9 8c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.6-1a14.5 14.5 0 0 0 1.1-5h3.2a8.03 8.03 0 0 1-4.3 5Z"/></svg>
            <span>Website</span>
          </a>
        </div>
      </div>
    </div>



    <div class="team-card">
      <img src="{{ '/assets/people/rama.png' | relative_url }}" alt="Rama Vasudevan">
      <div class="team-meta">
        <div class="name">Rama Vasudevan</div>
        <div class="affil">CNMS, Oak Ridge National Laboratory</div>
        <div class="socials">
          <a href="https://github.com/ramav87" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
            <span>GitHub</span>
          </a>
          <a href="https://www.ornl.gov/staff-profile/rama-k-vasudevan" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2Zm7.9 9h-3.2a14.5 14.5 0 0 0-1.1-5 8.03 8.03 0 0 1 4.3 5ZM12 4c.9 0 2.4 2.1 3 6h-6c.6 3.9-2.1 6-3 6Zm-3.6 1a14.5 14.5 0 0 0-1.1 5H4.1a8.03 8.03 0 0 1 4.3-5Zm-4.3 7h3.2c.1 1.8.5 3.5 1.1 5a8.03 8.03 0 0 1-4.3-5Zm7.9 8c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.6-1a14.5 14.5 0 0 0 1.1-5h3.2a8.03 8.03 0 0 1-4.3 5Z"/></svg>
            <span>Website</span>
          </a>
        </div>
      </div>
    </div>

    <!-- ADD NEW PEOPLE HERE: duplicate a .team-card and update fields -->
    <!-- Example new person -->
    <div class="team-card">
      <img src="{{ '/assets/rliu.png' | relative_url }}" alt="Richard Liu">
      <div class="team-meta">
        <div class="name">Richard Liu</div>
        <div class="affil">Postdoctoral Researcher, University of Tennessee, Knoxville</div>
        <div class="socials">
          <a href="https://github.com/RichardLiuCoding" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
        </div>
      </div>
    </div>

        <div class="team-card">
      <img src="{{ '/assets/Boris.png' | relative_url }}" alt="Boris Slautin">
      <div class="team-meta">
        <div class="name">Boris Slautin</div>
        <div class="affil">Postdoctoral Researcher, University of Tennessee, Knoxville</div>
        <div class="socials">
          <a href="https://github.com/Slautin" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
          <span>GitHub</span></a>
        </div>
      </div>
    </div>

    <div class="team-card">
      <img src="{{ '/assets/up.png' | relative_url }}" alt="Utkarsh Pratiush">
      <div class="team-meta">
        <div class="name">Utkarsh Pratiush</div>
        <div class="affil">University of Tennessee, Knoxville</div>
        <div class="socials">
          <a href="https://github.com/utkarshp1161" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.2.8-.5v-2c-3.3.7-4-1.4-4-1.4-.6-1.3-1.4-1.6-1.4-1.6-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.9 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.6-1.3-5.6-6a4.7 4.7 0 0 1 1.2-3.2 4.3 4.3 0 0 1 .1-3.1s1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2a4.3 4.3 0 0 1 .1 3.1 4.7 4.7 0 0 1 1.2 3.2c0 4.7-2.9 5.7-5.6 6 .4.3.8 1 .8 2.1v3.1c0 .3.2.6.8.5A12 12 0 0 0 12 .5Z"/></svg>
            <span>GitHub</span>
          </a>
          <a href="[https://WEBSITE.URL](https://utkarshp1161.github.io/UtkarshsAIInScience.github.io/)" target="_blank" rel="noopener">
            <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 .001 20.001A10 10 0 0 0 12 2Zm7.9 9h-3.2a14.5 14.5 0 0 0-1.1-5 8.03 8.03 0 0 1 4.3 5ZM12 4c.9 0 2.4 2.1 3 6h-6c.6 3.9 2.1 6 3 6Zm-3.6 1a14.5 14.5 0 0 0-1.1 5H4.1a8.03 8.03 0 0 1 4.3-5Zm-4.3 7h3.2c.1 1.8.5 3.5 1.1 5a8.03 8.03 0 0 1-4.3-5Zm7.9 8c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.6-1a14.5 14.5 0 0 0 1.1-5h3.2a8.03 8.03 0 0 1-4.3 5Z"/></svg>
            <span>Website</span>
          </a>
        </div>
      </div>
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
