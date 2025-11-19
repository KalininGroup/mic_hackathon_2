---
layout: page
title: About the Hackathon
menu_title: About
menu_icon: info-circle
permalink: /about/
published: true
---

<style>
.section-card {
  background: #fafbfd;
  border: 1px solid #e8ecf3;
  border-radius: 14px;
  padding: 26px 24px;
  margin: 32px 0;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}

.section-card h2 {
  font-size: 1.3rem;
  color: #1d2a56;
  margin-top: 0;
  margin-bottom: 10px;
  font-weight: 600;
  border-left: 4px solid #b4c8ff;
  padding-left: 10px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

@media (max-width: 760px) {
  .grid-2 { grid-template-columns: 1fr; }
}

.table-soft {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0 6px;
}

.table-soft th {
  text-align: left;
  font-weight: 700;
  font-size: .95rem;
  color: #344054;
  padding: 10px 12px;
}

.table-soft td {
  background: #fff;
  border: 1px solid #eef0f5;
  border-radius: 10px;
  padding: 12px;
}

/* --- small helpers --- */
.badge {
  display: inline-block;
  font-size: .8rem;
  padding: .2rem .5rem;
  border-radius: 999px;
  background: #eef2ff;
  color: #334155;
  border: 1px solid #c7d2fe;
}

.logo-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  align-items: center;
}

.logo-row img {
  max-height: 56px;
  width: auto;
  height: auto;
  object-fit: contain;
  background: #fff;
  padding: .25rem .4rem;
  border-radius: 10px;
  border: 1px solid #eef0f5;
}

/* --- Core Team cards --- */
.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.team-card {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: #fff;
  border: 1px solid #eef0f5;
  border-radius: 14px;
  padding: 14px;
}

.team-card img {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  object-fit: cover;
  border: 1px solid #e8ecf3;
  background: #fff;
}

.team-meta .name {
  font-weight: 700;
  color: #1d2a56;
}

.team-meta .affil {
  font-size: .92rem;
  color: #64748b;
}

.team-meta .role {
  font-size: .92rem;
  color: #475569;
}

/* --- clean inline social icons beside names --- */
.socials {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-left: 6px;
  margin-bottom: -2px;
}

.socials a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #1d4ed8;
  text-decoration: none;
  background: none;
  border: none;
  box-shadow: none;
  width: auto;
  height: auto;
  padding: 0;
  transition: color 0.2s ease;
}

.socials a:hover {
  color: #0f172a;
}

.socials svg {
  width: 17px;
  height: 17px;
  fill: currentColor;
  vertical-align: middle;
}
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

    <!-- Sergei -->
    <div class="team-card">
      <img src="{{ '/assets/svk.png' | relative_url }}" alt="Sergei V. Kalinin">
      <div class="team-meta">
        <div class="name">
          Sergei V. Kalinin
          <span class="socials">
            <!-- Website -->
            <a href="https://ae-spm.utk.edu/group-leader-pi/" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <!-- GitHub -->
            <a href="https://github.com/SergeiVKalinin" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
            <!-- Email -->
            <a href="mailto:sergei2@utk.edu" title="Email" aria-label="Email">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M2 6.75A2.75 2.75 0 0 1 4.75 4h14.5A2.75 2.75 0 0 1 22 6.75v10.5A2.75 2.75 0 0 1 19.25 20H4.75A2.75 2.75 0 0 1 2 17.25V6.75Zm2.02-.36 7.48 4.67c.3.19.7.19 1 0l7.48-4.67a.75.75 0 1 1 .78 1.28l-8 5a1.75 1.75 0 0 1-1.82 0l-8-5a.75.75 0 1 1 .78-1.28Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">University of Tennessee, Knoxville; Pacific Northwest National Laboratory</div>
        
      </div>
    </div>

    <!-- Gerd -->
    <div class="team-card">
      <img src="{{ '/assets/GD.png' | relative_url }}" alt="Gerd Duscher">
      <div class="team-meta">
        <div class="name">
          Gerd Duscher
          <span class="socials">
            <a href="https://tickle.utk.edu/mse/faculty/gerd-duscher/" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <a href="https://github.com/gduscher" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">University of Tennessee, Knoxville</div>
        
      </div>
    </div>

    <!-- Rama -->
    <div class="team-card">
      <img src="{{ '/assets/rama.png' | relative_url }}" alt="Rama Vasudevan">
      <div class="team-meta">
        <div class="name">
          Rama Vasudevan
          <span class="socials">
            <a href="https://www.ornl.gov/staff-profile/rama-k-vasudevan" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <a href="https://github.com/ramav87" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">CNMS, Oak Ridge National Laboratory</div>
       
      </div>
    </div>

    
    <!-- Yongtao -->
    <div class="team-card">
      <img src="{{ '/assets/Yliu.png' | relative_url }}" alt="Yongtao">
      <div class="team-meta">
        <div class="name">
          Yongtao Liu
          <span class="socials">
            <a href="https://www.ornl.gov/staff-profile/yongtao-liu" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <a href="https://github.com/yongtaoliu" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">CNMS, Oak Ridge National Laboratory, TN</div>
       
      </div>
    </div>

        <!-- Steven -->
    <div class="team-card">
      <img src="{{ '/assets/ss.png' | relative_url }}" alt="Steven R Spurgeon">
      <div class="team-meta">
        <div class="name">
          Steven R Spurgeon
          <span class="socials">
            <a href="https://www.stevenspurgeon.com/group" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <a href="https://github.com/stevenspurgeon" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">National Renewable Energy Laboratory, CO; Colorado SChool of Mines</div>
       
      </div>
    </div>

    <!-- Richard -->
    <div class="team-card">
      <img src="{{ '/assets/rliu.png' | relative_url }}" alt="Richard Liu">
      <div class="team-meta">
        <div class="name">
          Richard Liu
          <span class="socials">
            <a href="https://github.com/RichardLiuCoding" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil"> University of Tennessee, Knoxville</div>
        
      </div>
    </div>

    <!-- Boris -->
    <div class="team-card">
      <img src="{{ '/assets/Boris.png' | relative_url }}" alt="Boris Slautin">
      <div class="team-meta">
        <div class="name">
          Boris Slautin
          <span class="socials">
            <a href="https://github.com/Slautin" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil"> University of Tennessee, Knoxville</div>
        
      </div>
    </div>

    <!-- Austin -->
    <div class="team-card">
      <img src="{{ '/assets/Austin.png' | relative_url }}" alt="Austin Houston">
      <div class="team-meta">
        <div class="name">
          Austin Houston
          <span class="socials">
            <a href="https://github.com/AustinHouston" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">University of Tennessee, Knoxville</div>
       
      </div>
    </div>
    
    <!-- Utkarsh -->
    <div class="team-card">
      <img src="{{ '/assets/up.png' | relative_url }}" alt="Utkarsh Pratiush">
      <div class="team-meta">
        <div class="name">
          Utkarsh Pratiush
          <span class="socials">
            <a href="https://utkarshp1161.github.io/UtkarshsAIInScience.github.io/" target="_blank" rel="noopener" title="Website" aria-label="Website">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 2c.9 0 2.4 2.1 3 6H9c.6-3.9 2.1-6 3-6ZM5.06 11a7.8 7.8 0 0 1 .16-1h3.01a20 20 0 0 0 0 2H5.22a7.8 7.8 0 0 1-.16-1Zm6.94 9c-.9 0-2.4-2.1-3-6h6c-.6 3.9-2.1 6-3 6Zm3.77-7a20 20 0 0 0 0-2h3.01c.21.98.21 2.02 0 3H15.77ZM7.2 7H4.9a8 8 0 0 1 3.4-2.6c-.43.82-.8 1.7-1.1 2.6Zm9.6 0c-.3-.9-.67-1.78-1.1-2.6A8 8 0 0 1 19.1 7h-2.3ZM7.2 17c.3.9.67 1.78 1.1 2.6A8 8 0 0 1 4.9 17h2.3Zm9.6 0h2.3a8 8 0 0 1-3.4 2.6c.43-.82.8-1.7 1.1-2.6Z"/></svg>
            </a>
            <a href="https://github.com/utkarshp1161" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">University of Tennessee, Knoxville</div>
       
      </div>
    </div>

     <!-- Aditya -->
    <div class="team-card">
      <img src="{{ '/assets/Aditya.jpg' | relative_url }}" alt="Aditya Raghavan">
      <div class="team-meta">
        <div class="name">
          Aditya Raghavan
          <span class="socials">
            <a href="https://github.com/adityaraghavan98" target="_blank" rel="noopener" title="GitHub" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.11.79-.25.79-.56v-2.02c-3.34.73-4.04-1.61-4.04-1.61-.53-1.35-1.29-1.71-1.29-1.71-1.06-.72.08-.71.08-.71 1.17.08 1.78 1.2 1.78 1.2 1.04 1.77 2.73 1.26 3.39.96.1-.76.41-1.26.74-1.55-2.66-.3-5.46-1.33-5.46-5.93 0-1.31.47-2.38 1.24-3.22-.13-.3-.54-1.52.12-3.17 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.66 1.65.25 2.87.12 3.17.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.47 5.92.42.36.79 1.07.79 2.16v3.2c0 .31.21.68.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </span>
        </div>
        <div class="affil">University of Tennessee, Knoxville</div>
       
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
    <img src="{{ '/assets/renaissance.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/TheiaScientific.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/DENS.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/Waviks.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/TRI.png' | relative_url }}"  style="max-height:64px;">
    <img src="{{ '/assets/msa_stc.png' | relative_url }}"  style="max-height:64px;">
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
