---
layout: page
title: Submissions
menu_title: Submissions
menu_icon: cloud-upload
permalink: /submissions/
published: true
nav_exclude: false
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
  margin-top:0;
  margin-bottom:10px;
  font-weight:600;
  border-left:4px solid #b4c8ff;
  padding-left:10px;
}
.section-card p{
  margin:8px 0 0;
  color:#344054;
  line-height:1.55;
}
.resource-btn {
  display:inline-block;
  margin-top:12px;
  background:#4a63e7;
  color:white !important;
  padding:8px 16px;
  border-radius:8px;
  text-decoration:none;
  font-weight:600;
}
</style>

## Hackathon Submissions

<table>
  <thead>
    <tr>
      <th>Project</th>
      <th>Description</th>
      <th>Links</th>
      <th>Team</th>
    </tr>
  </thead>
  <tbody>
    {% for p in site.data.projects %}
    <tr>
      <td><strong>{{ p.title }}</strong></td>

      <td>{{ p.description }}</td>

      <td>
        {% if p.repo and p.repo != "N/A" %}
          <a href="{{ p.repo }}" target="_blank">Code</a><br>
        {% endif %}
        {% if p.video and p.video != "N/A" %}
          <a href="{{ p.video }}" target="_blank">Video</a><br>
        {% endif %}
        {% if p.slides and p.slides != "N/A" %}
          <a href="{{ p.slides }}" target="_blank">Slides</a>
        {% endif %}
      </td>

      <td>
        {% for i in (1..8) %}
          {% assign key = "member" | append: i %}
          {% if p[key] and p[key] != "N/A" %}
            {{ p[key] }}<br>
          {% endif %}
        {% endfor %}
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>




Use the resources below to prepare your final hackathon submission.  
Each section includes links and instructions to help you navigate the process.

---

<div class="section-card">
  <h2>📦 Access the Hackathon Dataset</h2>
  <p>
    Find all microscopy datasets used in the Digital Twin Microscope and the Hackathon.
    This includes raw STEM data, metadata, and other related files.
  </p>
  <a class="resource-btn"
     href="https://github.com/pycroscopy/DTMicroscope/tree/main/data"
     target="_blank">Open Dataset</a>
</div>

<div class="section-card">
  <h2>🧪 Digital Twin Microscope – Demo Notebooks</h2>
  <p>
    Explore the Digital Twin Microscope through interactive notebooks.
    These show how to simulate scans, load data, and work with the digital twin environment.
  </p>
  <a class="resource-btn"
     href="https://github.com/pycroscopy/DTMicroscope/tree/main/notebooks"
     target="_blank">Open Demo Notebooks</a>
</div>

<div class="section-card">
  <h2>📁 Preparing Your Data For Submission</h2>
  <p>
    Use this notebook to properly format your datasets and prepare files for submission.
    It demonstrates how to create clean, well-structured datasets from raw microscopy data.
  </p>
  <a class="resource-btn"
     href="https://github.com/pycroscopy/DTMicroscope/blob/main/notebooks/STEM/0_create_dataset_for_DTmicroscope.ipynb"
     target="_blank">Open Preparation Notebook</a>
</div>

---

<div class="section-card">
  <h2>📬 Need Help?</h2>
  <p>
    For more details or assistance with the hackathon datasets, please contact:
  </p>
  <p><strong>Rama Vasudevan</strong>: <a href="mailto:vasudevanrk@ornl.gov">vasudevanrk@ornl.gov</a></p>
  <p><strong>Utkarsh Pratiush</strong>: <a href="mailto:upratius@vols.utk.edu">upratius@vols.utk.edu</a></p>
</div>
