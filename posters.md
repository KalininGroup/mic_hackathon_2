---
layout: default
title: Poster Gallery
permalink: /posters/
---

# Poster Gallery

A visual gallery of posters and social media art shared by local organizing teams.

As more sites share their designs, we’ll keep adding them here.

---

## Penn State University

<div class="poster-gallery">
  <figure class="poster-card">
    <img src="/posters/PSU_Poster.jfif" alt="Penn State University poster">
    <figcaption>Penn State University</figcaption>
  </figure>
</div>

---

## ICN2

<div class="poster-gallery">
  <figure class="poster-card">
    <img src="/posters/ICN2.jfif" alt="ICN2 poster">
    <figcaption>ICN2</figcaption>
  </figure>
</div>

---

## University of Toronto

<div class="poster-gallery">

  <figure class="poster-card">
    <img src="/posters/Toronto1.jfif" alt="University of Toronto poster 1">
    <figcaption>University of Toronto — Poster 1</figcaption>
  </figure>

  <figure class="poster-card">
    <img src="/posters/Toronto2.jfif" alt="University of Toronto poster 2">
    <figcaption>University of Toronto — Poster 2</figcaption>
  </figure>

  <figure class="poster-card">
    <img src="/posters/Toronto3.jfif" alt="University of Toronto poster 3">
    <figcaption>University of Toronto — Poster 3</figcaption>
  </figure>

</div>

---

<p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
  Future idea: enable community voting for favorite posters across all sites.
</p>

<style>
.poster-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 20px;
  margin-top: 1.2rem;
}

.poster-card {
  background: #fafafa;
  border-radius: 12px;
  padding: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  text-align: center;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.poster-card img {
  width: 100%;
  max-height: 340px;
  object-fit: contain;
  border-radius: 8px;
}

.poster-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(0,0,0,0.10);
}

.poster-card figcaption {
  margin-top: 0.6rem;
  font-size: 0.9rem;
  color: #555;
}
</style>
