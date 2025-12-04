---
layout: default
title: Poster Gallery
permalink: /posters/
---

# Poster Gallery

A visual gallery of posters and social media art shared by local organizing teams.

After viewing the posters below, you can cast your vote for the **Popular Opinion Prize**.  
Each button opens a Google Form with that specific poster already selected.

---

## Penn State University

<div class="poster-gallery">
  <figure class="poster-card">
    <img src="{{ '/posters/PSU_Poster.jfif' | relative_url }}" alt="Penn State University poster">
    <figcaption>Penn State University</figcaption>
    <a
      class="vote-btn"
      target="_blank"
      href="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=pp_url&entry.2036557565=Penn+State+University+Poster"
    >
      Vote for this poster
    </a>
  </figure>
</div>

---

## ICN2

<div class="poster-gallery">
  <figure class="poster-card">
    <img src="{{ '/posters/ICN2.jfif' | relative_url }}" alt="ICN2 poster">
    <figcaption>ICN2</figcaption>
    <a
      class="vote-btn"
      target="_blank"
      href="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=pp_url&entry.2036557565=ICN2+Poster"
    >
      Vote for this poster
    </a>
  </figure>
</div>

---

## University of Toronto

<div class="poster-gallery">

  <figure class="poster-card">
    <img src="{{ '/posters/Toronto1.jfif' | relative_url }}" alt="University of Toronto poster 1">
    <figcaption>University of Toronto — Poster 1</figcaption>
    <a
      class="vote-btn"
      target="_blank"
      href="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=pp_url&entry.2036557565=University+of+Toronto+%E2%80%93+Poster+1"
    >
      Vote for this poster
    </a>
  </figure>

  <figure class="poster-card">
    <img src="{{ '/posters/Toronto2.jfif' | relative_url }}" alt="University of Toronto poster 2">
    <figcaption>University of Toronto — Poster 2</figcaption>
    <a
      class="vote-btn"
      target="_blank"
      href="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=pp_url&entry.2036557565=University+of+Toronto+%E2%80%93+Poster+2"
    >
      Vote for this poster
    </a>
  </figure>

  <figure class="poster-card">
    <img src="{{ '/posters/Toronto3.jfif' | relative_url }}" alt="University of Toronto poster 3">
    <figcaption>University of Toronto — Poster 3</figcaption>
    <a
      class="vote-btn"
      target="_blank"
      href="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=pp_url&entry.2036557565=University+of+Toronto+%E2%80%93+Poster+3"
    >
      Vote for this poster
    </a>
  </figure>

</div>

---

<p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
  This is a community Popular Opinion vote. Organizers may review responses for obvious duplicate patterns before announcing the winner.
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

/* Vote button styling */
.vote-btn {
  display: inline-block;
  margin-top: 0.6rem;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid #ddd;
  background: #ffffff;
  cursor: pointer;
  font-size: 0.85rem;
  text-decoration: none;
  color: inherit;
}

.vote-btn:hover {
  background: #f0f0f0;
}
</style>
