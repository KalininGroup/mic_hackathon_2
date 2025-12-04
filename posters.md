---
layout: default
title: Poster Gallery
permalink: /posters/
---

# Poster Gallery

A visual gallery of posters and social media art shared by local organizing teams.

After viewing the posters below, you can cast your vote for the **Popular Opinion Prize**.

<p id="vote-status" class="vote-status"></p>
<p class="vote-note">
  Click <strong>“Vote for this poster”</strong> under your favorite design.
  A Google Form will open in a new tab where you confirm your choice.
  One vote per browser (soft lockout).
</p>

---

## Penn State University

<div class="poster-gallery">
  <figure class="poster-card" data-poster-id="psu" data-poster-label="Penn State University Poster">
    <img src="{{ '/posters/PSU_Poster.jfif' | relative_url }}" alt="Penn State University poster">
    <figcaption>Penn State University</figcaption>
    <button class="vote-btn" type="button">
      Vote for this poster
    </button>
  </figure>
</div>

---

## ICN2

<div class="poster-gallery">
  <figure class="poster-card" data-poster-id="icn2" data-poster-label="ICN2 Poster">
    <img src="{{ '/posters/ICN2.jfif' | relative_url }}" alt="ICN2 poster">
    <figcaption>ICN2</figcaption>
    <button class="vote-btn" type="button">
      Vote for this poster
    </button>
  </figure>
</div>

---

## University of Toronto

<div class="poster-gallery">

  <figure class="poster-card" data-poster-id="toronto1" data-poster-label="University of Toronto – Poster 1">
    <img src="{{ '/posters/Toronto1.jfif' | relative_url }}" alt="University of Toronto poster 1">
    <figcaption>University of Toronto — Poster 1</figcaption>
    <button class="vote-btn" type="button">
      Vote for this poster
    </button>
  </figure>

  <figure class="poster-card" data-poster-id="toronto2" data-poster-label="University of Toronto – Poster 2">
    <img src="{{ '/posters/Toronto2.jfif' | relative_url }}" alt="University of Toronto poster 2">
    <figcaption>University of Toronto — Poster 2</figcaption>
    <button class="vote-btn" type="button">
      Vote for this poster
    </button>
  </figure>

  <figure class="poster-card" data-poster-id="toronto3" data-poster-label="University of Toronto – Poster 3">
    <img src="{{ '/posters/Toronto3.jfif' | relative_url }}" alt="University of Toronto poster 3">
    <figcaption>University of Toronto — Poster 3</figcaption>
    <button class="vote-btn" type="button">
      Vote for this poster
    </button>
  </figure>

</div>

---

<p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
  Note: This is a community Popular Opinion vote. Organizers may check for obvious duplicate patterns before announcing the winner.
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

/* Vote UI */
.vote-note {
  font-size: 0.9rem;
  margin-bottom: 0.8rem;
  opacity: 0.9;
}

.vote-status {
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 0.3rem;
}

.vote-btn {
  margin-top: 0.6rem;
  padding: 5px 12px;
  border-radius: 999px;
  border: 1px solid #ddd;
  background: #ffffff;
  cursor: pointer;
  font-size: 0.85rem;
}

.vote-btn:hover:not(:disabled) {
  background: #f0f0f0;
}

.vote-btn:disabled {
  cursor: default;
  opacity: 0.6;
}
</style>

<script>
// 1) Put your Google Form "view form" link here (NOT the embed iframe)
// Example: "https://docs.google.com/forms/d/e/.../viewform"
const FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/viewform?usp=dialog";

document.addEventListener('DOMContentLoaded', function () {
  const statusEl = document.getElementById('vote-status');

  function hasVoted() {
    return localStorage.getItem('poster_vote_done') === '1';
  }

  function getVotedLabel() {
    return localStorage.getItem('poster_vote_label') || null;
  }

  function updateUI() {
    const voted = hasVoted();
    const votedLabel = getVotedLabel();

    const buttons = document.querySelectorAll('.vote-btn');
    buttons.forEach(btn => {
      if (voted) {
        btn.disabled = true;
        btn.textContent = 'Vote submitted';
      }
    });

    if (statusEl) {
      if (voted && votedLabel) {
        statusEl.textContent = 'You have already cast your vote for: ' + votedLabel + '.';
      } else if (voted) {
        statusEl.textContent = 'You have already cast your vote.';
      } else {
        statusEl.textContent = '';
      }
    }
  }

  updateUI();

  document.querySelectorAll('.vote-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      if (hasVoted()) {
        alert('Our records show you already voted from this browser. Thank you!');
        return;
      }

      const card = btn.closest('.poster-card');
      if (!card) return;

      const label = card.getAttribute('data-poster-label') || 'this poster';

      const ok = confirm(
        'Vote for "' + label + '" as your Popular Opinion choice?\n\n' +
        'This is recorded once per browser. After clicking OK, a Google Form will open where you confirm this choice.'
      );
      if (!ok) return;

      // Soft lockout: store as "voted" in this browser
      localStorage.setItem('poster_vote_done', '1');
      localStorage.setItem('poster_vote_label', label);

      // Open the Google Form where user selects the same option
      if (FORM_URL && FORM_URL !== "PASTE_YOUR_GOOGLE_FORM_URL_HERE") {
        window.open(FORM_URL, '_blank');
      } else {
        alert('FORM_URL is not set yet. Please contact the organizers.');
      }

      updateUI();
    });
  });
});
</script>
