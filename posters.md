---
layout: default
title: Poster Gallery
permalink: /posters/
---

# Poster Gallery

A visual gallery of posters and social media art shared by local organizing teams.

After viewing the posters, **click “Select this poster”** under your favorite design.  
Then scroll to the bottom and click **“Submit Vote”** to record your choice.

<p id="vote-status" class="vote-status"></p>

---

## Pennsylvania State University, Pennsylvania, USA

<div class="poster-gallery">
  <figure class="poster-card"
          data-vote-value="Penn State University Poster">
    <img src="{{ '/posters/PSU_Poster.jfif' | relative_url }}" alt="Penn State University poster">
    <figcaption>Penn State University</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>
</div>

---

## ICN2 — Institut Català de Nanociència i Nanotecnologia, Barcelona, Spain

<div class="poster-gallery">
  <figure class="poster-card"
          data-vote-value="ICN2 Poster">
    <img src="{{ '/posters/ICN2.jfif' | relative_url }}" alt="ICN2 poster">
    <figcaption>ICN2</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>
</div>

---

## University of Toronto, Ontario, Canada

<div class="poster-gallery">

  <figure class="poster-card"
          data-vote-value="University of Toronto – Poster 1">
    <img src="{{ '/posters/Toronto1.jfif' | relative_url }}" alt="University of Toronto poster 1">
    <figcaption>University of Toronto — Poster 1</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>

  <figure class="poster-card"
          data-vote-value="University of Toronto – Poster 2">
    <img src="{{ '/posters/Toronto2.jfif' | relative_url }}" alt="University of Toronto poster 2">
    <figcaption>University of Toronto — Poster 2</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>

  <figure class="poster-card"
          data-vote-value="University of Toronto – Poster 3">
    <img src="{{ '/posters/Toronto3.jfif' | relative_url }}" alt="University of Toronto poster 3">
    <figcaption>University of Toronto — Poster 3</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>

  <figure class="poster-card"
          data-vote-value="University of Toronto – Poster 4">
    <img src="{{ '/posters/Toronto4.png' | relative_url }}" alt="University of Toronto poster 4">
    <figcaption>University of Toronto — Poster 4</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>

</div>

---

## AISCIA, Qatar

<div class="poster-gallery">

  <figure class="poster-card"
          data-vote-value="AISCIA, Qatar">
    <img src="{{ '/posters/AISCIA.jfif' | relative_url }}" alt="AISCIA Qatar poster">
    <figcaption>AISCIA, Qatar</figcaption>
    <button type="button" class="select-btn">
      Select this poster
    </button>
  </figure>

</div>


---

## Submit Your Popular Opinion Vote

## Submit Your Popular Opinion Vote

<div id="vote-controls">
  <p class="vote-note">
    Selected poster: <strong><span id="selected-label">None</span></strong>
  </p>

  <div id="voter-info-box">
    <label>Your Name:<br>
      <input type="text" id="voter-name" class="voter-input">
    </label>
    <br><br>
    <label>Your Institution:<br>
      <input type="text" id="voter-inst" class="voter-input">
    </label>
  </div>

  <button id="submit-vote-btn" type="button" class="submit-vote-btn">
    Submit Vote
  </button>
</div>


<p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
  This is a community Popular Opinion vote. Organizers may review responses for obvious duplicate patterns before announcing the winner.
</p>

<!-- Hidden form + iframe that actually sends the vote to Google Forms -->
<iframe name="hidden_vote_iframe" style="display:none;"></iframe>

<form id="vote-form"
      action="https://docs.google.com/forms/d/e/1FAIpQLSd-0mXsyFZr7D9ov-CfWhWEzIYT-DBEahj3FlHk1wN3wJcvyA/formResponse"
      method="POST"
      target="hidden_vote_iframe"
      style="display:none;">
  <!-- This entry ID is from your form (entry.2036557565) -->
  <input type="hidden" name="entry.2036557565" id="vote-entry">
</form>

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
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  border: 2px solid transparent;
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

.poster-card.selected {
  border-color: #0077cc;
  box-shadow: 0 0 0 2px rgba(0,119,204,0.2);
}

.poster-card figcaption {
  margin-top: 0.6rem;
  font-size: 0.9rem;
  color: #555;
}

/* Buttons */
.select-btn,
.submit-vote-btn {
  margin-top: 0.6rem;
  padding: 6px 14px;
  border-radius: 999px;
  border: 1px solid #ddd;
  background: #ffffff;
  cursor: pointer;
  font-size: 0.85rem;
}

.select-btn:hover,
.submit-vote-btn:hover {
  background: #f0f0f0;
}

.submit-vote-btn:disabled {
  cursor: default;
  opacity: 0.6;
}

.vote-status {
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.vote-note {
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

#vote-controls {
  margin-top: 1.2rem;
  padding: 10px;
  border-radius: 10px;
  background: #f7f9fb;
  border: 1px solid #e1e5ea;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function () {
  const cards = document.querySelectorAll('.poster-card');
  const selectButtons = document.querySelectorAll('.select-btn');
  const selectedLabelEl = document.getElementById('selected-label');
  const statusEl = document.getElementById('vote-status');
  const submitBtn = document.getElementById('submit-vote-btn');
  const form = document.getElementById('vote-form');
  const entryInput = document.getElementById('vote-entry');

  const STORAGE_KEY_DONE = 'poster_vote_done';
  const STORAGE_KEY_LABEL = 'poster_vote_label';

  function hasVoted() {
    return localStorage.getItem(STORAGE_KEY_DONE) === '1';
  }

  function getVotedLabel() {
    return localStorage.getItem(STORAGE_KEY_LABEL) || null;
  }

  let currentSelection = null;  // vote-value string

  function updateStatusUI() {
    if (hasVoted()) {
      const lbl = getVotedLabel();
      statusEl.textContent = lbl
        ? 'You have already submitted your vote for: ' + lbl + '.'
        : 'You have already submitted your vote.';
      submitBtn.disabled = true;
      selectButtons.forEach(btn => btn.disabled = true);
    } else {
      statusEl.textContent = '';
      submitBtn.disabled = false;
      selectButtons.forEach(btn => btn.disabled = false);
    }
  }

  function clearSelectionHighlight() {
    cards.forEach(card => card.classList.remove('selected'));
  }

  // Initialize selection UI
  selectButtons.forEach(btn => {
    btn.addEventListener('click', function () {
      if (hasVoted()) {
        alert('Our records show you already voted from this browser. Thank you!');
        return;
      }
      const card = btn.closest('.poster-card');
      if (!card) return;

      const voteValue = card.getAttribute('data-vote-value');
      currentSelection = voteValue;

      clearSelectionHighlight();
      card.classList.add('selected');
      selectedLabelEl.textContent = voteValue;
    });
  });

  submitBtn.addEventListener('click', function () {
    if (hasVoted()) {
      alert('Our records show you already voted from this browser. Thank you!');
      return;
    }
    if (!currentSelection) {
      alert('Please select a poster before submitting your vote.');
      return;
    }

    // Confirm with the user
    const ok = confirm(
      'Submit your Popular Opinion vote for:\n\n' +
      currentSelection + '\n\n' +
      'Press OK to submit your vote.'
    );
    if (!ok) return;

    // Fill hidden form and submit in background
    entryInput.value = currentSelection;
    form.submit();

    // Soft lockout in this browser
    localStorage.setItem(STORAGE_KEY_DONE, '1');
    localStorage.setItem(STORAGE_KEY_LABEL, currentSelection);

    alert('Thank you! Your vote has been submitted.');
    updateStatusUI();
  });

  // On load, reflect prior vote, if any
  const prev = getVotedLabel();
  if (prev) {
    selectedLabelEl.textContent = prev;
  }
  updateStatusUI();
});
</script>
