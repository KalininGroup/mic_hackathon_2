---
layout: page
title: Live Event Updates
permalink: /event/
---
<style>
/* Thumbnail grid */
.photo-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:14px;
  margin:16px 0 28px 0;
}

.photo-grid img{
  width:100%;
  height:150px;
  object-fit:cover;
  border-radius:10px;
  box-shadow:0 4px 10px rgba(0,0,0,0.12);
}

/* Make any image with this class clickable */
.lightboxable{ cursor:pointer; }

/* Lightbox modal */
.lb{
  position:fixed;
  inset:0;
  z-index:9999;
  display:none;
}
.lb.is-open{ display:block; }

.lb__backdrop{
  position:absolute;
  inset:0;
  background:rgba(0,0,0,0.70);
}

.lb__dialog{
  position:relative;
  max-width:min(950px,92vw);
  max-height:86vh;
  margin:6vh auto 0 auto;
  background:#fff;
  border-radius:14px;
  overflow:hidden;
  box-shadow:0 16px 50px rgba(0,0,0,0.35);
}

.lb__close{
  position:absolute;
  top:8px;
  right:12px;
  width:40px;
  height:40px;
  border:0;
  border-radius:20px;
  font-size:28px;
  line-height:40px;
  cursor:pointer;
  background:rgba(255,255,255,0.92);
}

.lb__img{
  display:block;
  width:100%;
  height:auto;
  max-height:78vh;
  object-fit:contain;
  background:#111;
}

.lb__cap{
  padding:10px 14px;
  font-size:14px;
  color:#444;
}
</style>

## Hackathon Participation Overview

Registration for the 2025 AI/ML Microscopy Hackathon is now closed.

We received **689 registrations** in total, with participants joining from universities, national laboratories, and industry worldwide.

- **Online participation:** 468 registered participants  
- **Local site participation:** 221 participants across multiple host institutions  

This page shares live snapshots from the ongoing hackathon, including online sessions and local site activities, and will continue to be updated throughout the event.

## Zoom Kick-off Sessions
<div class="photo-grid">
  <img class="lightboxable"
       src="{{ '/assets/event_photos/Online/Zoom1.png' | relative_url }}"
       alt="Hackathon Zoom session 1">

  <img class="lightboxable"
       src="{{ '/assets/event_photos/Online/Zoom2.png' | relative_url }}"
       alt="Hackathon Zoom session 2">

  <img class="lightboxable"
       src="{{ '/assets/event_photos/Online/Zoom3.png' | relative_url }}"
       alt="Hackathon Zoom session 3">
</div>

<!-- Global Lightbox Modal (one-time) -->
<div id="lightbox" class="lb" aria-hidden="true">
  <div class="lb__backdrop" data-lb-close></div>

  <div class="lb__dialog" role="dialog" aria-modal="true" aria-label="Image preview">
    <button class="lb__close" type="button" aria-label="Close" data-lb-close>&times;</button>
    <img id="lightbox-img" class="lb__img" alt="">
    <div id="lightbox-cap" class="lb__cap"></div>
  </div>
</div>

<script>
(function () {
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lightbox-img');
  const lbCap = document.getElementById('lightbox-cap');

  function openLightbox(src, alt) {
    lbImg.src = src;
    lbImg.alt = alt || '';
    lbCap.textContent = alt || '';
    lb.classList.add('is-open');
    lb.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    lb.classList.remove('is-open');
    lb.setAttribute('aria-hidden', 'true');
    lbImg.src = '';
    document.body.style.overflow = '';
  }

  // Works for all current and future images with class="lightboxable"
  document.addEventListener('click', (e) => {
    const img = e.target.closest('img.lightboxable');
    if (!img) return;
    const src = img.getAttribute('data-full') || img.currentSrc || img.src;
    openLightbox(src, img.alt);
  });

  // Close on backdrop or close button
  lb.querySelectorAll('[data-lb-close]').forEach(el => {
    el.addEventListener('click', closeLightbox);
  });

  // Close on ESC
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lb.classList.contains('is-open')) closeLightbox();
  });
})();
</script>

