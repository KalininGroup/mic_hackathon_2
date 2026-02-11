---
layout: page
title: Hackathon Summary
permalink: /summary/
---

<style>
/* =========================
   Page basics
========================= */
.summary-hero {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  margin: 18px 0 18px 0;
}

.summary-hero img {
  width: 100%;
  height: 360px;
  object-fit: cover;
  display: block;
  filter: saturate(1.05);
}

.summary-hero .overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgba(0,0,0,0.35) 0%,
    rgba(0,0,0,0.55) 60%,
    rgba(0,0,0,0.70) 100%
  );
}

.summary-hero .text {
  position: absolute;
  left: 22px;
  right: 22px;
  bottom: 18px;
  color: #fff;
}

.summary-hero h1 {
  margin: 0 0 6px 0;
  font-size: 2.0rem;
  line-height: 1.15;
}

.summary-hero p {
  margin: 0;
  opacity: 0.95;
  max-width: 900px;
}

   .summary-hero-collage {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  margin: 18px 0;
}

.hero-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  height: 360px;
}

.hero-grid img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.hero-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    rgba(0,0,0,0.35) 0%,
    rgba(0,0,0,0.65) 100%
  );
}

.hero-text {
  position: absolute;
  bottom: 20px;
  left: 24px;
  right: 24px;
  color: white;
}

.hero-text h1 {
  margin: 0 0 6px 0;
  font-size: 2rem;
}

.hero-text p {
  margin: 0;
  opacity: 0.95;
}

   
/* =========================
   Metrics row
========================= */
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 14px;
  margin: 16px 0 22px 0;
}

.metric {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 12px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.70);
}

.metric .num {
  font-size: 1.35rem;
  font-weight: 700;
  line-height: 1.1;
}

.metric .lbl {
  margin-top: 4px;
  font-size: 0.95rem;
  opacity: 0.8;
}

/* =========================
   Section headings
========================= */
.section {
  margin: 26px 0;
}

.section h2 {
  margin: 0 0 10px 0;
  font-size: 1.35rem;
}

/* =========================
   Videos
========================= */
.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 18px;
  margin-top: 8px;
}

.video-card {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 12px;
  overflow: hidden;
  background: #fff;
}

.video-card .desc {
  padding: 12px 14px;
}

.video-card .desc .title {
  font-weight: 700;
  margin-bottom: 4px;
}

.video-card .desc .sub {
  opacity: 0.8;
  font-size: 0.95rem;
}

/* Responsive embed */
.ratio {
  position: relative;
  width: 100%;
  padding-top: 56.25%; /* 16:9 */
  background: #000;
}

.ratio iframe,
.ratio video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  border: 0;
}

/* =========================
   Photo gallery + lightbox
========================= */
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 16px;
  margin-top: 10px;
}

.gallery-item {
  text-align: center;
  font-size: 0.92rem;
}

.gallery-item img {
  width: 100%;
  height: 180px;
  object-fit: cover;
  border-radius: 10px;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.gallery-item img:hover {
  transform: scale(1.02);
  box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

.caption {
  margin-top: 7px;
  opacity: 0.85;
}

/* Lightbox overlay */
.lightbox {
  display: none;
  position: fixed;
  inset: 0;
  z-index: 9999;
  background-color: rgba(0,0,0,0.88);
  padding: 50px 18px 24px 18px;
}

.lightbox-content {
  max-width: 1100px;
  margin: 0 auto;
  text-align: center;
  color: #fff;
}

.lightbox img {
  max-width: 95%;
  max-height: 78vh;
  border-radius: 10px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.lightbox .lb-caption {
  margin-top: 12px;
  opacity: 0.92;
  font-size: 1.0rem;
}

.lightbox .close {
  position: fixed;
  top: 14px;
  right: 18px;
  color: #fff;
  font-size: 34px;
  line-height: 1;
  cursor: pointer;
  user-select: none;
}

.lightbox .hint {
  margin-top: 10px;
  opacity: 0.7;
  font-size: 0.9rem;
}

/* =========================
   Sponsor logo grid
========================= */
.sponsors {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 16px;
  align-items: center;
  margin-top: 10px;
}

.sponsor-logo {
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 12px;
  padding: 14px;
  background: #fff;
  height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sponsor-logo img {
  max-height: 60px;
  max-width: 100%;
  object-fit: contain;
  opacity: 0.9;
}

/* Small note text */
.note {
  opacity: 0.78;
  font-size: 0.95rem;
  margin-top: 8px;
}

/* Footer CTA */
.cta {
  margin: 28px 0 10px 0;
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(255,255,255,0.75);
}

.cta a {
  font-weight: 700;
  text-decoration: none;
}
</style>

<!-- =========================
     HERO (replace hero image path)
========================= -->
<div class="summary-hero-collage">
  <div class="hero-grid">
    <img src="{{ '/assets/event_photos/Online/Zoom1.png' | relative_url }}" alt="Zoom Kickoff 1">
    <img src="{{ '/assets/event_photos/Online/Zoom2.png' | relative_url }}" alt="Zoom Kickoff 2">
    <img src="{{ '/assets/event_photos/Online/Zoom3.png' | relative_url }}" alt="Zoom Kickoff 3">
  </div>

  <div class="hero-overlay"></div>

  <div class="hero-text">
    <h1>AI/ML for Microscopy Hackathon 2025</h1>
    <p>Global collaboration across sites, connected live through Zoom and local events.</p>
  </div>
</div>

<!-- =========================
     METRICS (edit values)
========================= -->
<div class="metrics">
  <div class="metric">
    <div class="num">250+</div>
    <div class="lbl">Participants</div>
  </div>
  <div class="metric">
    <div class="num">70+</div>
    <div class="lbl">Projects</div>
  </div>
  <div class="metric">
    <div class="num">12+</div>
    <div class="lbl">Global Sites</div>
  </div>
  <div class="metric">
    <div class="num">3</div>
    <div class="lbl">Days</div>
  </div>
</div>

<!-- =========================
     VIDEOS (replace embeds)
========================= -->
<div class="section">
  <h2>Videos</h2>
  <div class="video-grid">

    <div class="video-card">
      <div class="ratio">
        <!-- Replace with your embed iframe OR a <video> tag -->
        <iframe
          src="https://drive.google.com/file/d/1PgsxxLM7vup39rpNvNhC0-TeUbB_lBE7/preview"
          title="Hackathon Launch"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen>
        </iframe>
      </div>
      <div class="desc">
        <div class="title">Hackathon Launch</div>
        <div class="sub">Watch the recording from the live kickoff session — including the welcome, introduction, and overview of the hackathon.</div>
      </div>
    </div>

    <div class="video-card">
      <div class="ratio">
        <!-- Replace with your embed iframe OR a <video> tag -->
        <iframe
          src="[https://www.youtube.com/embed/VIDEO_ID_AWARDS](https://drive.google.com/file/d/1wNcB2CheQbFXenAAZffMaa12lge-8H8e/preview)"
          title="Awards Ceremony"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowfullscreen>
        </iframe>
      </div>
      <div class="desc">
        <div class="title">Awards Ceremony</div>
        <div class="sub">Revisiting the Hackathon, award announcements, and closing remarks.</div>
      </div>
    </div>

  </div>
</div>

<!-- =========================
     PHOTO GALLERY (add as many items as you want)
========================= -->
<div class="section">
  <h2>Photo Gallery</h2>

  <div class="gallery">
    <!-- Duplicate a block per photo -->

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/knoxville_1.jpg" alt="Knoxville, USA"
           onclick="openLightbox(this.src, 'Knoxville, USA')">
      <div class="caption">Knoxville, USA</div>
    </div>

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/delhi_1.jpg" alt="Delhi, India"
           onclick="openLightbox(this.src, 'Delhi, India')">
      <div class="caption">Delhi, India</div>
    </div>

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/barcelona_1.jpg" alt="Barcelona, Spain"
           onclick="openLightbox(this.src, 'Barcelona, Spain')">
      <div class="caption">Barcelona, Spain</div>
    </div>

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/qatar_1.jpg" alt="Qatar"
           onclick="openLightbox(this.src, 'Qatar')">
      <div class="caption">Qatar</div>
    </div>

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/canada_1.jpg" alt="Canada"
           onclick="openLightbox(this.src, 'Canada')">
      <div class="caption">Canada</div>
    </div>

    <div class="gallery-item">
      <img src="/assets/event_photos/summary/mena_1.jpg" alt="MENA"
           onclick="openLightbox(this.src, 'MENA')">
      <div class="caption">MENA</div>
    </div>

  </div>

  <div class="note">Tip: click any image to enlarge. Click outside the image (or press Esc) to close.</div>
</div>

<!-- =========================
     SPONSORS (logo collage only)
========================= -->
<div class="section">
  <h2>Supported By</h2>

  <div class="sponsors">

    <div class="sponsor-logo">
      <img src="{{ '/assets/ONR.png' | relative_url }}" alt="Office of Naval Research">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/renaissance.png' | relative_url }}" alt="Renaissance Philanthropy">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/Jeol.png' | relative_url }}" alt="JEOL">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/covalent.png' | relative_url }}" alt="Covalent">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/TRI.png' | relative_url }}" alt="Toyota Research Institute">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/DENS.png' | relative_url }}" alt="DENSsolutions">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/mat3ra_logo.png' | relative_url }}" alt="Mat3ra">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/msa_stc.png' | relative_url }}" alt="Microscopy Society of America Student Council">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/asylum.png' | relative_url }}" alt="Asylum Research">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/hf.png' | relative_url }}" alt="Hugging Face">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/waviks.png' | relative_url }}" alt="Waviks">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/commat.png' | relative_url }}" alt="COMMAT">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/TheiaScientific.png' | relative_url }}" alt="Theia Scientific">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/polaron.png' | relative_url }}" alt="Polaron">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/sponsors/elsevier.png' | relative_url }}" alt="Elsevier">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/hitachi.png' | relative_url }}" alt="Hitachi HiTech">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/AISCIA.png' | relative_url }}" alt="AISCIA">
    </div>

    <div class="sponsor-logo">
      <img src="{{ '/assets/ncsu.png' | relative_url }}" alt="NCSU College of Engineering">
    </div>

  </div>

  <div class="note">
    See the <a href="{{ '/awards/' | relative_url }}">Awards</a> page for award categories and sponsors.
  </div>
</div>
<!-- =========================
     CLOSING / LINKS
========================= -->
<div class="cta">
  Thank you to all participants, site organizers, judges, and sponsors who made this global event possible.
  <br><br>
  <a href="/projects/">Explore all projects →</a>
</div>

<!-- =========================
     LIGHTBOX (single instance)
========================= -->
<div id="lightbox" class="lightbox" onclick="closeLightbox()">
  <div class="close" onclick="closeLightbox()">&times;</div>
  <div class="lightbox-content" onclick="event.stopPropagation()">
    <img id="lightbox-img" alt="Expanded photo">
    <div class="lb-caption" id="lightbox-caption"></div>
    <div class="hint">Press Esc to close</div>
  </div>
</div>

<script>
function openLightbox(src, caption) {
  const lb = document.getElementById("lightbox");
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-caption").textContent = caption || "";
  lb.style.display = "block";
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lb = document.getElementById("lightbox");
  lb.style.display = "none";
  document.body.style.overflow = "";
}

// Close on Esc
document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") closeLightbox();
});
</script>
