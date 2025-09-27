---
layout: page
title: Sites
menu_title: Sites
menu_icon: geo-alt
permalink: /sites/
published: true
---
<style>
  #worldmap { height: 520px; border-radius: 14px; border:1px solid #e6e6e6; margin: 1rem 0 1.5rem; }
  .leaflet-popup-content { margin: 8px 10px; }
  .leaflet-popup-content h4 { margin: 0 0 .25rem; font-size: 1rem; }
  .leaflet-popup-content p { margin: 0; font-size: .92rem; color:#444; }
</style>

<div id="worldmap"></div>

<!-- Leaflet (no key needed) -->
<link
  rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
  crossorigin=""
/>
<script
  src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
  crossorigin=""
></script>

{% raw %}
<script>
  // Custom UTK orange marker (#FF8200)
  const utkIcon = new L.Icon({
    iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
  });

  // --- Sites with coordinates ---
  const sites = [
    {
      name: "University of Tennessee, Knoxville (UTK)",
      city: "Knoxville, TN, USA",
      lat: 35.954, lon: -83.929,
      icon: utkIcon
    },
    {
      name: "North Carolina State University (NCSU)",
      city: "Raleigh, NC, USA",
      lat: 35.7847, lon: -78.6821
    },
    {
      name: "Northwestern University",
      city: "Evanston, IL, USA",
      lat: 42.05598, lon: -87.67517
    },
    {
      name: "University of Illinois at Chicago (UIC)",
      city: "Chicago, IL, USA",
      lat: 41.8708, lon: -87.6505
    },
    {
      name: "ICN2 — Institut Català de Nanociència i Nanotecnologia",
      city: "Barcelona (Bellaterra), Spain",
      lat: 41.501, lon: 2.105
    },
    {
      name: "University of Toronto",
      city: "Toronto, ON, Canada",
      lat: 43.6629, lon: -79.3957
    },
    {
      name: "University of Wisconsin",
      city: "Madison, WI, USA",
      lat: 43.0766, lon: -89.4125
    },
    {
      name: "University of Colorado Boulder",
      city: "Boulder, CO, USA",
      lat: 40.0076, lon: -105.2659
    },
    {
      name: "Colorado School of Mines",
      city: "Golden, CO, USA",
      lat: 39.7510, lon: -105.2226
    }
    // Online = global, no pin
  ];

  // --- Build map ---
  const map = L.map('worldmap', { scrollWheelZoom: false });
  const osm = L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '&copy; OpenStreetMap contributors' }
  ).addTo(map);

  // Add markers
  const markers = [];
  sites.forEach(s => {
    const opts = s.icon ? { icon: s.icon } : {};
    const m = L.marker([s.lat, s.lon], opts).addTo(map);
    m.bindPopup(`
      <h4>${s.name}</h4>
      <p>${s.city}</p>
    `);
    markers.push(m);
  });

  // Fit to bounds
  if (markers.length) {
    const group = L.featureGroup(markers);
    map.fitBounds(group.getBounds().pad(0.2));
  } else {
    map.setView([20, 0], 2);
  }

  // Resize fix for mobile/tab switches
  window.addEventListener('resize', () => map.invalidateSize());
</script>
{% endraw %}



<p class="hint">Pick the site that’s closest to you (or choose <strong>Online</strong>). Final room details and building access instructions will be emailed to registered participants.</p>

<style>
/* Sites page styles (scoped) */
.sites-wrap, .sites-wrap * { box-sizing: border-box; }
.sites-grid{
  display:grid; gap:1rem;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin: 1rem 0 2rem;
}
.site-card{
  background:#fff; border:1px solid #e6e6e6; border-radius:14px;
  padding:1rem 1rem 1.1rem; box-shadow:0 6px 18px rgba(0,0,0,.05);
}
.site-card h3{ margin:.2rem 0 .4rem; font-size:1.1rem; }
.site-meta{ color:#555; font-size:.95rem; margin:.3rem 0 .6rem; }
.site-actions a{
  display:inline-block; padding:.5rem .75rem; border-radius:10px; margin-right:.4rem; margin-top:.3rem;
  text-decoration:none; font-weight:600; border:1px solid #d6d6d6; background:#fafafa;
}
.site-actions a:hover{ border-color:#3a7bd5; box-shadow:0 4px 12px rgba(58,123,213,.18); }
.badge{ display:inline-block; font-size:.78rem; padding:.18rem .5rem; border-radius:999px; background:#eef2ff; color:#334155; border:1px solid #c7d2fe; }
.btn-primary{
  display:inline-block; padding:.6rem 1rem; border-radius:10px;
  border:1px solid #2e6bd6; background:#3a7bd5; color:#fff; font-weight:700;
  text-decoration:none;
}
.hint{ font-size:.95rem; color:#555; }
</style>

<div class="sites-wrap">
  <div class="sites-grid">

    <!-- UTK -->
    <div class="site-card">
      <span class="badge">Tennessee, USA</span>
      <h3>University of Tennessee, Knoxville (UTK)</h3>
      <div class="site-meta">
        Knoxville, TN • Building/Room: <em>TBD</em> <br>
        Local lead: <a href="mailto:sergei2@utk.edu">sergei2@utk.edu</a>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Tennessee%20Knoxville" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- NCSU -->
    <div class="site-card">
      <span class="badge">North Carolina, USA</span>
      <h3>North Carolina State University (NCSU)</h3>
      <div class="site-meta">
        Raleigh, NC • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=North%20Carolina%20State%20University" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Northwestern -->
    <div class="site-card">
      <span class="badge">Illinois, USA</span>
      <h3>Northwestern University</h3>
      <div class="site-meta">
        Evanston, IL • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Northwestern%20University" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- UIC -->
    <div class="site-card">
      <span class="badge">Illinois, USA</span>
      <h3>University of Illinois at Chicago (UIC)</h3>
      <div class="site-meta">
        Chicago, IL • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Illinois%20at%20Chicago" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- ICN2 -->
    <div class="site-card">
      <span class="badge">Barcelona, Spain</span>
      <h3>ICN2 — Institut Català de Nanociència i Nanotecnologia</h3>
      <div class="site-meta">
        Barcelona • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=ICN2%20Barcelona" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Toronto -->
    <div class="site-card">
      <span class="badge">Ontario, Canada</span>
      <h3>University of Toronto</h3>
      <div class="site-meta">
        Toronto, ON • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Toronto" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Wisconsin -->
    <div class="site-card">
      <span class="badge">Wisconsin, USA</span>
      <h3>University of Wisconsin</h3>
      <div class="site-meta">
        Madison, WI • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Wisconsin%20Madison" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Colorado -->
    <div class="site-card">
      <span class="badge">Colorado, USA</span>
      <h3>University of Colorado</h3>
      <div class="site-meta">
        Boulder, CO • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Colorado%20Boulder" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Colorado School of Mines -->
    <div class="site-card">
      <span class="badge">Colorado, USA</span>
      <h3>Colorado School of Mines</h3>
      <div class="site-meta">
        Golden, CO • Building/Room: <em>TBD</em> <br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Colorado%20School%20of%20Mines%20Golden" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Online -->
    <div class="site-card">
      <span class="badge">Global</span>
      <h3>Online</h3>
      <div class="site-meta">
        Participate remotely via Zoom + Slack <br>
        Access details will be emailed after registration.
      </div>
      <div class="site-actions">
        <a href="{{ '/registration/' | relative_url }}" class="btn-primary">Register</a>
      </div>
    </div>

  </div>
</div>


## Map of sites


