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
  /* Card logos + highlight */
  .site-card .site-logo{ display:block; max-height:44px; margin:.2rem 0 .6rem; object-fit:contain; }
  .site-card.highlight{ animation: sitePulse 1.8s ease 1; box-shadow: 0 10px 24px rgba(58,123,213,.25); }
  @keyframes sitePulse{
    0%{box-shadow:0 0 0 rgba(58,123,213,0)}
    30%{box-shadow:0 10px 28px rgba(58,123,213,.35)}
    100%{box-shadow:0 10px 24px rgba(58,123,213,.10)}
  }
</style>

<style>
.site-logo-row img {
  background:#fff;
  border-radius:10px;
  padding:2px;
  border:1px solid #e6e6e6;
}
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
  // UTK orange pin
  const utkIcon = new L.Icon({
    iconUrl: "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25,41], iconAnchor: [12,41], popupAnchor: [1,-34], shadowSize: [41,41]
  });

  // Site list for the map (ids MUST match the card ids below)
  const sites = [
    { id:"site-utk",   name:"University of Tennessee, Knoxville (UTK)", city:"Knoxville, TN, USA", lat:35.954,   lon:-83.929,  icon:utkIcon },
    { id:"site-ncsu",  name:"North Carolina State University (NCSU)",   city:"Raleigh, NC, USA",   lat:35.7847,  lon:-78.6821 },
    { id:"site-nwu",   name:"Northwestern University",                  city:"Evanston, IL, USA",  lat:42.05598, lon:-87.67517 },
    { id:"site-uic",   name:"University of Illinois at Chicago (UIC)",  city:"Chicago, IL, USA",   lat:41.8708,  lon:-87.6505 },
    { id:"site-icn2",  name:"ICN2 + ALBA Synchrotron", city:" Cerdanyola del Vallès, Spain", lat:41.4915, lon:2.1834 },
    { id:"site-iitd", name:"Indian Institute of Technology Delhi", city:"New Delhi, India", lat:28.5450, lon:77.1926 },
    { id:"site-utor",  name:"University of Toronto",                    city:"Toronto, ON, Canada",lat:43.6629,  lon:-79.3957 },
    { id:"site-uwisc", name:"University of Wisconsin",                  city:"Madison, WI, USA",   lat:43.0766,  lon:-89.4125 },
    { id:"site-ucolo", name:"University of Colorado",                   city:"Boulder, CO, USA",   lat:40.0076,  lon:-105.2659 },
    { id:"site-mines", name:"Colorado School of Mines",                 city:"Golden, CO, USA",    lat:39.7510,  lon:-105.2226 },
    { id:"site-tfs", name:"Thermo Fisher Scientific – Eindhoven", city:"Eindhoven, Netherlands", lat:51.469218, lon:5.421370 },
    { id:"site-aiscia", name:"AISCIA Informatics – Doha, Qatar", city:"Qatar Science and Technology Park, Doha", lat:25.325, lon:51.435 },
    { id:"site-psu",  name:"Pennsylvania State University", city:"State College, PA, USA", lat:40.7982, lon:-77.8599 },
    { id:"site-upenn", name:"University of Pennsylvania", city:"Philadelphia, PA, USA", lat:39.9522, lon:-75.1932 },
    { id:"site-umich", name:"University of Michigan, Ann Arbor", city:"Ann Arbor, MI, USA", lat:42.2780, lon:-83.7382 },
    { id:"site-skku",name:"Sungkyunkwan University (SKKU)", city:"Suwon, Republic of Korea", lat:37.2930, lon:126.9769 },
    { id:"site-ntu", name:"Nanyang Technological University (NTU), Singapore", city:"Singapore", lat:1.3483, lon:103.6831 },
    { id:"site-tamu", name:"Texas A&M University", city:"College Station, TX, USA", lat:30.6150, lon:-96.3410},
    { id:"site-jhu", name:"Johns Hopkins University", city:"Baltimore, MD, USA", lat:39.3299, lon:-76.6205},
    { id:"site-cambridge", name:"University of Cambridge", city:"Cambridge, United Kingdom", lat:52.2043, lon:0.1149},
    { id:"site-ucinci", name:"University of Cincinnati", city:"Cincinnati, OH, USA", lat:39.1311, lon:-84.5160},
    { id:"site-ucdublin", name:"University College Dublin", city:"Dublin, Ireland", lat:53.3080, lon:-6.2236},
    { id:"site-osu", name:"Oklahoma State University", city:"Stillwater, OK, USA", lat:36.1255, lon:-97.0720},
    { id:"site-kaist", name:"Korea Advanced Institute of Science & Technology (KAIST), Republic of Korea", city:"Daejeon, Republic of Korea", lat:36.3723, lon:127.3622},



  ];

  const map = L.map('worldmap', { scrollWheelZoom: false });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors, &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);

  // Scroll to and highlight a card
  window.focusSite = function(id){
    const cards = document.querySelectorAll('.site-card');
    cards.forEach(c => c.classList.remove('expanded'));   // collapse any open one

    const el = document.getElementById(id);
    if(!el) return;

    el.classList.add('expanded');                         // expand this one
    el.scrollIntoView({ behavior:'smooth', block:'start' });

    // Optional: briefly pulse the border to draw the eye
    el.style.willChange = 'box-shadow, border-color';
    setTimeout(() => { el.style.willChange = 'auto'; }, 600);
  };
  // Add markers
  const markers = [];
  sites.forEach(s => {
    const m = L.marker([s.lat, s.lon], s.icon ? {icon:s.icon} : {}).addTo(map);
    m.bindPopup(`
      <h4>${s.name}</h4>
      <p>${s.city}</p>
      <p style="margin-top:.4rem"><a href="#" onclick="focusSite('${s.id}'); return false;">More details</a></p>
    `);
    markers.push(m);
  });

  // Fit bounds
  if (markers.length) {
    const group = L.featureGroup(markers);
    map.fitBounds(group.getBounds().pad(0.2));
  } else {
    map.setView([20, 0], 2);
  }

  window.addEventListener('resize', () => map.invalidateSize());
</script>
{% endraw %}

<p class="hint">Pick the site that’s closest to you (or choose <strong>Online</strong>). Final room details and building access instructions will be emailed to registered participants.</p>

<style>
/* Sites page card styles */
.sites-wrap, .sites-wrap * { box-sizing: border-box; }
.sites-grid{ display:grid; gap:1rem; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin: 1rem 0 2rem; }
.site-card{ background:#fff; border:1px solid #e6e6e6; border-radius:14px; padding:1rem 1rem 1.1rem; box-shadow:0 6px 18px rgba(0,0,0,.05); }
.site-card h3{ margin:.2rem 0 .4rem; font-size:1.1rem; }
.site-meta{ color:#555; font-size:.95rem; margin:.3rem 0 .6rem; }
.site-actions a{ display:inline-block; padding:.5rem .75rem; border-radius:10px; margin-right:.4rem; margin-top:.3rem; text-decoration:none; font-weight:600; border:1px solid #d6d6d6; background:#fafafa; }
.site-actions a:hover{ border-color:#3a7bd5; box-shadow:0 4px 12px rgba(58,123,213,.18); }
.badge{ display:inline-block; font-size:.78rem; padding:.18rem .5rem; border-radius:999px; background:#eef2ff; color:#334155; border:1px solid #c7d2fe; }
.btn-primary{ display:inline-block; padding:.6rem 1rem; border-radius:10px; border:1px solid #2e6bd6; background:#3a7bd5; color:#fff; font-weight:700; text-decoration:none; }
.hint{ font-size:.95rem; color:#555; }
</style>

<style>
/* Fix: make Register buttons always look like buttons */
.site-actions .btn-primary,
.site-actions .btn-primary:link,
.site-actions .btn-primary:visited {
  background: #3a7bd5 !important;
  border-color: #2e6bd6 !important;
  color: #fff !important;
}

.site-actions .btn-primary:hover,
.site-actions .btn-primary:focus {
  background: #325fbf !important;
  color: #fff !important;
  text-decoration: none;
}

.site-actions .btn-primary:active {
  background: #2d55a9 !important;
  color: #fff !important;
}
</style>

<div class="sites-wrap">
  <div class="sites-grid">

    <!-- UTK -->
    <div class="site-card" id="site-utk">
      <span class="badge">Tennessee, USA</span>
      <h3>University of Tennessee, Knoxville (UTK)</h3>
      <img class="site-logo" src="{{ '/assets/UTK.jpg' | relative_url }}" alt="UTK logo">
      <div class="site-meta">
        Knoxville, TN • Building/Room: <em>TBD</em><br>
        Local lead: <a href="mailto:sergei2@utk.edu">sergei2@utk.edu</a>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Tennessee%20Knoxville" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Tennessee,%20Knoxville" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- NCSU -->
    <div class="site-card" id="site-ncsu">
      <span class="badge">North Carolina, USA</span>
      <h3>North Carolina State University (NCSU)</h3>
      <img class="site-logo" src="{{ '/assets/ncsu.png' | relative_url }}" alt="NCSU logo">
      <div class="site-meta">
        Raleigh, NC • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=North%20Carolina%20State%20University" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=North%20Carolina%20State%20University" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Northwestern -->
    <div class="site-card" id="site-nwu">
      <span class="badge">Illinois, USA</span>
      <h3>Northwestern University</h3>
      <img class="site-logo" src="{{ '/assets/northwestern.jpg' | relative_url }}" alt="Northwestern logo">
      <div class="site-meta">
        Evanston, IL • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Northwestern%20University" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Northwestern%20University" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- UIC -->
    <div class="site-card" id="site-uic">
      <span class="badge">Illinois, USA</span>
      <h3>University of Illinois at Chicago (UIC)</h3>
      <img class="site-logo" src="{{ '/assets/UIC.jpg' | relative_url }}" alt="UIC logo">
      <div class="site-meta">
        Chicago, IL • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Illinois%20at%20Chicago" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Illinois%20at%20Chicago" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- ICN2 -->
    <div class="site-card" id="site-icn2">
      <span class="badge">Barcelona, Spain</span>
      <h3>ICN2 — Institut Català de Nanociència i Nanotecnologia</h3>
     <div class="site-logo-row" style="display:flex; gap:10px; align-items:center;">
      <img class="site-logo" src="{{ '/assets/ICN2.png' | relative_url }}" alt="ICN2 logo" style="max-height:44px;">
      <img class="site-logo" src="{{ '/assets/ALBA.png' | relative_url }}" alt="ALBA logo" style="max-height:44px;">
     </div>
      
      <div class="site-meta">
        ALBA Synchrotron, Cerdanyola del Vallès • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=ICN2%20Barcelona" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Institut%20Catal%C3%A0%20de%20Nanoci%C3%A8ncia%20i%20Nanotecnologia%20(ICN2),%20Barcelona" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Toronto -->
    <div class="site-card" id="site-utor">
      <span class="badge">Ontario, Canada</span>
      <h3>University of Toronto</h3>
      <img class="site-logo" src="{{ '/assets/UofT.png' | relative_url }}" alt="University of Toronto logo">
      <div class="site-meta">
        Toronto, ON • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Toronto" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Toronto" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Wisconsin -->
    <div class="site-card" id="site-uwisc">
      <span class="badge">Wisconsin, USA</span>
      <h3>University of Wisconsin</h3>
      <img class="site-logo" src="{{ '/assets/uw.png' | relative_url }}" alt="University of Wisconsin logo">
      <div class="site-meta">
        Madison, WI • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Wisconsin%20Madison" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Wisconsin" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Colorado (Boulder) -->
    <div class="site-card" id="site-ucolo">
      <span class="badge">Colorado, USA</span>
      <h3>University of Colorado</h3>
      <img class="site-logo" src="{{ '/assets/UCB.png' | relative_url }}" alt="University of Colorado logo">
      <div class="site-meta">
        Boulder, CO • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Colorado%20Boulder" target="_blank" rel="noopener">Map</a>
        <!-- match registration option EXACTLY ("University of Colorado") -->
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Colorado" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Colorado School of Mines -->
    <div class="site-card" id="site-mines">
      <span class="badge">Colorado, USA</span>
      <h3>Colorado School of Mines</h3>
      <img class="site-logo" src="{{ '/assets/Mines.png' | relative_url }}" alt="Colorado School of Mines logo">
      <div class="site-meta">
        Golden, CO • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Colorado%20School%20of%20Mines%20Golden" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Colorado%20School%20of%20Mines" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- IIT Delhi -->
    <div class="site-card" id="site-iitd">
      <span class="badge">Delhi, India</span>
      <h3>Indian Institute of Technology Delhi</h3>
      <img class="site-logo" src="{{ '/assets/IITD.png' | relative_url }}" alt="IIT Delhi logo">
      <div class="site-meta">
        New Delhi, India • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Indian%20Institute%20of%20Technology%20Delhi" target="_blank" rel="noopener">Map</a>
        <!-- IMPORTANT: this query string MUST match the checkbox value on /registration/ -->
        <a href="{{ '/registration/' | relative_url }}?site=Indian%20Institute%20of%20Technology%20Delhi%20(IIT%20Delhi)" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Johns Hopkins University -->
    <div class="site-card" id="site-jhu">
      <span class="badge">Maryland, USA</span>
      <h3>Johns Hopkins University</h3>
      <img class="site-logo" src="{{ '/assets/JHU.png' | relative_url }}" alt="Johns Hopkins University logo">
      <div class="site-meta">
        Baltimore, MD • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Johns%20Hopkins%20University%20Baltimore" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Johns%20Hopkins%20University" class="btn-primary">Register</a>
      </div>
    </div>


    <!-- Thermo Fisher Scientific Eindhoven -->
    <div class="site-card" id="site-tfs">
      <span class="badge">Eindhoven, Netherlands</span>
      <h3>Thermo Fisher Scientific – Eindhoven</h3>
      <img class="site-logo" src="{{ '/assets/tf_logo.png' | relative_url }}" alt="Thermo Fisher Scientific logo">
      <div class="site-meta">
        Eindhoven, Netherlands • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Thermo%20Fisher%20Scientific%20Eindhoven" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Thermo%20Fisher%20Scientific%20%E2%80%93%20Eindhoven" class="btn-primary">Register</a>
      </div>
    </div>
    
    <!-- AISCIA Informatics – Doha -->
    <div class="site-card" id="site-aiscia">
      <span class="badge">Doha, Qatar</span>
      <h3>AISCIA Informatics – Doha, Qatar</h3>
      <img class="site-logo" src="{{ '/assets/AISCIA.png' | relative_url }}" alt="AISCIA Informatics logo">
      <div class="site-meta">
        Qatar Science and Technology Park, Doha • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Qatar%20Science%20and%20Technology%20Park%20Doha" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=AISCIA%20Informatics%20%E2%80%93%20Doha,%20Qatar" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Pennsylvania State University -->
    <div class="site-card" id="site-psu">
      <span class="badge">Pennsylvania, USA</span>
      <h3>Pennsylvania State University</h3>
      <img class="site-logo" src="{{ '/assets/PSU.png' | relative_url }}" alt="Pennsylvania State University logo">
      <div class="site-meta">
        State College, PA • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Pennsylvania%20State%20University%20State%20College" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Pennsylvania%20State%20University" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- University of Pennsylvania -->
    <div class="site-card" id="site-upenn">
      <span class="badge">Pennsylvania, USA</span>
      <h3>University of Pennsylvania</h3>
      <img class="site-logo" src="{{ '/assets/UPenn.png' | relative_url }}" alt="University of Pennsylvania logo">
      <div class="site-meta">
        Philadelphia, PA • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Pennsylvania%20Philadelphia" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Pennsylvania" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- University of Michigan, Ann Arbor -->
    <div class="site-card" id="site-umich">
      <span class="badge">Michigan, USA</span>
      <h3>University of Michigan, Ann Arbor</h3>
      <img class="site-logo" src="{{ '/assets/UMich.png' | relative_url }}" alt="University of Michigan logo">
      <div class="site-meta">
        Ann Arbor, MI • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Michigan%20Ann%20Arbor" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Michigan,%20Ann%20Arbor" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Sungkyunkwan University (SKKU) -->
    <div class="site-card" id="site-skku">
      <span class="badge">Republic of Korea</span>
      <h3>Sungkyunkwan University (SKKU)</h3>
      <img class="site-logo" src="{{ '/assets/SKKU.jpg' | relative_url }}" alt="SKKU logo">
      <div class="site-meta">
        Suwon, Republic of Korea • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Sungkyunkwan%20University%20Suwon%20Korea" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Sungkyunkwan%20University%20(SKKU),%20Republic%20of%20Korea" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Nanyang Technological University (NTU), Singapore -->
    <div class="site-card" id="site-ntu">
      <span class="badge">Singapore</span>
      <h3>Nanyang Technological University (NTU), Singapore</h3>
      <img class="site-logo" src="{{ '/assets/NTU.png' | relative_url }}" alt="NTU logo">
      <div class="site-meta">
        Singapore • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Nanyang%20Technological%20University%20Singapore" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Nanyang%20Technological%20University%20(NTU),%20Singapore" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Texas A&M University -->
    <div class="site-card" id="site-tamu">
      <span class="badge">Texas, USA</span>
      <h3>Texas A&M University</h3>
      <img class="site-logo" src="{{ '/assets/TAMU.png' | relative_url }}" alt="Texas A&M University logo">
      <div class="site-meta">
        College Station, TX • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Texas%20A%26M%20University%20College%20Station" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Texas%20A%26M%20University" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- University of Cambridge -->
    <div class="site-card" id="site-cambridge">
      <span class="badge">United Kingdom</span>
      <h3>University of Cambridge</h3>
      <img class="site-logo" src="{{ '/assets/Cambridge.png' | relative_url }}" alt="University of Cambridge logo">
      <div class="site-meta">
        Cambridge, UK • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Cambridge" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Cambridge" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- University of Cincinnati -->
    <div class="site-card" id="site-ucinci">
      <span class="badge">Ohio, USA</span>
      <h3>University of Cincinnati</h3>
      <img class="site-logo" src="{{ '/assets/UCinci.png' | relative_url }}" alt="University of Cincinnati logo">
      <div class="site-meta">
        Cincinnati, OH • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=University%20of%20Cincinnati" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=University%20of%20Cincinnati" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- University College Dublin -->
    <div class="site-card" id="site-ucdublin">
    <span class="badge">Ireland</span>
    <h3>University College Dublin, Ireland</h3>
    <img class="site-logo" src="{{ '/assets/UCDublin.jpg' | relative_url }}" alt="University College Dublin logo">
    <div class="site-meta">
      Dublin, Ireland • Building/Room: <em>TBD</em><br>
      Contact: <em>TBD</em>
    </div>
    <div class="site-actions">
      <a href="https://maps.google.com/?q=University%20College%20Dublin" target="_blank" rel="noopener">Map</a>
      <a href="{{ '/registration/' | relative_url }}?site=University%20College%20Dublin,%20Ireland" class="btn-primary">Register</a>
    </div>
    </div>

    <!-- Oklahoma State University -->
    <div class="site-card" id="site-osu">
      <span class="badge">Oklahoma, USA</span>
      <h3>Oklahoma State University</h3>
      <img class="site-logo" src="{{ '/assets/OSU.png' | relative_url }}" alt="Oklahoma State University logo">
      <div class="site-meta">
        Stillwater, OK • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=Oklahoma%20State%20University%20Stillwater" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Oklahoma%20State%20University" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Korea Advanced Institute of Science & Technology (KAIST), Daejeon -->
    <div class="site-card" id="site-kaist">
      <span class="badge">Republic of Korea</span>
      <h3>Korea Advanced Institute of Science & Technology (KAIST), Republic of Korea</h3>
      <img class="site-logo" src="{{ '/assets/KAIST.png' | relative_url }}" alt="KAIST logo">
      <div class="site-meta">
        Daejeon, Republic of Korea • Building/Room: <em>TBD</em><br>
        Contact: <em>TBD</em>
      </div>
      <div class="site-actions">
        <a href="https://maps.google.com/?q=KAIST%20Daejeon" target="_blank" rel="noopener">Map</a>
        <a href="{{ '/registration/' | relative_url }}?site=Korea%20Advanced%20Institute%20of%20Science%20%26%20Technology%20(KAIST),%20Republic%20of%20Korea" class="btn-primary">Register</a>
      </div>
    </div>

    <!-- Online (no pin) -->
    <div class="site-card" id="site-online">
      <span class="badge">Global</span>
      <h3>Online</h3>
      <div class="site-meta">
        Participate remotely via Zoom + Slack<br>
        Access details will be emailed after registration.
      </div>
      <div class="site-actions">
        <a href="{{ '/registration/' | relative_url }}?site=Online" class="btn-primary">Register</a>
      </div>
    </div>

  </div>
</div>
<style>
/* allow grid items to re-pack when one expands */
.sites-grid{
  grid-auto-flow: dense;   /* let items fill gaps */
}

/* default card transitions */
.site-card{
  transition: box-shadow .2s ease, transform .2s ease, border-color .2s ease;
}

/* the expanded state */
.site-card.expanded{
  grid-column: span 2;                     /* take two columns */
  border-color: #3a7bd5;
  box-shadow: 0 12px 28px rgba(58,123,213,.22);
  transform: translateY(-2px);
}

/* optional: reveal extra content area only when expanded */
.site-card .site-more{ display:none; }
.site-card.expanded .site-more{ display:block; margin-top:.6rem; color:#333; }

/* on narrow screens, keep normal size */
@media (max-width: 720px){
  .site-card.expanded{ grid-column: span 1; transform:none; }
}

/* on very wide screens, let expanded cards take 3 columns */
@media (min-width: 1200px){
  .site-card.expanded { grid-column: span 3; }
}
</style>
