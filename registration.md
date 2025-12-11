---
layout: page
title: Register for the Hackathon
menu_title: Register
menu_icon: pencil-square
permalink: /registration/
published: true
---

<style>
/* keep everything inside the card */
.reg-card, .reg-card * { box-sizing: border-box; }

.reg-card{
  max-width: 820px; margin: 1.2rem auto; padding: 1.2rem 1.4rem;
  background:#fff; border:1px solid #e6e6e6; border-radius:14px;
  box-shadow: 0 6px 18px rgba(0,0,0,.05);
  overflow:hidden;
}

.reg-form p, .reg-form fieldset{ margin: .9rem 0; }
.reg-form label{ font-weight:600; display:block; }
.reg-form input[type="text"],
.reg-form input[type="email"],
.reg-form textarea,
.reg-form select{
  width:100%; padding:.65rem .75rem; border:1px solid #d6d6d6; border-radius:10px;
  outline:none; background:#fafafa; transition: box-shadow .15s, border-color .15s, background .15s;
}
.reg-form textarea{ resize: vertical; min-height: 120px; }
.reg-form input:focus, .reg-form textarea:focus, .reg-form select:focus{
  border-color:#3a7bd5; background:#fff; box-shadow: 0 0 0 3px rgba(58,123,213,.15);
}

.reg-form fieldset{
  border:1px solid #eee; border-radius:12px; padding: .8rem 1rem;
}
.reg-form legend{ font-weight:700; padding:0 .4rem; }
.required{ color:#d00; }

/* prettier, even checklist: responsive grid */
.checkgrid{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap:.5rem 1rem;
}
.checkgrid label{
  display:flex; align-items:flex-start; gap:.5rem;
  padding:.45rem .6rem; border:1px solid #eee; border-radius:10px; background:#fafafa;
}
.checkgrid input{ margin-top:.2rem; }

/* GREYED-OUT CLOSED SITES */
.checkgrid label.site-closed{
  opacity: 0.45;
  cursor: not-allowed;
}
.checkgrid label.site-closed input{
  cursor: not-allowed;
}

/* button */
.btn-primary{
  display:inline-block; padding:.7rem 1.1rem; border-radius:10px;
  border:1px solid #2e6bd6; background:#3a7bd5; color:#fff; font-weight:700;
  text-decoration:none; cursor:pointer; transition: transform .03s ease, box-shadow .15s;
}
.btn-primary:hover{ box-shadow: 0 8px 18px rgba(58,123,213,.25); }
.btn-primary:active{ transform: translateY(1px); }
.hint{ font-size:.9rem; color:#666; margin-top:.3rem; }
</style>

{% raw %}
<iframe name="gform_target" id="gform_target" style="display:none;"></iframe>

<form class="reg-form"
      action="https://docs.google.com/forms/d/e/1FAIpQLScDGl0L5HVDjOKBpGQMLPIFekOiFywDBH_Kut02T9I-DwqpbQ/formResponse"
      method="POST"
      target="gform_target"
      id="onsite-registration-form">

  <!-- ↓↓↓ keep ALL your existing fields exactly as you have them ↓↓↓ -->

  <p>
    <label>Name <span class="required">*</span><br>
      <input type="text" name="entry.2092238618" required placeholder="Your full name">
    </label>
  </p>

  <p>
    <label>Email <span class="required">*</span><br>
      <input type="email" name="entry.1556369182" required placeholder="you@university.edu">
    </label>
  </p>

  <p>
    <label>Organization <span class="required">*</span><br>
      <input type="text" name="entry.479301265" required placeholder="e.g., University of Tennessee">
    </label>
  </p>

  <fieldset>
    <legend>Role in your Organization <span class="required">*</span></legend>
    <label><input type="radio" name="entry.2064945275" value="Undergraduate Student" required> Undergraduate Student</label>
    <label><input type="radio" name="entry.2064945275" value="Graduate Student"> Graduate Student</label>
    <label><input type="radio" name="entry.2064945275" value="Postdoctoral Associates"> Postdoctoral Associates</label>
    <label><input type="radio" name="entry.2064945275" value="Faculty"> Faculty</label>
    <label><input type="radio" name="entry.2064945275" value="Other"> Other</label>
    <div class="hint"><input type="text" name="entry.2064945275.other_option_response" placeholder="If Other, specify"></div>
  </fieldset>

  <fieldset>
    <legend>Where do you want to attend the hackathon? <span class="required">*</span></legend>
    <div class="checkgrid">

      <!-- ACTIVE OPTIONS (put these first so they appear together) -->
      <label><input type="checkbox" name="entry.1753222212" value="Online"> Online</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Cambridge">University of Cambridge</label>
      <label><input type="checkbox" name="entry.1753222212" value="Sungkyunkwan University (SKKU), Republic of Korea"> Sungkyunkwan University (SKKU), Republic of Korea</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Michigan, Ann Arbor"> University of Michigan, Ann Arbor </label>
    
      <!-- ALL OTHER SITES (greyed out by your JS) -->
      <label><input type="checkbox" name="entry.1753222212" value="University of Tennessee, Knoxville"> University of Tennessee, Knoxville</label>
      <label><input type="checkbox" name="entry.1753222212" value="North Carolina State University"> North Carolina State University</label>
      <label><input type="checkbox" name="entry.1753222212" value="Northwestern University"> Northwestern University</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Illinois at Chicago"> University of Illinois at Chicago</label>
      <label><input type="checkbox" name="entry.1753222212" value="Institut Català de Nanociència i Nanotecnologia (ICN2), Barcelona"> Institut Català de Nanociència i Nanotecnologia (ICN2) - ALBA Synchrotron, Barcelona</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Toronto"> University of Toronto</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Wisconsin"> University of Wisconsin</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Colorado"> University of Colorado</label>
      <label><input type="checkbox" name="entry.1753222212" value="Colorado School of Mines"> Colorado School of Mines</label>
      <label><input type="checkbox" name="entry.1753222212" value="Indian Institute of Technology Delhi (IIT Delhi)"> Indian Institute of Technology Delhi (IIT Delhi)</label>
      <label><input type="checkbox" name="entry.1753222212" value="Thermo Fisher Scientific – Eindhoven"> Thermo Fisher Scientific – Eindhoven</label>
      <label><input type="checkbox" name="entry.1753222212" value="Johns Hopkins University"> Johns Hopkins University</label>
      <label><input type="checkbox" name="entry.1753222212" value="AISCIA Informatics – Doha, Qatar"> AISCIA Informatics – Doha, Qatar </label>
      <label><input type="checkbox" name="entry.1753222212" value="Pennsylvania State University"> Pennsylvania State University </label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Pennsylvania"> University of Pennsylvania </label>
      
      <label><input type="checkbox" name="entry.1753222212" value="Nanyang Technological University (NTU), Singapore">Nanyang Technological University (NTU), Singapore </label>
      <label><input type="checkbox" name="entry.1753222212" value="Texas A&M University">Texas A&M University</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Cincinnati">University of Cincinnati</label>
      <label><input type="checkbox" name="entry.1753222212" value="University College Dublin, Ireland">University College Dublin, Ireland</label>
      <label><input type="checkbox" name="entry.1753222212" value="Oklahoma State University">Oklahoma State University</label>
      <label><input type="checkbox" name="entry.1753222212" value="Korea Advanced Institute of Science & Technology (KAIST), Republic of Korea">Korea Advanced Institute of Science & Technology (KAIST), Republic of Korea</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Illinois Urbana–Champaign (UIUC)">University of Illinois Urbana–Champaign (UIUC)</label>
    
    </div>
  </fieldset>

  <p>
    <label>What is your area of research?<br>
      <textarea name="entry.2109138769" rows="4" placeholder="e.g., electron microscopy, optimization, active learning"></textarea>
    </label>
  </p>

  <p><button class="btn-primary" type="submit" id="reg-submit">Submit registration</button></p>
</form>

<!-- success alert -->
<div id="reg-success" style="display:none; margin-top:.8rem; padding:.75rem 1rem; border:1px solid #c8e6c9; background:#e8f5e9; border-radius:10px; color:#256029;">
  ✅ Thanks! Your registration was received.
</div>

<script>
(function() {
  const form   = document.getElementById('onsite-registration-form');
  const btn    = document.getElementById('reg-submit');
  const ok     = document.getElementById('reg-success');
  const iframe = document.getElementById('gform_target');

  iframe.addEventListener('load', function () {
    if (!form.dataset.submitted) return;
    btn.disabled = false;
    btn.textContent = 'Submit registration';
    form.reset();
    ok.style.display = 'block';
    form.dataset.submitted = '';
  });

  form.addEventListener('submit', function () {
    ok.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Submitting...';
    form.dataset.submitted = '1';
  });
})();
</script>
{% endraw %}

<script>
(function(){
  const params = new URLSearchParams(location.search);
  const site = params.get('site');
  if(!site) return;

  // Find the checkbox group (entry.1753222212) and tick the one that matches
  const boxes = document.querySelectorAll('input[type="checkbox"][name="entry.1753222212"]');
  let matched = false;
  boxes.forEach(b => {
    if (b.value.trim() === site.trim()) {
      b.checked = true;
      matched = true;
    }
  });
  if (matched) {
    // Optionally scroll into view to show it's preselected
    const fieldset = boxes[0].closest('fieldset') || boxes[0].parentElement;
    fieldset && fieldset.scrollIntoView({behavior:'smooth', block:'start'});
  }
})();
</script>

<!-- LOCK ALL SITES EXCEPT ONLINE + CAMBRIDGE -->
<script>
(function(){
  const boxes = document.querySelectorAll('input[type="checkbox"][name="entry.1753222212"]');
  const MESSAGE = 'Registration is currently available only for Online, University of Cambridge, University of Michigan, Ann Arbor and Sungkyunkwan University (SKKU) sites.';

  boxes.forEach(box => {
    const val = box.value.trim();
    const isOpen = (val === 'Online' || val === 'University of Cambridge' || val === 'Sungkyunkwan University (SKKU), Republic of Korea' || val === 'University of Michigan, Ann Arbor');

    if (!isOpen) {
      // mark visual state
      const label = box.closest('label');
      if (label) label.classList.add('site-closed');

      // ensure it starts unchecked
      box.checked = false;

      // intercept clicks
      box.addEventListener('click', function(e){
        e.preventDefault();
        box.checked = false;
        alert(MESSAGE);
      });

      // also catch clicks on the label text
      const labelEl = box.closest('label');
      if (labelEl) {
        labelEl.addEventListener('click', function(e){
          e.preventDefault();
          box.checked = false;
          alert(MESSAGE);
        });
      }
    }
  });
})();
</script>
