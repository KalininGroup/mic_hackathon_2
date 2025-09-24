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
      <label><input type="checkbox" name="entry.1753222212" value="Online"> Online</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Tennessee, Knoxville"> University of Tennessee, Knoxville</label>
      <label><input type="checkbox" name="entry.1753222212" value="North Carolina State University"> North Carolina State University</label>
      <label><input type="checkbox" name="entry.1753222212" value="Northwestern University"> Northwestern University</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Illinois at Chicago"> University of Illinois at Chicago</label>
      <label><input type="checkbox" name="entry.1753222212" value="Institut Català de Nanociència i Nanotecnologia (ICN2), Barcelona"> Institut Català de Nanociència i Nanotecnologia (ICN2), Barcelona</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Toronto"> University of Toronto</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Wisconsin"> University of Wisconsin</label>
      <label><input type="checkbox" name="entry.1753222212" value="University of Colorado"> University of Colorado</label>
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
