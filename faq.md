---
layout: page
title:  Frequently Asked Questions
permalink: /faq/
nav_order: 4
menu_title: "FAQ"
---

<style>
.section-card {
  background: #fafbfd;
  border: 1px solid #e8ecf3;
  border-radius: 14px;
  padding: 26px 24px;
  margin: 32px 0;
  box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}

.section-card h2 {
  font-size: 1.3rem;
  color: #1d2a56;
  margin-top: 0;
  margin-bottom: 10px;
  font-weight: 600;
  border-left: 4px solid #b4c8ff;
  padding-left: 10px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

@media (max-width: 760px) {
  .grid-2 { grid-template-columns: 1fr; }
}

.table-soft {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0 6px;
}

.table-soft th {
  text-align: left;
  font-weight: 700;
  font-size: .95rem;
  color: #344054;
  padding: 10px 12px;
}

.table-soft td {
  background: #fff;
  border: 1px solid #eef0f5;
  border-radius: 10px;
  padding: 12px;
}

/* --- small helpers --- */
.badge {
  display: inline-block;
  font-size: .8rem;
  padding: .2rem .5rem;
  border-radius: 999px;
  background: #eef2ff;
  color: #334155;
  border: 1px solid #c7d2fe;
}

.logo-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  align-items: center;
}

.logo-row img {
  max-height: 56px;
  width: auto;
  height: auto;
  object-fit: contain;
  background: #fff;
  padding: .25rem .4rem;
  border-radius: 10px;
  border: 1px solid #eef0f5;
}

/* --- Core Team cards --- */
.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 16px;
}

.team-card {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: #fff;
  border: 1px solid #eef0f5;
  border-radius: 14px;
  padding: 14px;
}

.team-card img {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  object-fit: cover;
  border: 1px solid #e8ecf3;
  background: #fff;
}

.team-meta .name {
  font-weight: 700;
  color: #1d2a56;
}

.team-meta .affil {
  font-size: .92rem;
  color: #64748b;
}

.team-meta .role {
  font-size: .92rem;
  color: #475569;
}

/* --- clean inline social icons beside names --- */
.socials {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-left: 6px;
  margin-bottom: -2px;
}

.socials a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #1d4ed8;
  text-decoration: none;
  background: none;
  border: none;
  box-shadow: none;
  width: auto;
  height: auto;
  padding: 0;
  transition: color 0.2s ease;
}

.socials a:hover {
  color: #0f172a;
}

.socials svg {
  width: 17px;
  height: 17px;
  fill: currentColor;
  vertical-align: middle;
}

/* ========== FAQ-specific styling ========== */

/* remove inner padding of cards for <details> so we can control it */
details.section-card {
  padding: 0;
}

/* question row */
.faq-item summary {
  padding: 1rem 1.25rem;
  cursor: pointer;
  font-size: 1.05rem;
  font-weight: 600;
  list-style: none;
  display: flex;
  align-items: center;
  color: #1d2a56;
}

/* hide default marker */
.faq-item summary::-webkit-details-marker {
  display: none;
}

/* blue arrow icon */
.faq-item summary::before {
  content: "▸";
  font-size: 1.1rem;
  margin-right: 0.75rem;
  transition: transform 0.2s ease;
  color: #1d4ed8;
}

.faq-item[open] summary::before {
  transform: rotate(90deg);
}

/* answer content */
.faq-item > *:not(summary) {
  padding: 0 1.5rem 1.25rem 1.5rem;
  font-size: 0.98rem;
  line-height: 1.6;
  color: #4b5563;
}
</style>

<details class="section-card faq-item">
  <summary>Why should I participate?</summary>

  <p>Whether you are coming from microscopy or ML/AI side, hackathon is an opportunity to make real impact.</p>

  <p>You will work on real microscopy and spectroscopy problems, gain hands-on experience with digital twin AFM/STEM simulators, and learn practical ML workflows used in research labs and industry.</p>

  <p>You will get visibility with judges and sponsors from universities, national labs, and companies — great for jobs, internships, collaborations, and graduate school.</p>

  <p>You also build a strong GitHub portfolio, connect with a global community across 20+ sites and online, and compete for multiple awards.</p>
</details>

<details class="section-card faq-item">
  <summary>Why organize a local site?</summary>

  <p>Local sites create a stronger, more social hackathon experience. They allow participants to collaborate in person, form teams more easily, and get real-time guidance from local mentors.</p>

  <p>For organizers, it gives increased visibility for the institution, community engagement, easier outreach to local students, and opportunities to identify strong candidates for internships and research positions.</p>
</details>

<details class="section-card faq-item">
  <summary>What is the benefit to the community?</summary>

  <p>The hackathon strengthens the global microscopy + ML ecosystem by producing open-source code, curated datasets, tutorials, and new ideas. It promotes cross-institution collaboration, connects microscopy scientists with the mainstream ML community, and encourages reproducible, shared tools that others can build on.</p>
</details>

<details class="section-card faq-item">
  <summary>What happens at a local site during the hackathon?</summary>

  <p>Local sites host group work sessions, discussions, and social time (during lunch).</p>

  <p>Participants work together in teams, use shared resources, and stay connected to the global event through Slack and the main program. Sites can also run their own micro-activities — troubleshooting, brainstorming, short walkthroughs, etc.</p>
</details>

<details class="section-card faq-item">
  <summary>Who can participate?</summary>

  <p>Anyone with interest in ML, microscopy, imaging, or materials science:</p>

  <ul>
    <li>Undergraduate students</li>
    <li>Graduate students</li>
    <li>Postdocs</li>
    <li>Faculty</li>
    <li>Industry researchers</li>
    <li>Independent learners</li>
  </ul>

  <p>No prior microscopy experience is required.</p>
</details>

<details class="section-card faq-item">
  <summary>But what if I am not familiar with ML/Python?</summary>

  <p>As long as you understand your problem from microscopy side, code assistants like ChatGPT and teaming up with the ML experts can be a way to proceed! Hackathon is also the environment to build interdisciplinary teams.</p>
</details>

<details class="section-card faq-item">
  <summary>Should the data and code be open?</summary>

  <p>Yes. All submitted data, code, and results must be fully open.</p>

  <p>The goal of the hackathon is to create a shared ecosystem of tools, datasets, and workflows that the entire community can reuse and build upon. Open data ensures reproducibility, enables follow-up collaborations, and allows future participants to extend and improve the submitted projects. Openness is a core principle of this event.</p>
</details>

<details class="section-card faq-item">
  <summary>Who supports the awards, and how will they be paid?</summary>

  <p>Awards are provided by sponsors such as Renaissance Philanthropy, Covalent Metrology, Thermo Fisher Scientific, Theia Scientific, MSA Student Council, Toyota Research Institute, Waviks, Polaron and others.</p>

  <p>Awards may be paid directly by the sponsor in cognizance of possible political and other restrictions. Some sponsors may choose winners independently, while others committed to follow jury recommendations.</p>
</details>

<details class="section-card faq-item">
  <summary>Can we provide our own data for participants?</summary>

  <p>Yes. Additional datasets are welcome.</p>

  <p>Organizers ask for advance notice because datasets must be converted into the digital twin microscope–compatible format, documented, and added to the central GitHub repository.</p>

  <p>Rama Vasudevan (vasudevanrk@ornl.gov) and the core team will assist with formatting and integration.</p>
</details>

<details class="section-card faq-item">
  <summary>What level of coding experience is needed to participate?</summary>

  <p>Basic Python is sufficient.</p>

  <p>We provide starter notebooks, data loaders, digital twin examples, and template workflows. More experienced coders can dive deeper into ML modeling, active learning, or real-time analysis, but beginners can still contribute meaningfully.</p>

  <p>If you are new to Python, this article is a great place to start:<br>
  “The New Language of Science: How to Learn Python Effectively” — <a href="https://medium.com/@sergei2vk/the-new-language-of-science-how-to-learn-python-effectively-c8ce51012a64" target="_blank" rel="noopener">https://medium.com/@sergei2vk/the-new-language-of-science-how-to-learn-python-effectively-c8ce51012a64</a></p>
</details>

<details class="section-card faq-item">
  <summary>How do teams form?</summary>

  <p>Teams can form:</p>

  <ul>
    <li>Through the hackathon Slack channels (recommended)</li>
    <li>Also, at local sites</li>
  </ul>

  <p>We encourage teams with mixed expertise (ML, microscopy, physics, coding, domain knowledge), but single-person teams are also allowed.</p>
</details>

<details class="section-card faq-item">
  <summary>How is the hackathon organized?</summary>

  <ol>
    <li>
      <strong>Pre-Hackathon Launch (≈2 weeks before):</strong>
      We introduce the problems and datasets, show where participants can communicate (Slack, local sites) to form teams, and explain how to access the digital twins.
    </li>
    <li>
      <strong>Main Hackathon (3 days):</strong>
      Opening session, mentoring, Slack support, hands-on work with digital twins and datasets, collaboration across local sites and online, and final project submission.
    </li>
  </ol>

  <p>After the hackathon, organizers coordinate judging, feedback, and joint paper writing.</p>
</details>

