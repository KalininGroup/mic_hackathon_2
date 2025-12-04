---
layout: page
title: FAQ
nav_order: 4
---

<style>
/* FAQ accordion styling, matching site cards */
.faq-item {
  margin: 18px 0;
  padding: 0;
  border: 1px solid #e5e7eb; /* light border like site */
  border-radius: 12px;        /* round corners */
  background: #ffffff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08); /* soft shadow */
}

/* SUMMARY (QUESTION) */
.faq-item summary {
  padding: 18px 22px;
  cursor: pointer;
  font-size: 1.05rem;
  font-weight: 500;  /* softer than bold */
  list-style: none;
  color: #111827;  /* match site text color */
  display: flex;
  align-items: center;
}

/* Remove default marker */
.faq-item summary::-webkit-details-marker {
  display: none;
}

/* Arrow */
.faq-item summary::before {
  content: "▸";
  font-size: 1.15rem;
  margin-right: 12px;
  transition: transform 0.2s ease;
  color: #2563eb;  /* subtle blue accent */
}

.faq-item[open] summary::before {
  transform: rotate(90deg);
}

/* ANSWER CONTENT */
.faq-item > *:not(summary) {
  padding: 16px 26px 22px 26px;
  font-size: 1rem;
  line-height: 1.6;
  color: #4b5563; /* neutral gray for body text */
}
</style>

<details class="faq-item">
  <summary><strong>Why should I participate?</strong></summary>

  <p>Whether you are coming from microscopy or ML/AI side, hackathon is an opportunity to make real impact.</p>

  <p>You will work on real microscopy and spectroscopy problems, gain hands-on experience with digital twin AFM/STEM simulators, and learn practical ML workflows used in research labs and industry.</p>

  <p>You will get visibility with judges and sponsors from universities, national labs, and companies — great for jobs, internships, collaborations, and graduate school.</p>

  <p>You also build a strong GitHub portfolio, connect with a global community across 20+ sites and online, and compete for multiple awards.</p>
</details>

---

<details class="faq-item">
  <summary><strong>Why organize a local site?</strong></summary>

  <p>Local sites create a stronger, more social hackathon experience. They allow participants to collaborate in person, form teams more easily, and get real-time guidance from local mentors.</p>

  <p>For organizers, it gives increased visibility for the institution, community engagement, easier outreach to local students, and opportunities to identify strong candidates for internships and research positions.</p>
</details>

---

<details class="faq-item">
  <summary><strong>What is the benefit to the community?</strong></summary>

  <p>The hackathon strengthens the global microscopy + ML ecosystem by producing open-source code, curated datasets, tutorials, and new ideas. It promotes cross-institution collaboration, connects microscopy scientists with the mainstream ML community, and encourages reproducible, shared tools that others can build on.</p>
</details>

---

<details class="faq-item">
  <summary><strong>What happens at a local site during the hackathon?</strong></summary>

  <p>Local sites host group work sessions, discussions, and social time (during lunch).</p>

  <p>Participants work together in teams, use shared resources, and stay connected to the global event through Slack and the main program. Sites can also run their own micro-activities — troubleshooting, brainstorming, short walkthroughs, etc.</p>
</details>

---

<details class="faq-item">
  <summary><strong>Who can participate?</strong></summary>

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

---

<details class="faq-item">
  <summary><strong>But what if I am not familiar with ML/Python?</strong></summary>

  <p>As long as you understand your problem from microscopy side, code assistants like ChatGPT and teaming up with the ML experts can be a way to proceed! Hackathon is also the environment to build interdisciplinary teams.</p>
</details>

---

<details class="faq-item">
  <summary><strong>Should the data and code be open?</strong></summary>

  <p>Yes. All submitted data, code, and results must be fully open.</p>

  <p>The goal of the hackathon is to create a shared ecosystem of tools, datasets, and workflows that the entire community can reuse and build upon. Open data ensures reproducibility, enables follow-up collaborations, and allows future participants to extend and improve the submitted projects. Openness is a core principle of this event.</p>
</details>

---

<details class="faq-item">
  <summary><strong>Who supports the awards, and how will they be paid?</strong></summary>

  <p>Awards are provided by sponsors such as Renaissance Philanthropy, Covalent Metrology, Thermo Fisher Scientific, Theia Scientific, MSA Student Council, Toyota Research Institute, Waviks, Polaron and others.</p>

  <p>Awards may be paid directly by the sponsor in cognizance of possible political and other restrictions. Some sponsors may choose winners independently, while others committed to follow jury recommendations.</p>
</details>

---

<details class="faq-item">
  <summary><strong>Can we provide our own data for participants?</strong></summary>

  <p>Yes. Additional datasets are welcome.</p>

  <p>Organizers ask for advance notice because datasets must be converted into the digital twin microscope–compatible format, documented, and added to the central GitHub repository.</p>

  <p>Rama Vasudevan (vasudevanrk@ornl.gov) and the core team will assist with formatting and integration.</p>
</details>

---

<details class="faq-item">
  <summary><strong>What level of coding experience is needed to participate?</strong></summary>

  <p>Basic Python is sufficient.</p>

  <p>We provide starter notebooks, data loaders, digital twin examples, and template workflows. More experienced coders can dive deeper into ML modeling, active learning, or real-time analysis, but beginners can still contribute meaningfully.</p>

  <p>If you are new to Python, this article is a great place to start:<br>
  “The New Language of Science: How to Learn Python Effectively” — <a href="https://medium.com/@sergei2vk/the-new-language-of-science-how-to-learn-python-effectively-c8ce51012a64" target="_blank" rel="noopener">https://medium.com/@sergei2vk/the-new-language-of-science-how-to-learn-python-effectively-c8ce51012a64</a></p>
</details>

---

<details class="faq-item">
  <summary><strong>How do teams form?</strong></summary>

  <p>Teams can form:</p>

  <ul>
    <li>Through the hackathon Slack channels (recommended)</li>
    <li>Also, at local sites</li>
  </ul>

  <p>We encourage teams with mixed expertise (ML, microscopy, physics, coding, domain knowledge), but single-person teams are also allowed.</p>
</details>

---

<details class="faq-item">
  <summary><strong>How is the hackathon organized?</strong></summary>

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
