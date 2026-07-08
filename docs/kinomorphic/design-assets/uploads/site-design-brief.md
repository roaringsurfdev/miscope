# Kinomorphic Research — Design Handover

> **Direction of this document.** This flows *from the project to Design.* It is
> authored from the repository, where the substance lives, and handed to Design
> Claude (claude.ai) as grounded source-of-truth. Design Claude's job is the
> **graphic and site design** — visual treatment, layout, typography, mockups,
> assets (JSX/HTML/PNG). It is **not** asked to design the system, the
> architecture, the build, or the publishing pipeline. Where this document names
> structure (pages, sections), it does so only so the visual design can reflect
> the right *weight and hierarchy* — not as an invitation to design the system.
>
> If a section tempts a system-design decision, that decision is made on the
> build side, not here. Those are marked **[build-owned]**.

---

## 1. What this is (and the register that governs everything)

**Kinomorphic Research** is the public identity for the independent research program that the MIScope libary and platform were built to support. The research focuses on the **learning dynamics of neural networks** — mechanistic
interpretability examined through a complex-/nonlinear-dynamical-systems lens.

The tone of the site should communicate quite confidence, professionalism, and scientific rigor. Being independent means there is no institution behind the research, so the site and the work should be able to speak for itself in quality.

This is the primary surface where results can be shared with the communities that might be interested in the research, whether that's other mechanistic interpretability researchers, computational neuroscientists, nonlinear dynamics researchers, or anyone starting out in these areas.

"Kinomorphic" as a name is meant to capture the role of movement in identifying shape. In the context of mechanistic interpretability, *kino-* (motion) + *-morphic* (form/shape) — the shape of how learning moves and the shape of representation that emerges over time. 

The name itself opens up other opportunities for interpretation, and thus leaves room for the research program to grow.


**Hard avoids:**
- Anything that reads as an AI product, startup, or "solution provider."
- Growth/urgency/hype signals; breathless AI copy; feature lists; CTAs.
- Leading with "AI Safety" or "Human-AI Collaboration" as front-door badges
  (see §4 — they belong downstream, as consequence, in prose).
- Dark, high-drama "launch page" palette with a single neon accent.
- Gradient hero sections, orb/blob backgrounds.

---

## 2. What the work actually is (grounded, for authentic copy)

Use these real facts as raw material so the design fills with truth, not
plausible invention:

- **The question:** *How did learning happen?* — not *did the model learn the
  task?*
- **The method:** the primary differentiator for this research is that it focuses on analyzing internal representation over training by examining models across a discrete series of checkpoints. 
- **Toy model callibration:** The current focus is on well-studied toy models, but extended to analysis over time instead of isolated to a final, static model. The toy models allow for callibration of useful lenses of analysis while also stress-testing the infrastructure that allows for this study.
- **The platform:** **MIScope** — a research infrastructure platform specifically designed to solve problems that make dynamics research on high-dimensional systems challenging - applied to model internals.

---

## 3. The applications of design treatment

**Kinomorphic.com** - the primary public surface for sharing research findings (via FieldNotes blog), access to the repository, and guidance on using the research platform

**Fieldnotes** - the Astro/MDX blog for posting research findings. Posts should be surfaced on the Kinomorphic.com site.

**MIScope library** - the middle tier component scheduled for publication (PyPi). This library encapsulates an ETL-like framework for analyzing model variations and their internals across checkpoints. This library also standarizes data visualizations to reduce drift and errors in data interpretation.

**Dashboard platform** - this is the primary interface for training, analyzing, and view model data. It is currently designed to run locally within a researcher's environment. As models under study grow, there are future plans to allow for remote hosting of the data behind the visualizations. Depending on future community interest, the dashboard may be evolved to support multi-user research environments.

**Plotly plots** - while there are some consistencies across plots, specifically around frequency coloring - the plots were not created under a shared visual language.

**Hero / signature image** - it would be nice to have some kind of interim signature image/logo that captures the kinomorphic concept. The primary use of this asset would be as a top navigation home link and a favicon to replace the Plotly Dash favicon and the default globe/missing favicon on GitHub pages.

---

## 4. Where "safety" and "collaboration" belong

The research genuinely connects to AI safety and to healthy human-AI
collaboration. A primary motivation to start this research was to better understand the results out of the Betley, et al (2025) paper in which fine tuning on a small task led to global misalignment. This is a motivator that may be elaborated on a static top-level page or within a post in the FieldNotes blog.

By examining model internals and how they are shaped over training, we can build deeper diagnostics for model health that might allow for more targeted steering during production training runs.

Design implication: these themes live **in prose, in the Approach section, lower
on the page** — never as top-of-hero chips.

---

## 5. Palette and type direction

The research shared on this site will contain many data visualizations, including animations of the data where appropriate. Because movement is a key axis in this research project, animations surfaced on the site will need to look intentional and well-designed.

The platform is heavily invested in Plotly figures, and while the figures themselves have been designed to convey rich information, they have not received an overall cohesive design treatment.

There are at least two color constraints on the data visualizations that should be flagged. These constraints are currently driven by the need to cleanly interpret results in the Modulo Addition Grokking model.

Residue classes and frequencies are rendered on a colorwheel to make them legible. Because there are many visualizations that show frequency behaviors, the frequency colorwheel is locked and standardized per prime to allow consistent tracking across visualizations.

It might not be possible to reduce the frequency and class colors down to a smaller color palette, but I'm open to explorations of this.

The site color palette will need to be able to support the legibility requirements of the data visualizations.

I'm open to supporting dark mode, as many engineers and researchers who stare at screens all day might appreciate this. It might present a design challenge, though, and it risks coming off as dramatic in the highly hyped AI space.

- **Body/UI:** geometric sans-serif, mathematical clarity.
- **Display/titles:** serif for titles and pull quotes; modest scale — titles do
  not shout.
- **Measure:** 65–75 characters for body — posts should read like papers.

**Tone references:** Transformer-Circuits.pub (understated rigor, no
institutional decoration), Distill.pub (interactive, deeply annotated figures),
3Blue1Brown (the work leads, the tool follows).

---

## 6. Site structure — for *weight and hierarchy*, not system design

```
kinomorphic.com
├── Home          — identity (research, not product); latest Fieldnotes; light MIScope mention
├── Fieldnotes    — PRIMARY. Post index (filterable) + visualization-heavy posts
├── Project / Approach — research direction; AI-collaboration philosophy; infrastructure rationale; about
└── MIScope       — SECONDARY/downstream: overview, library (PyPI), dashboard setup
```

**Navigation weight:** *Fieldnotes and Approach/Project are primary. MIScope is
present but visually secondary* — it is downstream of the research, discovered
through it, not the headline.

**Real content, so mockups don't confabulate an archive:** Fieldnotes currently
has **three** published posts. Use these real ones; do not invent a large back
catalog:
- *Starting Here* — 2026-03-28 — tags: meta
- *Variants and Variables* — 2026-03-29 — tags: infrastructure, methodology
- *Reproducibility* — 2026-03-30 — tags: infrastructure, methodology, research goals

(Many more are in progress as drafts; the published surface is intentionally
small. The design should make three good posts feel like a beginning, not a
sparse page begging to be filled.)

---

## 7. Components Design should treat as first-class

- **Post layout:** a reading column with a *breakout width* for multi-panel
  figure grids. Static MIScope figure embeds, captions, inline annotation, phase-
  transition markers. Posts read like short papers.
- **Checkpoint animation reading-experience:** the finding is often *in the
  movement* across training epochs. Design should treat scrubbing/playing through
  checkpoints as a native part of reading a post — not a bolted-on widget. *(The
  animation harness mechanics are **[build-owned]**; Design owns only how it
  feels to read.)*
- **Fieldnotes index:** browsable and filterable. Tagging spans audiences (mech
  interp / comp-neuro / nonlinear dynamics / new researchers) and analysis
  methods. *(The tagging taxonomy + filtering mechanics are **[build-owned]**;
  Design owns the index's look and the filter interaction.)*

---

## 8. Facts that prevent confabulation

Round 1 invented details; ground these:

- **MIScope** is a **Python** library (Plotly for figures, Dash for the
  dashboard), targeting **PyPI v1.0.0**. It is not a JS/TS project.
- **The dashboard** is **Plotly Dash** (Python), run **locally** by researchers
  who clone the repo. It is *not* TypeScript/WebGL and is *not* a hosted product.
- **Do not stamp invented version strings** on figures (e.g. "v0.3"). If a label
  is shown, leave it generic or omit it.
- **Domain:** kinomorphic.com. **GitHub org:** Kinomorphic. Staging today:
  roaringsurfdev.github.io/miscope (being retired into the new identity).
- **One monorepo**, open: tooling + dashboard + fieldnotes + site under one roof.
  Honest framing, not a product showcase — modules earn their own homes only when
  stable enough to deserve them.

---

## 9. Out of scope for Design (do not design these — **[build-owned]**)

- Static-site generator choice, Astro integrations, build/deploy pipeline.
- Monorepo strategy, repo splitting, PyPI/versioning, release coordination.
- The dashboard's internal architecture or analysis-view implementation.
- The animation harness implementation and the tagging taxonomy *mechanics*.

If a design exploration starts specifying any of the above, that is the signal it
has drifted out of lane — flag it back rather than designing it.

