# Kinomorphic — design treatment (v1)

A design language and four reference pages for **kinomorphic.com**. Built as plain
HTML/CSS/JS so the system reads clearly and ports cleanly into the Astro/MDX site.

> Concept: *shape emerging from movement.* The identity borrows the research's own
> grammar — checkpoints tracing a path that resolves into structure (the residue ring,
> the parameter sweep). Motion is meaning, used sparingly.

## Chosen defaults
- **Type:** Paper — `Spectral` (serif display) · `IBM Plex Sans` (body/UI) · `IBM Plex Mono` (technical register)
- **Color:** Mono — near-neutral grays so the saturated figure colorwheels own the page
- **Accents:** ink-blue (structure) + vermilion (phase transition), both used sparingly
- **Appearance:** light-first, with a restrained dark toggle

Every page reads these from `localStorage` (`kino-theme` / `kino-temp` / `kino-type`),
so the controls in the top-right persist a visitor's choice across pages. Defaults are
set in the inline `<head>` script and in `assets/kino.js` (`DEF`).

## What's here
```
Foundation.html        Design-language system page (type, color, mark, figure frame, motion)
Home.html              Identity · latest Fieldnotes · approach · MIScope (secondary)
Fieldnotes.html        Filterable post index
Post.html              Reading column + breakout figure grids + checkpoint scrubber
assets/
  kino.css             Design tokens (theme/temperature/type) + base + components
  kino-ui.css          Richer components (nav popover, scrubber, post layout, specimens)
  kino.js              Theme/type/temp switching, the mark, filter, checkpoint scrubbers
  brand/
    favicon.svg            Settling Ring — primary mark/favicon (light+dark aware)
    favicon-lissajous.svg  Lissajous — alternate mark
    preview.html           Asset sheet showing both marks at favicon sizes
  figures/             Static Plotly stills used in the post (your screenshots)
```

## Drop-in vs. reference
- **Drop-in now:** `assets/brand/favicon.svg` — an SVG favicon with built-in light/dark
  support. Replaces the Plotly Dash / default GitHub-Pages favicon directly:
  `<link rel="icon" href="/favicon.svg" type="image/svg+xml">`
- **Port into the site:** the tokens in `kino.css` (`:root` custom properties) map onto
  a global stylesheet or Tailwind theme. The mark generator and checkpoint scrubber in
  `kino.js` are framework-agnostic and can wrap into Astro components / a small web
  component.
- **Reference templates:** the four `.html` pages are the visual spec for the Astro
  layouts — match the structure, spacing, and the figure-frame markup.

## The figure frame
Figures keep their existing Plotly rendering; cohesion comes from the *wrapper*
(`.figure` / `.figure-head` / `.figure-cap`): numbered `FIG n` label, sans title, mono
run metadata, captions in site type, on a true-white bed so dark mode never muddies a
plot. The locked frequency/residue colorwheels are never touched.

## Out of scope (build-owned)
Static-site/build pipeline, monorepo strategy, PyPI/versioning, the dashboard's
internals, and the animation-harness / tagging *mechanics*. The post's scrubber is a
**schematic** of the cloud→ring motion, not a data export — it can later be wired to a
real sequence of MIScope stills.
