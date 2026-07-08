/* ============================================================
   KINOMORPHIC — behavior
   ============================================================ */
(function () {
  'use strict';
  var LS = { theme: 'kino-theme', temp: 'kino-temp', type: 'kino-type' };
  var DEF = { theme: 'light', temp: 'mono', type: 'paper' };

  function get(k) { try { return localStorage.getItem(LS[k]) || DEF[k]; } catch (e) { return DEF[k]; } }
  function set(k, v) { try { localStorage.setItem(LS[k], v); } catch (e) {} }

  function apply(settings) {
    var r = document.documentElement;
    if (settings.theme) r.setAttribute('data-theme', settings.theme);
    if (settings.temp) r.setAttribute('data-temp', settings.temp);
    if (settings.type) r.setAttribute('data-type', settings.type);
    syncControls();
  }

  // expose for inline head script + controls
  window.Kino = window.Kino || {};
  window.Kino.set = function (k, v) { set(k, v); var s = {}; s[k] = v; apply(s); };
  window.Kino.get = get;

  function current() { return { theme: get('theme'), temp: get('temp'), type: get('type') }; }

  /* ---------- sync any on-page controls to current state ---------- */
  function syncControls() {
    var s = current();
    document.querySelectorAll('[data-set]').forEach(function (el) {
      var key = el.getAttribute('data-set'), val = el.getAttribute('data-val');
      el.classList.toggle('on', s[key] === val);
      el.setAttribute('aria-pressed', s[key] === val);
    });
    document.querySelectorAll('.js-theme').forEach(function (b) {
      b.setAttribute('aria-pressed', s.theme === 'dark');
    });
  }

  /* ============================================================
     SIGNATURE MARK  — "shape emerging from movement"
     Variants: ring (primary), trajectory, aperture, lissajous
     ============================================================ */
  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(name, attrs) {
    var e = document.createElementNS(SVGNS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function svgRoot(extra) {
    var s = el('svg', { viewBox: '0 0 100 100', class: 'kmark ' + (extra || '') });
    s.setAttribute('aria-hidden', 'true');
    return s;
  }

  // Primary: discrete checkpoints settle onto a ring; one live marker orbits.
  function markRing(opts) {
    opts = opts || {};
    var n = opts.dots || 14, R = 33, cx = 50, cy = 50;
    var s = svgRoot('mk-ring');
    // faint guide ring
    s.appendChild(el('circle', { cx: cx, cy: cy, r: R, fill: 'none', stroke: 'currentColor', 'stroke-width': 1, 'stroke-opacity': 0.16 }));
    // inward spiral tail (the path of arrival) — drawn faint
    var tail = '';
    for (var t = 0; t <= 1; t += 0.04) {
      var ang = -Math.PI / 2 + t * Math.PI * 1.6;
      var rr = R * (0.18 + 0.82 * t);
      var x = cx + rr * Math.cos(ang), y = cy + rr * Math.sin(ang);
      tail += (t === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }
    var sp = el('path', { d: tail, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.1, 'stroke-opacity': 0.22, 'stroke-linecap': 'round' });
    s.appendChild(sp);
    // settling dots
    var dots = [];
    for (var i = 0; i < n; i++) {
      var a = -Math.PI / 2 + (i / n) * Math.PI * 2;
      var x2 = cx + R * Math.cos(a), y2 = cy + R * Math.sin(a);
      var c = el('circle', { cx: x2.toFixed(2), cy: y2.toFixed(2), r: 2.6, fill: 'currentColor' });
      if (opts.animate) {
        c.style.transformOrigin = cx + 'px ' + cy + 'px';
        c.style.opacity = '0';
        c.style.animation = 'kmSettle .62s var(--ease,ease) forwards';
        c.style.animationDelay = (0.15 + i * 0.045) + 's';
      }
      s.appendChild(c); dots.push({ x: x2, y: y2 });
    }
    // defensive: if the animation clock never advanced (throttled/bg tab), force visible
    if (opts.animate) {
      var settleDots = s.querySelectorAll('circle[style]');
      setTimeout(function () { settleDots.forEach(function (c) { c.style.opacity = '1'; }); }, 2000);
    }
    // live orbiting marker (vermilion) — the only moving element after settle
    var live = el('circle', { r: 3.2, fill: 'var(--red, crimson)' });
    s.appendChild(live);
    if (opts.animate && !reduced()) {
      var start = performance.now() + 900;
      (function spin(now) {
        var el2 = live;
        var p = ((now - start) / 9000) % 1; if (p < 0) p = 0;
        var a2 = -Math.PI / 2 + p * Math.PI * 2;
        el2.setAttribute('cx', (cx + R * Math.cos(a2)).toFixed(2));
        el2.setAttribute('cy', (cy + R * Math.sin(a2)).toFixed(2));
        el2.setAttribute('opacity', now < start ? 0 : 1);
        requestAnimationFrame(spin);
      })(performance.now());
    } else {
      var a3 = -Math.PI / 2;
      live.setAttribute('cx', cx + R * Math.cos(a3)); live.setAttribute('cy', cy + R * Math.sin(a3));
    }
    return s;
  }

  // Trajectory: a sweeping path with checkpoint dots + head marker.
  function markTrajectory() {
    var s = svgRoot('mk-traj');
    var d = 'M14 78 C 26 70, 30 40, 50 38 S 78 56, 86 22';
    s.appendChild(el('path', { d: d, fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-opacity': 0.85, 'stroke-linecap': 'round' }));
    var pts = [[14, 78], [22, 66], [29, 52], [38, 41], [50, 38], [63, 43], [74, 49], [86, 22]];
    pts.forEach(function (p, i) {
      var last = i === pts.length - 1;
      s.appendChild(el('circle', { cx: p[0], cy: p[1], r: last ? 3.6 : 2.1, fill: last ? 'var(--red,crimson)' : 'currentColor', 'fill-opacity': last ? 1 : 0.55 }));
    });
    return s;
  }

  // Aperture: concentric arc segments (phase).
  function markAperture() {
    var s = svgRoot('mk-ap');
    [[30, 0.55, 0.16], [22, 0.42, 0.4], [14, 0.3, 0.85]].forEach(function (a) {
      var r = a[0], frac = a[1], op = a[2];
      var a0 = -Math.PI / 2, a1 = a0 + frac * Math.PI * 2;
      var x0 = 50 + r * Math.cos(a0), y0 = 50 + r * Math.sin(a0);
      var x1 = 50 + r * Math.cos(a1), y1 = 50 + r * Math.sin(a1);
      s.appendChild(el('path', { d: 'M' + x0.toFixed(1) + ' ' + y0.toFixed(1) + ' A' + r + ' ' + r + ' 0 0 1 ' + x1.toFixed(1) + ' ' + y1.toFixed(1), fill: 'none', stroke: 'currentColor', 'stroke-width': 3, 'stroke-opacity': op, 'stroke-linecap': 'round' }));
    });
    s.appendChild(el('circle', { cx: 50, cy: 50, r: 2.6, fill: 'var(--red,crimson)' }));
    return s;
  }

  // Lissajous: a continuous parametric curve (form drawn by motion).
  function markLissajous() {
    var s = svgRoot('mk-lj');
    var d = '', A = 32;
    for (var t = 0; t <= Math.PI * 2 + 0.05; t += 0.05) {
      var x = 50 + A * Math.sin(3 * t + Math.PI / 2), y = 50 + A * Math.sin(2 * t);
      d += (t === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
    }
    s.appendChild(el('path', { d: d, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.8, 'stroke-opacity': 0.8, 'stroke-linejoin': 'round' }));
    return s;
  }

  var BUILDERS = { ring: markRing, trajectory: markTrajectory, aperture: markAperture, lissajous: markLissajous };
  function reduced() { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches; }

  function mountMarks() {
    document.querySelectorAll('[data-mark]').forEach(function (host) {
      if (host.dataset.mounted) return;
      host.dataset.mounted = '1';
      var kind = host.getAttribute('data-mark') || 'ring';
      var animate = host.hasAttribute('data-animate') && !reduced();
      var svg = BUILDERS[kind] ? BUILDERS[kind]({ animate: animate, dots: host.getAttribute('data-dots') ? +host.getAttribute('data-dots') : undefined }) : markRing({ animate: animate });
      host.appendChild(svg);
    });
  }

  /* ============================================================
     REVEAL ON SCROLL
     ============================================================ */
  function reveals() {
    var els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window) || reduced()) { els.forEach(function (e) { e.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (ents) {
      ents.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    els.forEach(function (e) { io.observe(e); });
  }

  /* ============================================================
     FIELDNOTES FILTER
     ============================================================ */
  function filters() {
    var bar = document.querySelector('[data-filterbar]');
    if (!bar) return;
    var posts = Array.prototype.slice.call(document.querySelectorAll('[data-tags]'));
    var countEl = document.querySelector('[data-count]');
    var active = new Set();
    function run() {
      var shown = 0;
      posts.forEach(function (p) {
        var tags = (p.getAttribute('data-tags') || '').split(/\s+/);
        var ok = active.size === 0 || Array.from(active).some(function (a) { return tags.indexOf(a) > -1; });
        p.style.display = ok ? '' : 'none';
        if (ok) shown++;
      });
      if (countEl) countEl.textContent = shown + (shown === 1 ? ' post' : ' posts');
    }
    bar.querySelectorAll('.tag').forEach(function (t) {
      t.addEventListener('click', function () {
        var tag = t.getAttribute('data-tag');
        if (tag === '*') { active.clear(); bar.querySelectorAll('.tag').forEach(function (x) { x.classList.remove('on'); }); t.classList.add('on'); run(); return; }
        bar.querySelector('[data-tag="*"]').classList.remove('on');
        if (active.has(tag)) { active.delete(tag); t.classList.remove('on'); } else { active.add(tag); t.classList.add('on'); }
        if (active.size === 0) bar.querySelector('[data-tag="*"]').classList.add('on');
        run();
      });
    });
  }

  /* ============================================================
     CHECKPOINT SCRUBBER (reading-experience mock)
     Crossfades a small sequence of stills as you scrub epochs.
     ============================================================ */
  function scrubbers() {
    document.querySelectorAll('[data-scrubber]').forEach(function (root) {
      var frames = Array.prototype.slice.call(root.querySelectorAll('[data-frame]'));
      var range = root.querySelector('input[type=range]');
      var epochOut = root.querySelector('[data-epoch]');
      var playBtn = root.querySelector('[data-play]');
      var marks = Array.prototype.slice.call(root.querySelectorAll('[data-jump]'));
      if (!frames.length || !range) return;
      var epochs = frames.map(function (f) { return +f.getAttribute('data-frame'); });
      range.min = 0; range.max = frames.length - 1; range.value = root.getAttribute('data-start') || (frames.length - 1);
      var playing = false, timer = null;
      function show(idx) {
        idx = Math.max(0, Math.min(frames.length - 1, idx));
        frames.forEach(function (f, i) { f.style.opacity = i === idx ? '1' : '0'; });
        if (epochOut) epochOut.textContent = epochs[idx].toLocaleString();
        range.value = idx;
        // progress fill
        var pct = frames.length > 1 ? (idx / (frames.length - 1)) * 100 : 0;
        range.style.setProperty('--fill', pct + '%');
        marks.forEach(function (m) {
          var e = +m.getAttribute('data-jump');
          m.classList.toggle('passed', epochs[idx] >= e);
        });
      }
      range.addEventListener('input', function () { stop(); show(+range.value); });
      marks.forEach(function (m) {
        m.addEventListener('click', function () {
          var e = +m.getAttribute('data-jump');
          var best = 0, bd = Infinity;
          epochs.forEach(function (ev, i) { var d = Math.abs(ev - e); if (d < bd) { bd = d; best = i; } });
          stop(); show(best);
        });
      });
      function step() { var v = (+range.value + 1) % frames.length; show(v); }
      function play() { playing = true; if (playBtn) playBtn.classList.add('on'); if (+range.value >= frames.length - 1) show(0); timer = setInterval(step, 900); }
      function stop() { playing = false; if (playBtn) playBtn.classList.remove('on'); if (timer) clearInterval(timer); timer = null; }
      if (playBtn) playBtn.addEventListener('click', function () { playing ? stop() : play(); });
      show(+range.value);
    });
  }

  /* ============================================================
     MORPH SCRUBBER — live schematic: residue classes move from a
     cloud (early training) into a ring (post-grokking) as you scrub.
     Honestly a schematic, but the *reading is in the movement*.
     ============================================================ */
  function rng(seed) { return function () { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }; }
  function smooth(a, b, x) { var t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

  function initMorph(root) {
    var canvas = root.querySelector('canvas');
    var range = root.querySelector('input[type=range]');
    var epochOut = root.querySelector('[data-epoch]');
    var playBtn = root.querySelector('[data-play]');
    var marks = Array.prototype.slice.call(root.querySelectorAll('[data-jump]'));
    if (!canvas || !range) return;
    var ctx = canvas.getContext('2d');
    var MAX = +root.getAttribute('data-max') || 25000;
    var ONSET = +root.getAttribute('data-onset') || 9500;
    var GROK = +root.getAttribute('data-grok') || 12400;
    var N = +root.getAttribute('data-n') || 60;
    range.min = 0; range.max = MAX; range.step = 100; range.value = +root.getAttribute('data-start') || 6000;

    // deterministic per-point cloud start + ring target
    var r = rng(7);
    var pts = [];
    for (var i = 0; i < N; i++) {
      var ang = (i / N) * Math.PI * 2 - Math.PI / 2;
      var cloudA = r() * Math.PI * 2, cloudR = 0.10 + r() * 0.34;
      pts.push({ ang: ang, hue: (i / N) * 360, cx: Math.cos(cloudA) * cloudR, cy: Math.sin(cloudA) * cloudR, jx: (r() - 0.5) * 0.05, jy: (r() - 0.5) * 0.05 });
    }
    var W, H, DPR;
    function resize() {
      DPR = Math.min(2, window.devicePixelRatio || 1);
      var rect = canvas.getBoundingClientRect();
      W = rect.width; H = rect.height || rect.width * 0.62;
      canvas.width = W * DPR; canvas.height = H * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    }
    function draw(epoch) {
      if (!W) resize();
      var t = smooth(ONSET - 2500, GROK + 1200, epoch);     // cloud -> ring
      var radius = smooth(ONSET - 3000, GROK + 2000, epoch); // mean radius grows
      var cx = W / 2, cy = H / 2, S = Math.min(W, H) * 0.40;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fcfcfa'; ctx.fillRect(0, 0, W, H);
      // faint guide ring (appears as structure forms)
      ctx.beginPath(); ctx.arc(cx, cy, S, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(70,90,120,' + (0.05 + 0.12 * t) + ')'; ctx.lineWidth = 1; ctx.stroke();
      // crosshair axes
      ctx.strokeStyle = 'rgba(120,130,150,0.10)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx - S * 1.25, cy); ctx.lineTo(cx + S * 1.25, cy); ctx.moveTo(cx, cy - S * 1.25); ctx.lineTo(cx, cy + S * 1.25); ctx.stroke();
      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        var tx = Math.cos(p.ang) * (0.55 + 0.45 * radius), ty = Math.sin(p.ang) * (0.55 + 0.45 * radius);
        var x = p.cx + (tx + p.jx * (1 - t) * 6 - p.cx) * t;
        var y = p.cy + (ty + p.jy * (1 - t) * 6 - p.cy) * t;
        var px = cx + x * S, py = cy + y * S;
        ctx.beginPath(); ctx.arc(px, py, 3.4, 0, Math.PI * 2);
        ctx.fillStyle = 'hsl(' + p.hue + ',62%,52%)'; ctx.globalAlpha = 0.92; ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (epochOut) epochOut.textContent = Math.round(epoch).toLocaleString();
      var pct = (epoch / MAX) * 100; range.style.setProperty('--fill', pct + '%');
      marks.forEach(function (m) { m.classList.toggle('passed', epoch >= +m.getAttribute('data-jump')); });
    }
    var playing = false, raf = null;
    function loop() {
      var v = +range.value + 220; if (v > MAX) { v = MAX; stop(); }
      range.value = v; draw(v);
      if (playing) raf = requestAnimationFrame(loop);
    }
    function play() { playing = true; if (playBtn) playBtn.classList.add('on'); if (+range.value >= MAX) { range.value = 0; } raf = requestAnimationFrame(loop); }
    function stop() { playing = false; if (playBtn) playBtn.classList.remove('on'); if (raf) cancelAnimationFrame(raf); }
    range.addEventListener('input', function () { stop(); draw(+range.value); });
    if (playBtn) playBtn.addEventListener('click', function () { playing ? stop() : play(); });
    marks.forEach(function (m) { m.addEventListener('click', function () { stop(); range.value = m.getAttribute('data-jump'); draw(+range.value); }); });
    window.addEventListener('resize', function () { resize(); draw(+range.value); });
    resize(); draw(+range.value);
  }
  function morphScrubbers() { document.querySelectorAll('[data-morph]').forEach(initMorph); }

  /* ---------- progress bar for post reading ---------- */
  function readingProgress() {
    var bar = document.querySelector('[data-progress]');
    if (!bar) return;
    function upd() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.transform = 'scaleX(' + (max > 0 ? Math.min(1, h.scrollTop / max) : 0) + ')';
    }
    document.addEventListener('scroll', upd, { passive: true }); upd();
  }

  /* ---------- boot ---------- */
  function boot() {
    apply(current());
    mountMarks();
    reveals();
    filters();
    scrubbers();
    morphScrubbers();
    readingProgress();
    // theme toggle buttons
    document.querySelectorAll('.js-theme').forEach(function (b) {
      b.addEventListener('click', function () { window.Kino.set('theme', get('theme') === 'dark' ? 'light' : 'dark'); });
    });
    // generic data-set controls (segmented buttons in settings/foundation)
    document.querySelectorAll('[data-set]').forEach(function (b) {
      b.addEventListener('click', function () { window.Kino.set(b.getAttribute('data-set'), b.getAttribute('data-val')); });
    });
    // settings popover toggle
    var sBtn = document.querySelector('.js-settings'), sPop = document.querySelector('#settings-pop');
    if (sBtn && sPop) {
      sBtn.addEventListener('click', function (e) { e.stopPropagation(); sPop.classList.toggle('open'); });
      document.addEventListener('click', function (e) { if (!sPop.contains(e.target) && e.target !== sBtn) sPop.classList.remove('open'); });
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
