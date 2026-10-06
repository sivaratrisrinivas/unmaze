/* Wire it together: load the data, run the interactive timeline, update the HUD, expose the film renderer. */
(function (U) {
  "use strict";
  const { clamp, lerp, smooth } = U.util;
  const $ = (id) => document.getElementById(id);

  const data = window.UNMAZE;
  const params = new URLSearchParams(location.search);
  const filmMode = params.has("film");
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (filmMode) document.body.classList.add("film");

  if (!data || !data.mazes || !data.mazes.length) {
    document.body.insertAdjacentHTML("beforeend",
      '<p style="position:fixed;inset:0;display:grid;place-items:center;color:#9db8d3;font:18px sans-serif">' +
      "No data found. Run: python -m unmaze export</p>");
    return;
  }

  /* ---------- fit the 1920x1080 screen to the window ---------- */
  const screen = $("screen");
  function fit() {
    const k = filmMode ? 1 : Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    screen.style.transform = `scale(${k})`;
  }
  window.addEventListener("resize", fit);
  fit();

  /* ---------- the Overlook carpet, drawn once, behind the HUD ---------- */
  (function carpet() {
    const c = document.createElement("canvas"), w = 84, h = 146;
    c.width = w; c.height = h;
    const g = c.getContext("2d");
    g.fillStyle = "#1a0c08"; g.fillRect(0, 0, w, h);
    const hex = (cx, cy, r, fill, stroke) => {
      g.beginPath();
      for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + (i * Math.PI) / 3; g.lineTo(cx + Math.cos(a) * r, cy + Math.sin(a) * r); }
      g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2; g.stroke(); }
    };
    for (const [cx, cy] of [[w / 2, h / 4], [0, h * 0.75], [w, h * 0.75], [w / 2, h * 1.25 - h], [w / 2, h * 1.25]]) {
      hex(cx, cy, 34, "#3a1409", "#8a3a14");
      hex(cx, cy, 22, "#6b2310", "#c2641f");
      hex(cx, cy, 9, "#2a0e07", null);
    }
    document.documentElement.style.setProperty("--carpet", `url(${c.toDataURL()})`);
  })();

  /* ---------- scene ---------- */
  const scene = new U.Scene($("world"), data);
  const seesCanvas = $("seesCanvas"), guessCanvas = $("guessCanvas"), grain = $("grain");
  const mazes = new Array(data.mazes.length);
  const mazeAt = (i) => (mazes[i] = mazes[i] || U.decodeMaze(data.mazes[i], data));
  let index = 0;

  function loadMaze(i) {
    index = (i + mazes.length) % mazes.length;
    scene.load(mazeAt(index));
    buildDots();
    $("mazeLabel").textContent = `MAZE ${index + 1} / ${mazes.length}`;
  }

  function buildDots() {
    const box = $("mazeDots");
    if (box.childElementCount !== data.mazes.length) {
      box.innerHTML = "";
      data.mazes.forEach((entry, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = entry.solved ? "solved" : "failed";
        b.title = `maze ${i + 1}: ${entry.solved ? "solved" : "not solved"}`;
        b.setAttribute("aria-label", b.title);
        b.addEventListener("click", () => { loadMaze(i); restart(); });
        box.appendChild(b);
      });
    }
    [...box.children].forEach((b, i) => b.classList.toggle("current", i === index));
  }

  /* ---------- HUD ---------- */
  function applyHud(view, verdictT) {
    const m = scene.maze;
    $("sigmaValue").textContent = view.sigma.toFixed(2);
    $("sigmaBar").style.width = `${clamp(view.sigma, 0, 1) * 100}%`;
    $("stepValue").textContent = `STEP ${view.step + 1} / ${m.steps}`;
    $("levelValue").textContent = `LEVEL ${view.level}`;
    scene.drawMonitors(view, seesCanvas, guessCanvas);
    const phase = verdictT !== null && verdictT !== undefined && verdictT >= 0 ? -1 : view.step < 1 ? 0 : view.step < 9 ? 1 : 2;
    document.querySelectorAll("#how li").forEach((li, i) => li.classList.toggle("on", i === phase));

    const box = $("verdict"), word = $("verdictWord"), note = $("verdictNote");
    if (verdictT === null || verdictT === undefined || verdictT < 0) {
      box.className = "verdict";
      word.style.color = ""; word.style.transform = ""; word.innerHTML = "&nbsp;";
      note.textContent = " ";
      return;
    }
    const hit = smooth(verdictT / 0.35);
    box.className = "verdict " + (m.solved ? "solved" : "lost");
    word.textContent = m.solved ? "SOLVED" : "LOST";
    word.style.opacity = hit;
    word.style.transform = `scale(${lerp(1.5, 1, hit)}) translate(${(1 - hit) * 6 * Math.sin(verdictT * 90)}px, 0)`;
    const full = m.solved
      ? `${m.steps} STEPS · 0 REPAIRS · IOU ${m.iou.toFixed(2)}`
      : (m.reason || "not solved").toUpperCase();
    const perChar = Math.min(0.03, 1.0 / full.length); // typed out, but always finished within a second
    note.textContent = full.slice(0, Math.floor(Math.max(0, verdictT - 0.3) / perChar));
  }

  function setGrain(frame, seeded) {
    const g = grain.getContext("2d"), img = g.createImageData(grain.width, grain.height), d = img.data;
    for (let i = 0; i < grain.width * grain.height; i++) {
      const v = 128 + ((seeded ? U.util.hash(i, frame >> 1, 91) : Math.random()) - 0.5) * 150; // film: new grain every other frame
      d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v;
      d[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }

  /* ---------- interactive timeline ---------- */
  const T = { denoise: 14, verdict: 1.4, walk: 7.5, hold: 4 };
  let t = 0, playing = true, speed = 1, last = performance.now(), frame = 0, nextAt = null;

  function total() { return T.denoise + T.verdict + (scene.maze.solved ? T.walk : 0); }
  function restart() { t = 0; playing = true; nextAt = null; $("playBtn").textContent = "PAUSE"; }

  function drawUi() {
    const m = scene.maze, [mcx, mcy] = scene.mazeCentre();
    const u = clamp(t / T.denoise, 0, 1);
    const s = (m.steps - 1) * Math.pow(u, 1.5);
    const wk = clamp((t - T.denoise - T.verdict) / T.walk, 0, 1);
    const walking = m.solved && t > T.denoise + T.verdict;
    const drift = reduced ? 0 : 1;
    const base = scene.camera(mcx + drift * Math.sin(t * 0.21) * 0.3, mcy + drift * Math.cos(t * 0.17) * 0.2, 0.82, 560, 568);
    let cam = base;
    if (walking) {
      const [wx, wy] = scene.walkerPos(smooth(wk));
      const f = scene.camera(wx, wy, 1.4, 560, 540), k = smooth(clamp(wk * 6, 0, 1));
      cam = { cx: lerp(base.cx, f.cx, k), cy: lerp(base.cy, f.cy, k), zoom: lerp(0.82, 1.4, k) };
    }
    const view = scene.render({
      s, cam, t, frame,
      flicker: !reduced,
      thread: 1,
      walk: walking ? smooth(wk) : null,
      dark: walking ? 0.8 * smooth(clamp(wk * 5, 0, 1)) : 0,
      lightRadius: 4.6,
    });
    applyHud(view, t >= T.denoise ? t - T.denoise : null);
    $("scrub").value = String(Math.round(u * 1000));
  }

  function tick(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (playing) {
      t += dt * speed;
      if (t >= total()) {
        t = total();
        if (nextAt === null) nextAt = now + T.hold * 1000;
        if (now >= nextAt) { loadMaze(index + 1); restart(); }
      }
    }
    frame++;
    drawUi();
    if (!reduced && frame % 3 === 0) setGrain(frame, false);
    requestAnimationFrame(tick);
  }

  /* A handle for tests and screenshots: jump to a moment of the interactive timeline. */
  U.debug = { seek(time) { t = clamp(time, 0, total()); playing = false; $("playBtn").textContent = "PLAY"; }, total: () => total() };

  /* ---------- film mode: one frame at a time, on demand ---------- */
  const filmCard = $("card");
  function renderFilmFrame(time, frameNo) {
    const F = U.Film, st = F.at(time, scene, frameNo);
    const view = scene.render(st.showWorld ? st.scene : { ...st.scene, thread: 0, staticGain: 0, dark: 0, walk: null });
    applyHud(view, st.verdict ? st.verdict.t : null);
    $("hud").style.opacity = st.hudAlpha;
    $("title").style.opacity = st.titleAlpha;
    $("world").style.visibility = st.showWorld ? "visible" : "hidden";
    $("fade").style.opacity = st.fade;
    setGrain(frameNo, true);

    if (st.card) {
      filmCard.style.display = "flex";
      filmCard.style.opacity = st.card.alpha;
      if (st.card.kind === "typed") {
        filmCard.innerHTML = `<div class="typed"></div>`;
        filmCard.firstChild.textContent = st.card.text;
      } else if (!filmCard.dataset.end) {
        filmCard.dataset.end = "1";
        filmCard.innerHTML =
          `<div class="big">UNMAZE</div>` +
          `<div class="sub">a diffusion model finds the way out</div>` +
          `<div class="small">${data.mazeSize}&times;${data.mazeSize} hedge maze &middot; ${data.steps} steps &middot; 0 repairs &middot; trained on a CPU</div>` +
          `<div class="small" style="margin-top:14px">github.com/sivaratrisrinivas/unmaze</div>`;
      }
    } else {
      filmCard.style.display = "none";
      delete filmCard.dataset.end;
    }
  }

  function enterFilm() {
    const F = U.Film;
    const hero = params.has("maze") ? Number(params.get("maze")) : F.pickHero(data);
    loadMaze(hero);
    const bars = document.querySelectorAll("#bars i");
    bars.forEach((b) => (b.style.height = F.FILM.bars + "px"));
    window.renderFilmFrame = renderFilmFrame;
    window.FILM_INFO = { ...F.FILM, hero, mazeSize: data.mazeSize, steps: data.steps };
    renderFilmFrame(0, 0);
    document.fonts.ready.then(() => { window.FILM_READY = true; });
  }

  /* ---------- controls ---------- */
  if (!filmMode) {
    $("playBtn").addEventListener("click", () => {
      if (t >= total()) restart(); else { playing = !playing; nextAt = null; }
      $("playBtn").textContent = playing ? "PAUSE" : "PLAY";
    });
    $("scrub").addEventListener("input", (e) => { t = (Number(e.target.value) / 1000) * T.denoise; nextAt = null; });
    $("speed").addEventListener("change", (e) => { speed = Number(e.target.value); });
    $("nextBtn").addEventListener("click", () => { loadMaze(index + 1); restart(); });
    $("prevBtn").addEventListener("click", () => { loadMaze(index - 1); restart(); });
    $("replayBtn").addEventListener("click", restart);
    window.addEventListener("keydown", (e) => {
      if (e.target && /INPUT|SELECT/.test(e.target.tagName) && e.key !== " ") return;
      if (e.key === " ") { e.preventDefault(); $("playBtn").click(); }
      else if (e.key === "n" || e.key === "N") $("nextBtn").click();
      else if (e.key === "p" || e.key === "P") $("prevBtn").click();
      else if (e.key === "r" || e.key === "R") restart();
      else if (e.key === "ArrowRight") { t = Math.min(total(), t + 0.4); nextAt = null; }
      else if (e.key === "ArrowLeft") { t = Math.max(0, t - 0.4); nextAt = null; }
    });
  }

  document.fonts.ready.then(() => {
    if (filmMode) enterFilm();
    else { loadMaze(params.has("maze") ? Number(params.get("maze")) : 0); requestAnimationFrame(tick); }
  });
})((window.Unmaze = window.Unmaze || {}));
