/* The picture: an aerial night view of a snow-covered hedge maze, and everything the diffusion model does to it.

   Static world (ground, pines, hedges) is painted once per maze. Each frame then lays on top of it, in order:
   the model's guess as a red thread, the snow-static it is looking through, the lamps, and (during the walk) darkness.
   Everything is a pure function of the state passed to render(), so a frame can be reproduced exactly. */
(function (U) {
  "use strict";

  const W = 1920, H = 1080;
  const PX = 56;                 // pre-rendered pixels per cell
  const WW = 72, WH = 34;        // world size, in cells
  const BASE_PPC = 40;           // screen pixels per cell at zoom 1
  const SUB = 6;                 // static grains per grid pixel

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  function rng(seed) {
    let a = seed | 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hash(a, b, c) {
    let t = (Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(c, 1274126177)) | 0;
    t = Math.imul(t ^ (t >>> 13), 1274126177);
    return ((t ^ (t >>> 16)) >>> 0) / 4294967296;
  }
  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }
  function glow(ctx, x, y, r, rgb, a) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${rgb},${a})`);
    g.addColorStop(0.45, `rgba(${rgb},${a * 0.35})`);
    g.addColorStop(1, `rgba(${rgb},0)`);
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
  }

  /* ---------- the painted world ---------- */

  function pine(c, x, y, r, R) {
    const points = 15;
    for (const [scale, light] of [[1, 6], [0.72, 9], [0.42, 13]]) {
      c.beginPath();
      for (let i = 0; i < points * 2; i++) {
        const a = (i / (points * 2)) * Math.PI * 2;
        const rr = r * scale * (i % 2 ? 0.58 : 1) * (0.88 + R() * 0.24);
        c.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
      }
      c.closePath();
      c.fillStyle = `hsl(${150 + R() * 14}, 38%, ${light + R() * 3}%)`;
      c.fill();
    }
    for (let i = 0; i < 16; i++) { // snow on the needles
      c.fillStyle = `rgba(225,238,250,${0.25 + R() * 0.5})`;
      c.beginPath();
      c.arc(x - r * 0.25 + (R() - 0.5) * r * 0.9, y - r * 0.25 + (R() - 0.5) * r * 0.9, 1 + R() * 3.2, 0, 7);
      c.fill();
    }
  }

  function hedge(c, x, y, R) {
    const e = PX * 0.07, r = PX * 0.3;
    c.fillStyle = "#091a12";
    roundRect(c, x - e, y - e, PX + 2 * e, PX + 2 * e, r);
    c.fill();
    for (let i = 0; i < 78; i++) { // leaf clumps, lit from the upper left by the moon
      const px = x - e + R() * (PX + 2 * e), py = y - e + R() * (PX + 2 * e);
      const lit = 1 - ((px - x) + (py - y)) / (2 * PX); // 1 at the upper left
      const hue = 140 + R() * 24 + lit * 14;
      c.fillStyle = `hsl(${hue}, ${34 + R() * 22}%, ${5.5 + R() * 9 + lit * 6}%)`;
      c.beginPath();
      c.arc(px, py, 3 + R() * 8, 0, 7);
      c.fill();
    }
    c.lineWidth = 1.2;
    for (let i = 0; i < 26; i++) { // single leaves catching the light
      const px = x + R() * PX, py = y + R() * PX, a = R() * 6.28, l = 4 + R() * 7;
      c.strokeStyle = `rgba(${120 + R() * 40},${185 + R() * 40},${170 + R() * 50},${0.1 + R() * 0.22})`;
      c.beginPath();
      c.moveTo(px, py);
      c.lineTo(px + Math.cos(a) * l, py + Math.sin(a) * l);
      c.stroke();
    }
    for (let i = 0; i < 11; i++) { // snow settled on top, mostly toward the moonlit side
      const px = x + R() * R() * PX, py = y + R() * R() * PX;
      c.fillStyle = `rgba(228,240,252,${0.18 + R() * 0.42})`;
      c.beginPath();
      c.arc(px, py, 1.2 + R() * 3.2, 0, 7);
      c.fill();
    }
  }

  function buildWorld(S, m) {
    const N = S.N, cv = document.createElement("canvas");
    cv.width = WW * PX;
    cv.height = WH * PX;
    const c = cv.getContext("2d");
    const R = rng(9001 + m.seed * 131);
    const mx = (S.gx0 + N / 2) * PX, my = (S.gy0 + N / 2) * PX;

    const g = c.createRadialGradient(mx, my, N * PX * 0.15, mx, my, WW * PX * 0.5);
    g.addColorStop(0, "#2c4157");
    g.addColorStop(0.4, "#1b2a3a");
    g.addColorStop(1, "#090f17");
    c.fillStyle = g;
    c.fillRect(0, 0, cv.width, cv.height);

    for (let i = 0; i < 30000; i++) { // snow grain
      c.fillStyle = `rgba(205,225,245,${0.04 + R() * 0.09})`;
      const s = 0.5 + R() * 1.6;
      c.fillRect(R() * cv.width, R() * cv.height, s, s);
    }
    for (let i = 0; i < 150; i++) { // soft drifts
      const x = R() * cv.width, y = R() * cv.height, r = (1 + R() * 3.5) * PX;
      const d = c.createRadialGradient(x, y, 0, x, y, r);
      d.addColorStop(0, "rgba(222,236,250,0.07)");
      d.addColorStop(1, "rgba(222,236,250,0)");
      c.fillStyle = d;
      c.fillRect(x - r, y - r, r * 2, r * 2);
    }

    const gap = U.entranceGap(m);
    S.gap = gap;
    const cell = (gx, gy) => [(S.gx0 + gx) * PX, (S.gy0 + gy) * PX];

    // Trampled snow along the paths: a little lighter than the lawn.
    for (let r = 0; r < N; r++) {
      for (let col = 0; col < N; col++) {
        if (!m.walls[r * N + col]) {
          const [x, y] = cell(col, r);
          c.fillStyle = "rgba(190,214,240,0.085)";
          c.fillRect(x, y, PX, PX);
        }
      }
    }
    { // the way in from outside
      const [x, y] = cell(gap.c, gap.r);
      c.fillStyle = "rgba(200,222,245,0.12)";
      for (let i = 0; i < 9; i++) {
        c.fillRect(x + gap.dc * PX * i, y + gap.dr * PX * i, PX, PX);
      }
    }

    // A clearing of lawn, then pines further out.
    const x0 = S.gx0 - 6, x1 = S.gx0 + N + 6, y0 = S.gy0 - 3, y1 = S.gy0 + N + 3;
    const [ex, ey] = cell(gap.c + gap.dc * 4, gap.r + gap.dr * 4);
    for (let i = 0; i < 260; i++) {
      const wx = R() * WW, wy = R() * WH;
      if (wx > x0 && wx < x1 && wy > y0 && wy < y1) continue;
      if (Math.hypot(wx * PX - ex, wy * PX - ey) < 5.5 * PX) continue;
      c.fillStyle = "rgba(0,0,0,0.35)";
      c.beginPath();
      c.ellipse(wx * PX + 14, wy * PX + 18, PX * 1.5, PX * 1.3, 0, 0, 7);
      c.fill();
      pine(c, wx * PX, wy * PX, PX * (1 + R() * 1.2), R);
    }

    // Contact shadows under the hedges, then the hedges.
    c.save();
    c.shadowColor = "rgba(0,0,0,0.7)";
    c.shadowBlur = 22;
    c.shadowOffsetX = 9;
    c.shadowOffsetY = 13;
    c.fillStyle = "#000";
    for (let r = 0; r < N; r++) {
      for (let col = 0; col < N; col++) {
        if (m.walls[r * N + col] && !(r === gap.r && col === gap.c)) {
          const [x, y] = cell(col, r);
          roundRect(c, x, y, PX, PX, PX * 0.3);
          c.fill();
        }
      }
    }
    c.restore();
    for (let r = 0; r < N; r++) {
      for (let col = 0; col < N; col++) {
        if (m.walls[r * N + col] && !(r === gap.r && col === gap.c)) {
          const [x, y] = cell(col, r);
          hedge(c, x, y, R);
        }
      }
    }
    return cv;
  }

  /* ---------- scene ---------- */

  function Scene(canvas, data) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = data;
    this.N = data.grid;
    this.gx0 = (WW - this.N) / 2;
    this.gy0 = (WH - this.N) / 2;
    this.snow = document.createElement("canvas");
    this.snow.width = this.snow.height = this.N * SUB;
    this.snowCtx = this.snow.getContext("2d");
    this.snowImg = this.snowCtx.createImageData(this.N * SUB, this.N * SUB);
    this.dark = document.createElement("canvas");
    this.dark.width = W;
    this.dark.height = H;
    this.maze = null;
  }

  Scene.W = W; Scene.H = H; Scene.BASE_PPC = BASE_PPC;

  Scene.prototype.load = function (maze) {
    this.maze = maze;
    this.world = buildWorld(this, maze);
  };

  /* Centre of a grid pixel, in world cells. */
  Scene.prototype.cellPos = function (r, c) { return [this.gx0 + c + 0.5, this.gy0 + r + 0.5]; };
  Scene.prototype.mazeCentre = function () { return [this.gx0 + this.N / 2, this.gy0 + this.N / 2]; };

  /* A camera that keeps world point (fx, fy) at screen position (sx, sy). */
  Scene.prototype.camera = function (fx, fy, zoom, sx, sy) {
    const ppc = BASE_PPC * zoom;
    return { cx: fx + (W / 2 - sx) / ppc, cy: fy + (H / 2 - (sy === undefined ? H / 2 : sy)) / ppc, zoom };
  };
  Scene.prototype.toScreen = function (cam, wx, wy) {
    const k = BASE_PPC * cam.zoom;
    return [W / 2 + (wx - cam.cx) * k, H / 2 + (wy - cam.cy) * k];
  };

  /* Interpolated view of the model at fractional step s: its guess, its input, and how noisy that input is. */
  Scene.prototype.frameAt = function (s) {
    const m = this.maze, cells = this.N * this.N, d = this.data;
    const i0 = clamp(Math.floor(s), 0, m.steps - 1), i1 = Math.min(i0 + 1, m.steps - 1), f = clamp(s - i0, 0, 1);
    const guess = new Float32Array(cells);
    for (let i = 0; i < cells; i++) guess[i] = lerp(m.guesses[i0 * cells + i], m.guesses[i1 * cells + i], f);
    return {
      i0, guess,
      noisy: m.noisy.subarray(i0 * cells, (i0 + 1) * cells),
      sigma: lerp(d.sigma[i0], d.sigma[i1], f),
      level: Math.round(lerp(d.levels[i0], d.levels[i1], f)),
      step: Math.round(s),
    };
  };

  /* st: {s, cam, t, frame, flicker, walk (0..1 or null), dark (0..1), lightRadius (cells), thread (0..1 opacity)} */
  Scene.prototype.render = function (st) {
    const ctx = this.ctx, m = this.maze, N = this.N, gx0 = this.gx0, gy0 = this.gy0;
    const view = this.frameAt(st.s);
    const k = (BASE_PPC * st.cam.zoom) / PX;
    const flick = st.flicker ? 1 + 0.07 * Math.sin(st.t * 7.3) + 0.05 * Math.sin(st.t * 13.1 + 1.7) + 0.04 * Math.sin(st.t * 3.1) : 1;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#090f17";
    ctx.fillRect(0, 0, W, H);

    ctx.setTransform(k, 0, 0, k, W / 2 - st.cam.cx * PX * k, H / 2 - st.cam.cy * PX * k);
    ctx.drawImage(this.world, 0, 0);

    // The red thread: the model's current guess, thicker where it is surer.
    const threadAlpha = st.thread === undefined ? 1 : st.thread;
    if (threadAlpha > 0.002) {
      const p = (r, c) => (view.guess[r * N + c] + 1) / 2;
      ctx.lineCap = "round";
      ctx.shadowColor = "rgba(255,40,30,0.95)";
      for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
          const a = p(r, c);
          if (a < 0.06) continue;
          const wall = m.walls[r * N + c];
          const x = (gx0 + c + 0.5) * PX, y = (gy0 + r + 0.5) * PX;
          const alpha = a * (wall ? 0.4 : 1) * threadAlpha;
          ctx.shadowBlur = wall || a < 0.5 ? 0 : 22;
          ctx.strokeStyle = `rgba(${lerp(120, 222, a) | 0},${lerp(16, 24, a) | 0},${lerp(22, 30, a) | 0},${alpha})`;
          ctx.fillStyle = ctx.strokeStyle;
          ctx.lineWidth = PX * (0.1 + 0.26 * a);
          ctx.beginPath();
          ctx.arc(x, y, ctx.lineWidth / 2, 0, 7);
          ctx.fill();
          for (const [dr, dc] of [[0, 1], [1, 0]]) { // join to the right and below neighbours
            const rr = r + dr, cc = c + dc;
            if (rr >= N || cc >= N) continue;
            const b = Math.min(a, p(rr, cc));
            if (b < 0.06 || (wall && m.walls[rr * N + cc])) continue;
            ctx.lineWidth = PX * (0.1 + 0.26 * b);
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + dc * PX, y + dr * PX);
            ctx.stroke();
          }
        }
      }
      ctx.shadowBlur = 0;
    }

    // The snow the model is looking through: its noisy input, drawn as grain; thicker the noisier it is.
    const A = 0.96 * Math.pow(clamp(view.sigma, 0, 1), 1.3) * (st.staticGain === undefined ? 1 : st.staticGain);
    if (A > 0.004) {
      const M = N * SUB, d = this.snowImg.data;
      for (let j = 0; j < M; j++) {
        for (let i = 0; i < M; i++) {
          const x = view.noisy[((j / SUB) | 0) * N + ((i / SUB) | 0)];
          const r1 = hash(i, j, st.frame), r2 = hash(j, i, st.frame + 7919);
          const v = clamp(0.5 + 0.17 * clamp(x / 3, -1, 1) + (r1 - 0.5) * 0.95, 0, 1) * 255;
          const o = (j * M + i) * 4;
          d[o] = v * 0.93;
          d[o + 1] = v * 0.98;
          d[o + 2] = Math.min(255, v * 1.07);
          d[o + 3] = A * 255 * (0.7 + 0.3 * r2);
        }
      }
      this.snowCtx.putImageData(this.snowImg, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.snow, gx0 * PX, gy0 * PX, N * PX, N * PX);
      ctx.imageSmoothingEnabled = true;
    }

    // Lamps at the heart and at the entrance.
    const [hx, hy] = this.cellPos(m.goal[0], m.goal[1]);
    const gapPos = this.cellPos(this.gap.r, this.gap.c);
    ctx.globalCompositeOperation = "lighter";
    glow(ctx, hx * PX, hy * PX, PX * 3.6, "233,164,76", 0.5 * flick);
    glow(ctx, gapPos[0] * PX, gapPos[1] * PX, PX * 2.6, "233,164,76", 0.38 * flick);
    ctx.globalCompositeOperation = "source-over";
    ctx.lineWidth = PX * 0.07;
    ctx.strokeStyle = `rgba(247,196,112,${0.9 * Math.min(1, flick)})`;
    ctx.beginPath();
    ctx.arc(hx * PX, hy * PX, PX * 0.36, 0, 7);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,226,160,0.95)";
    ctx.beginPath();
    ctx.arc(hx * PX, hy * PX, PX * 0.12, 0, 7);
    ctx.fill();

    // The walk out: darkness, with a lantern.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (st.dark > 0.002) {
      const d = this.dark.getContext("2d");
      d.globalCompositeOperation = "source-over";
      d.clearRect(0, 0, W, H);
      d.fillStyle = `rgba(2,4,9,${st.dark})`;
      d.fillRect(0, 0, W, H);
      d.globalCompositeOperation = "destination-out";
      const hole = (x, y, R, a) => {
        const g = d.createRadialGradient(x, y, R * 0.04, x, y, R);
        g.addColorStop(0, `rgba(0,0,0,${a})`);
        g.addColorStop(0.55, `rgba(0,0,0,${a * 0.6})`);
        g.addColorStop(1, "rgba(0,0,0,0)");
        d.fillStyle = g;
        d.fillRect(x - R, y - R, R * 2, R * 2);
      };
      const ppc = BASE_PPC * st.cam.zoom;
      const [hsx, hsy] = this.toScreen(st.cam, hx, hy);
      hole(hsx, hsy, ppc * 2.4, 0.55);
      const [gsx, gsy] = this.toScreen(st.cam, gapPos[0], gapPos[1]);
      hole(gsx, gsy, ppc * 2.2, 0.5);
      if (st.walk !== null && st.walk !== undefined) {
        const [wx, wy] = this.walkerPos(st.walk);
        const [sx, sy] = this.toScreen(st.cam, wx, wy);
        hole(sx, sy, ppc * (st.lightRadius || 5), 1);
      }
      ctx.drawImage(this.dark, 0, 0);
      if (st.walk !== null && st.walk !== undefined) {
        const [wx, wy] = this.walkerPos(st.walk);
        const [sx, sy] = this.toScreen(st.cam, wx, wy);
        ctx.globalCompositeOperation = "lighter";
        glow(ctx, sx, sy, ppc * 3.2 * flick, "255,190,100", 0.32);
        glow(ctx, sx, sy, ppc * 0.9, "255,235,190", 0.55);
        ctx.globalCompositeOperation = "source-over";
      }
    }

    // A cold grade over everything.
    ctx.globalCompositeOperation = "soft-light";
    ctx.fillStyle = "rgba(46,86,150,0.22)";
    ctx.fillRect(0, 0, W, H);
    ctx.globalCompositeOperation = "source-over";
    return view;
  };

  /* Where the lantern is, in world cells, a fraction u of the way along the walk from the heart to the entrance. */
  Scene.prototype.walkerPos = function (u) {
    const path = this.maze.path, last = path.length - 1;
    const f = clamp(u, 0, 1) * last;
    const i = Math.min(last - 1, Math.floor(f)), t = f - i;
    // The stored path runs entrance -> heart; the walk goes the other way.
    const a = path[last - i], b = path[last - i - 1] || a;
    return [this.gx0 + lerp(a[1], b[1], t) + 0.5, this.gy0 + lerp(a[0], b[0], t) + 0.5];
  };

  /* The two little monitors: the noisy mask the model is shown, and the guess it makes of the clean mask. */
  Scene.prototype.drawMonitors = function (view, seesCanvas, guessCanvas) {
    const N = this.N, m = this.maze;
    for (const cv of [seesCanvas, guessCanvas]) { if (cv.width !== N) { cv.width = N; cv.height = N; } }
    const a = seesCanvas.getContext("2d").createImageData(N, N), b = guessCanvas.getContext("2d").createImageData(N, N);
    for (let i = 0; i < N * N; i++) {
      const wall = m.walls[i];
      const x = clamp(view.noisy[i] / 3, -1, 1) * 0.5 + 0.5;        // the noisy input
      const v = (wall ? 0.55 * x + 0.04 : x) * 255;
      a.data.set([v * 0.86, v * 0.95, Math.min(255, v * 1.06), 255], i * 4);
      const p = (view.guess[i] + 1) / 2;                            // the clean guess
      const base = wall ? [20, 30, 40] : [5, 7, 9];
      b.data.set([base[0] + p * 205, base[1] + p * 12, base[2] + p * 24, 255], i * 4);
    }
    seesCanvas.getContext("2d").putImageData(a, 0, 0);
    guessCanvas.getContext("2d").putImageData(b, 0, 0);
  };

  U.Scene = Scene;
  const smooth = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  U.util = { clamp, lerp, smooth, hash, rng };
})((window.Unmaze = window.Unmaze || {}));
