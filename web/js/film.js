/* The 30-second film: a pure function from time to what is on screen.
   Nothing here depends on the wall clock, so any frame can be rendered exactly, in any order. */
(function (U) {
  "use strict";
  const { clamp, lerp, smooth } = U.util;
  const ramp = (t, a, b) => clamp((t - a) / (b - a), 0, 1);

  const FILM = {
    duration: 30,
    fps: 30,
    lines: ["NO MAP.", "NO MEMORY.", "ONLY NOISE."],
    lineStart: [0.9, 1.9, 2.9],  // when each line starts typing
    perChar: 0.085,              // seconds per typed character
    typedOut: [4.3, 4.9],        // the words fade out
    reveal: [4.9, 6.6],          // the maze fades up out of black
    hudIn: [6.6, 7.6],
    denoise: [7.0, 20.0],
    verdictAt: 20.0,
    walk: [21.8, 27.0],
    darkIn: [21.6, 23.0],
    fadeOut: [26.8, 27.5],
    endCard: [27.7, 30.0],
    bars: 138,                   // 2.39:1 letterbox on a 1920x1080 frame
  };

  /* The solved maze with the longest winding route: the most to watch. */
  function pickHero(data) {
    let best = -1, bestLen = -1;
    data.mazes.forEach((entry, i) => {
      if (!entry.solved) return;
      const len = entry.solution.join("").split(".").length;
      if (len > bestLen) { best = i; bestLen = len; }
    });
    return Math.max(best, 0);
  }

  /* Typed text at time t: the lines typed so far, and a blinking cursor. */
  function typed(t) {
    let out = [], done = true;
    FILM.lines.forEach((line, k) => {
      const n = Math.floor((t - FILM.lineStart[k]) / FILM.perChar);
      if (n <= 0) { done = false; return; }
      out.push(line.slice(0, Math.min(n, line.length)));
      if (n < line.length) done = false;
    });
    const cursor = Math.floor(t * 2.2) % 2 === 0 ? "▌" : " ";
    return out.join("\n") + (done ? "" : cursor);
  }

  /* The state of the whole frame at time t (seconds). S is the Scene, with the hero maze loaded. */
  function at(t, S, frame) {
    const m = S.maze, steps = m.steps;
    const [d0, d1] = FILM.denoise;
    const u = ramp(t, d0, d1);
    const s = (steps - 1) * Math.pow(u, 1.5);          // dwell on the early steps, where the path appears
    const [mcx, mcy] = S.mazeCentre();

    // Camera: an aerial establishing shot, a slow push in, then following the lantern.
    const walkU = ramp(t, FILM.walk[0], FILM.walk[1]);
    const w = smooth(walkU);
    let cam;
    const push = smooth(ramp(t, FILM.reveal[0], d1));
    const aerial = S.camera(mcx + Math.sin(t * 0.21) * 0.35, mcy + Math.cos(t * 0.17) * 0.25, lerp(0.7, 0.86, push), 640);
    if (t < FILM.darkIn[0]) {
      cam = aerial;
    } else {
      const [wx, wy] = S.walkerPos(w);
      const follow = smooth(ramp(t, FILM.darkIn[0], FILM.darkIn[1] + 0.8));
      const f = S.camera(wx, wy, 1.5, 700);
      cam = { cx: lerp(aerial.cx, f.cx, follow), cy: lerp(aerial.cy, f.cy, follow), zoom: lerp(aerial.zoom, f.zoom, follow) };
    }

    const scene = {
      s, cam, t, frame,
      flicker: true,
      thread: 1,
      walk: t >= FILM.darkIn[0] ? w : null,
      dark: 0.82 * smooth(ramp(t, FILM.darkIn[0], FILM.darkIn[1])),
      lightRadius: 4.6,
    };

    const fade = Math.max(1 - smooth(ramp(t, FILM.reveal[0], FILM.reveal[1])) , smooth(ramp(t, FILM.fadeOut[0], FILM.fadeOut[1])));
    const showWorld = t >= FILM.reveal[0] - 0.05 && t < FILM.endCard[0] - 0.1;

    // The words on black, then the end card.
    let card = null;
    if (t < FILM.typedOut[1]) {
      card = { kind: "typed", text: typed(t), alpha: 1 - ramp(t, FILM.typedOut[0], FILM.typedOut[1]) };
    } else if (t >= FILM.endCard[0]) {
      card = { kind: "end", alpha: smooth(ramp(t, FILM.endCard[0] + 0.1, FILM.endCard[0] + 1.1)) };
    }

    const hudAlpha = smooth(ramp(t, FILM.hudIn[0], FILM.hudIn[1])) * (1 - smooth(ramp(t, FILM.fadeOut[0] - 0.3, FILM.fadeOut[1])));
    const verdictT = t - FILM.verdictAt;
    return {
      scene, showWorld, fade, card, hudAlpha,
      titleAlpha: smooth(ramp(t, 5.4, 6.4)) * (1 - smooth(ramp(t, FILM.fadeOut[0] - 0.3, FILM.fadeOut[1]))),
      verdict: verdictT >= 0 ? { t: verdictT } : null,
    };
  }

  U.Film = { FILM, at, pickHero, typed };
})((window.Unmaze = window.Unmaze || {}));
