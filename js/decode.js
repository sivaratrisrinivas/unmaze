/* Turn the exported data file back into numbers: the walls, the model's frames, and the order of the solution. */
(function (U) {
  "use strict";

  function unpack(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  /* Solution pixels in walking order, from the entrance (start) to the heart (goal). */
  function orderPath(solution, N, start, goal) {
    const path = [start];
    const seen = new Set([start[0] * N + start[1]]);
    let [r, c] = start;
    while (r !== goal[0] || c !== goal[1]) {
      let moved = false;
      for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nr = r + dr, nc = c + dc;
        if (nr < 0 || nc < 0 || nr >= N || nc >= N) continue;
        if (solution[nr * N + nc] && !seen.has(nr * N + nc)) {
          seen.add(nr * N + nc);
          path.push([nr, nc]);
          [r, c] = [nr, nc];
          moved = true;
          break;
        }
      }
      if (!moved) break;
    }
    return path;
  }

  U.decodeMaze = function (entry, data) {
    const N = data.grid, steps = data.steps, cells = N * N;
    const g8 = unpack(entry.guesses), n8 = unpack(entry.noisy);
    const guesses = new Float32Array(steps * cells), noisy = new Float32Array(steps * cells);
    for (let i = 0; i < steps * cells; i++) {
      guesses[i] = (g8[i] / 255) * 2 - 1;         // 0..255 -> -1..1
      noisy[i] = (((n8[i] << 24) >> 24) / 127) * 3; // int8 -> -3..3
    }
    const walls = new Uint8Array(cells), solution = new Uint8Array(cells);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        walls[r * N + c] = entry.walls[r][c] === "#" ? 1 : 0;
        solution[r * N + c] = entry.solution[r][c] === "." ? 1 : 0;
      }
    }
    return {
      seed: entry.seed, N, steps, walls, solution, guesses, noisy,
      start: entry.start, goal: entry.goal,
      solved: entry.solved, iou: entry.iou, reason: entry.reason,
      path: orderPath(solution, N, entry.start, entry.goal),
    };
  };

  /* The gap in the outer hedge: the border pixel just outside the entrance cell. */
  U.entranceGap = function (maze) {
    const [r, c] = maze.start, last = maze.N - 1;
    if (r === last - 1) return { r: last, c, dr: 1, dc: 0 };
    if (r === 1) return { r: 0, c, dr: -1, dc: 0 };
    if (c === 1) return { r, c: 0, dr: 0, dc: -1 };
    return { r, c: last, dr: 0, dc: 1 };
  };
})((window.Unmaze = window.Unmaze || {}));
