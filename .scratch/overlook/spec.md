Status: ready-for-agent

# Overlook: hedge mazes, a UI that shows the diffusion, and a demo film

## Problem Statement

The project solves plain random-endpoint mazes and shows them as small flat pictures. I want it to be about the Overlook hotel's hedge maze, the one from Kubrick's *The Shining*: a big square hedge maze entered from outside, with a heart at the centre. And I want to *see* the diffusion happen, in a browser, with the cold, eerie look of that film, well enough to record a 30-second clip and post it.

## Solution

- Every puzzle becomes an **Overlook Maze**: odd-sized, entered at a random cell on the outer ring, with the **Heart** at the centre. The model is retrained at 11×11 cells on exactly that.
- A static web page (HTML, CSS, JavaScript, no build step) replays the *real* model's denoising frame by frame: the snow-static the model starts from, its guess at the path firming up as red thread through the hedges, and the strict verdict at the end.
- The page has a scripted "film" mode that is a pure function of time, so a 30-second 1080p video can be rendered frame by frame and encoded, with the same look as the page.

## User Stories

1. As a viewer, I want to see an aerial view of a snow-dusted hedge maze at night, so that it feels like the Overlook.
2. As a viewer, I want the maze to look like thick hedges with a path between them, so that it reads as a hedge maze and not a wall diagram.
3. As a viewer, I want to see an entrance in the outer hedge and a marked heart at the centre, so that I know what is being solved.
4. As a viewer, I want to watch the model start from pure static (snow) and the path emerge out of it, so that I understand what a diffusion model does.
5. As a viewer, I want to see both what the model is looking at (the noisy mask) and what it currently believes (its clean guess), so that the denoising is not a black box.
6. As a viewer, I want a noise meter and a step counter, so that I know how far into the denoising it is.
7. As a viewer, I want the final verdict (solved or not, how many steps) shown in the page, so that the claim is checkable.
8. As a viewer, I want a play/pause button, a scrubber, and a speed control, so that I can study any moment.
9. As a viewer, I want a button (and a key) for the next maze, so that I can see it solve many different ones.
10. As a viewer, I want some of the shown mazes to be ones the model gets wrong, if there are any, so that the display is honest.
11. As a viewer, I want the page to work by double-clicking `index.html`, so that nothing needs installing or serving.
12. As a viewer, I want a film mode with cards, a slow camera move and a title, so that the page can be recorded as a cinematic clip.
13. As a viewer, I want film grain, a vignette, letterbox bars and a faint light flicker, so that it feels like 35mm.
14. As a viewer, I want a cold blue night palette with one warm hotel-amber accent and the path in blood red, so that the look is unmistakable and consistent.
15. As a viewer, I want the page to respect reduced-motion preferences, so that the flicker and drift do not hurt anyone.
16. As a maintainer, I want an `export` command that runs the real checkpoint and writes the page's data file, so that what the UI shows is what the model did.
17. As a maintainer, I want the exported frames to decode back to the model's real output within quantisation error, so that the UI cannot drift from the model.
18. As a maintainer, I want the exported verdicts to come from the same strict judge as `eval`, so that there is one definition of solved.
19. As a maintainer, I want generated puzzles to be deterministic from a seed and uniformly random among perfect mazes, so that results are repeatable and unbiased.
20. As a maintainer, I want asking for an even-sized Overlook Maze to be a clear error, so that nobody gets an off-centre heart by accident.
21. As a maker of social posts, I want a script that renders a 30-second 1080p video from the page deterministically, so that the video is reproducible and not a shaky screen recording.
22. As a maker of social posts, I want the film to open and close on text cards and to end on the project name, so that the clip stands alone.
23. As a maker of social posts, I want original ambient sound, so that I am not using anyone's copyrighted music.

## Implementation Decisions

- **Overlook Maze generator** replaces the random-endpoint one: odd n at least 3, Entrance uniformly random over the outer-ring cells, Heart is the centre cell, maze uniformly random among perfect mazes (Wilson's, unchanged). See ADR 0003.
- **Border stays solid in the Grid.** The gap in the outer hedge at the Entrance is drawn by the UI. This keeps the judge's guarantee that no path pixel is ever on the border.
- **Replay seam on the Solver:** a method returning, for each denoising step, the noise level, the noisy mask the model saw, and its clean guess.
- **Export module:** runs a checkpoint over chosen held-out seeds with the strict judge and writes one JavaScript data file (not JSON, so the page works from `file://`). Frames are quantised to 8 bits and base64-packed. The export records verdicts and does not hide failures.
- **Web page:** plain HTML, CSS and JavaScript, no dependencies, no network. Canvas 2D for the maze, procedural hedge texture, snow, static, grain, vignette. The playback state is a pure function of time and of the chosen maze, so film mode can be rendered offline frame by frame.
- **Film:** a scripted 30-second timeline inside the page, rendered by driving headless Chromium to each frame time and screenshotting, then encoded with ffmpeg. Audio is synthesised (original) and muxed in.
- **Model:** same U-Net, retrained on 11×11 Overlook mazes, CPU only.
- **No assets from the film.** The look is an homage built from colour, light, geometry and type. No stills, logos, music or recreations of the film's title cards.

## Testing Decisions

Same rules as the first spec: test behaviour at pre-agreed seams, with expected values from hand-drawn or independent sources; the Denoiser is the only injected thing.

**Seams:**
1. **The maze world seam** (`generate` now makes Overlook Mazes): tests for heart at centre, entrance on the outer ring and covering all four sides, odd-n error, determinism, perfectness, uniformity.
2. **The Solver seam** (`replay`): one step per level, noisy frame starts as unit-variance noise, last guess equals the Attempt.
3. **The export seam** (`export`): the file decodes back to the model's own frames within quantisation tolerance, and its verdicts equal `judge`'s.
4. **The CLI seam** (slow): the shipped checkpoint's solve rate on held-out Overlook Mazes.

The browser page is checked by loading it in headless Chromium: it must render without console errors, and screenshots at known times must show the expected stages. It has no unit tests.

## Out of Scope

- Reproducing the real film maze's blueprint, or any film footage, stills, title typography or music.
- Live in-browser inference (the page replays recorded frames from the real model).
- GPU training.
- Mazes with loops, or multiple entrances.
- Mobile layout polish beyond not breaking.

## Further Notes

- A path has no direction, so the film can narrate the Solution either as the way in or as the way out of the heart.
- Vocabulary is in `GLOSSARY.md`; the replacement of random endpoints is ADR 0003.
