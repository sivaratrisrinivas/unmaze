"""The web page, driven in headless Chromium: it loads cleanly, shows the model's verdict, and the film is a
pure function of time (so the video is reproducible).

Slow (needs a browser): run with `pytest -m slow`. Skipped when Playwright or Chromium is not installed.
"""

import json
from pathlib import Path

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.slow

WEB = Path(__file__).resolve().parent.parent / "web"
CHROME = next(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"), None)


def exported_data() -> dict:
    text = (WEB / "data" / "overlook.js").read_text()
    return json.loads(text[len("window.UNMAZE = "):].rstrip().rstrip(";"))


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        launch = {"args": ["--no-sandbox", "--disable-gpu"]}
        if CHROME:
            launch["executable_path"] = str(CHROME)
        try:
            b = p.chromium.launch(**launch)
        except Exception as error:  # no browser available here
            pytest.skip(f"no Chromium to drive: {error}")
        yield b
        b.close()


def open_page(browser, query: str):
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    problems = []
    page.on("pageerror", lambda e: problems.append(str(e)))
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.goto((WEB / "index.html").as_uri() + query)
    page.wait_for_function("document.fonts.status === 'loaded'")
    return page, problems


def first_solved() -> int:
    return next(i for i, m in enumerate(exported_data()["mazes"]) if m["solved"])


def test_the_page_loads_without_errors_and_ends_a_solved_maze_on_the_verdict(browser):
    page, problems = open_page(browser, f"?maze={first_solved()}")
    page.wait_for_timeout(500)
    page.evaluate("Unmaze.debug.seek(Unmaze.debug.total() - 3)")  # into the walk out, after the verdict
    page.wait_for_timeout(400)

    assert problems == []
    assert page.inner_text("#verdictWord") == "SOLVED"
    assert page.inner_text("#stepValue").endswith(f"/ {exported_data()['steps']}")
    luminance = page.evaluate("""() => {
      const c = document.getElementById('world'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let sum = 0, sum2 = 0, n = 0;
      for (let i = 0; i < d.length; i += 4 * 97) { const v = d[i] + d[i + 1] + d[i + 2]; sum += v; sum2 += v * v; n++; }
      return { mean: sum / n, sd: Math.sqrt(sum2 / n - (sum / n) ** 2) };
    }""")
    assert luminance["mean"] > 15 and luminance["sd"] > 15  # something is drawn: not a blank or flat canvas
    page.close()


def test_a_maze_the_model_got_wrong_is_labelled_lost_with_the_judges_reason(browser):
    # Whatever the shipped data holds, show the page a failure: the first maze, as the judge would report a miss.
    reason = "there is a gap between the start and the goal"
    data = exported_data()
    data["mazes"][0].update(solved=False, iou=0.82, reason=reason)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    problems = []
    page.on("pageerror", lambda e: problems.append(str(e)))
    page.route("**/data/overlook.js", lambda route: route.fulfill(
        body="window.UNMAZE = " + json.dumps(data) + ";", content_type="text/javascript"))
    page.goto((WEB / "index.html").as_uri() + "?maze=0")
    page.wait_for_function("document.fonts.status === 'loaded'")
    page.evaluate("Unmaze.debug.seek(Unmaze.debug.total())")
    page.wait_for_timeout(600)

    assert problems == []
    assert page.inner_text("#verdictWord") == "LOST"
    assert page.inner_text("#verdictNote").strip().lower() == reason
    assert "failed" in page.get_attribute("#mazeDots button:nth-child(1)", "class")
    page.close()


def test_the_film_opens_on_words_ends_on_the_title_and_every_frame_is_a_pure_function_of_time(browser):
    page, problems = open_page(browser, "?film=1")
    page.wait_for_function("window.FILM_READY === true")

    page.evaluate("renderFilmFrame(2.6, 78)")
    opening = page.inner_text("#card .typed")
    page.evaluate("renderFilmFrame(20.8, 624)")
    verdict = page.inner_text("#verdictWord")
    page.evaluate("renderFilmFrame(28.8, 864)")
    closing = page.inner_text("#card .big")

    page.evaluate("renderFilmFrame(12.0, 360)")
    first = page.screenshot()
    page.evaluate("renderFilmFrame(25.0, 750)")  # go somewhere else in between
    page.evaluate("renderFilmFrame(12.0, 360)")
    second = page.screenshot()

    assert problems == []
    assert "NO MAP.\nNO MEMORY.\nONLY NOISE.".startswith(opening.rstrip("\u258c "))  # typed so far
    assert opening.startswith("NO MAP.\nNO MEM")
    assert verdict == "SOLVED"
    assert closing == "UNMAZE"
    assert first == second
    page.close()
