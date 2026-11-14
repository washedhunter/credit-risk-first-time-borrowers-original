"""Take README screenshots of the running stack (API docs + dashboard).

Runs inside the official Playwright image against `docker compose up`:

    docker run --rm --add-host=host.docker.internal:host-gateway -v "${PWD}/docs/img:/out" \
        -v "${PWD}/scripts:/scripts" mcr.microsoft.com/playwright/python:v1.65.2-noble \
        python /scripts/screenshots.py
"""

from __future__ import annotations

import os

from playwright.sync_api import Page, sync_playwright

API = os.environ.get("API", "http://host.docker.internal:8000")
DASH = os.environ.get("DASH", "http://host.docker.internal:8501")
OUT = os.environ.get("OUT", "/out")


def wait_scored(page: Page) -> None:
    page.get_by_text("Probability of default").first.wait_for(timeout=60_000)
    page.wait_for_timeout(2500)  # let plotly finish drawing


def choose(page: Page, label: str, option: str) -> None:
    """Streamlit selectboxes only render visible options, so type to filter, then press Enter."""
    page.get_by_label(label).click()
    page.keyboard.type(option)
    page.get_by_role("option", name=option).first.wait_for()
    page.keyboard.press("Enter")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1)

        page.goto(f"{API}/docs")
        page.get_by_text("/predict/batch").first.wait_for()
        page.wait_for_timeout(1000)
        page.screenshot(path=f"{OUT}/api_docs.png")

        # Streamlit scrolls inside its own container, so use a tall viewport instead of full_page.
        page.set_viewport_size({"width": 1400, "height": 2000})
        page.goto(DASH)
        wait_scored(page)
        choose(page, "Segment", "has a credit-bureau record")
        page.wait_for_timeout(1500)
        wait_scored(page)
        page.screenshot(path=f"{OUT}/dashboard_sample.png")

        choose(page, "Segment", "thin-file (no bureau, no previous applications)")
        page.wait_for_timeout(1500)
        choose(page, "Applicant (synthetic)", "SYN-0031")
        page.wait_for_timeout(1500)
        wait_scored(page)
        page.screenshot(path=f"{OUT}/dashboard_thin_file.png")

        page.get_by_text("New applicant (form)").click()
        page.wait_for_timeout(1500)
        wait_scored(page)
        page.screenshot(path=f"{OUT}/dashboard_form.png")
        browser.close()


if __name__ == "__main__":
    main()
