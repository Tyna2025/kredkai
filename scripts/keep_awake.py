"""Odwiedza aplikację na Streamlit Community Cloud i budzi ją, jeśli zasnęła.

Uruchamiane przez GitHub Actions (.github/workflows/keep_awake.yml).
"""
import os
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "").strip()


def main():
    if not APP_URL:
        print("APP_URL nie jest ustawiony – pomijam (uzupełnij go w keep_awake.yml po wdrożeniu).")
        return 0

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print(f"Otwieram {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(8_000)

        wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        if wake.count() > 0:
            print("Aplikacja spała – budzę.")
            wake.first.click()
            try:
                page.wait_for_selector("text=/zostało/", timeout=240_000)
                print("Aplikacja obudzona.")
            except PlaywrightTimeout:
                print("Kliknięto „obudź”, aplikacja jeszcze się uruchamia.")
        else:
            print("Aplikacja działa – wizyta zarejestrowana.")
            page.wait_for_timeout(20_000)

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
