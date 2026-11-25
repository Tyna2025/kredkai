# 🖍️ KredkAI

**👉 Działająca aplikacja: https://kredkai.streamlit.app**

Aplikacja Streamlit, która tworzy kolorowanki do wydruku: z opisu tekstowego albo z wgranego zdjęcia, na trzech poziomach trudności.

Projekt z kursu „Od zera do AI”, moduł 8 (`#projekt-app_coloring_books`).

## Co potrafi

- **✏️ Z tekstu:** wpisujesz motyw, wybierasz poziom i liczbę obrazków (1–3). Model tekstowy wymyśla pomysły, które możesz poprawić, a potem model obrazowy rysuje kolorowanki w PNG.
- **📷 Ze zdjęcia:** wgrywasz zdjęcie i dostajesz kontury do kolorowania na wybranym poziomie.
- **🗂️ Moja galeria:** wszystkie kolorowanki z sesji, pobieranie jako ZIP, zapis sesji do pliku i ponowne wczytanie.

## Zmiany względem specyfikacji

| Specyfikacja | Ta aplikacja |
|---|---|
| Motyw → lista pomysłów → kolorowanki PNG | ✅ jak w specyfikacji, pomysły można edytować przed rysowaniem |
| Wybór liczby rysunków | ✅ 1–3 na raz |
| Zapisywanie i wczytywanie sesji | ✅ „Zapisz sesję” tworzy ZIP (PNG + `session.json`), „Wczytaj sesję” go odtwarza |
| — | ➕ **3 poziomy trudności:** łatwy (grube kontury, proste kształty), średni (więcej detali), zaawansowany (ozdobne wzory w stylu zentangle, dużo detali) |
| — | ➕ **Kolorowanka ze zdjęcia:** kontury na podstawie wgranego zdjęcia |
| `gpt-4o` + `dall-e-3` | 🔁 `gpt-4o-mini` (pomysły) + `gpt-image-2` (obrazy); DALL·E 3 został wycofany z API OpenAI |
| — | ➕ **Limit 15 kolorowanek dziennie** dla wszystkich użytkowników razem (ochrona kosztów) |
| — | ➕ Strona startowa z ilustracją „kredkowej krainy”, podglądy poziomów (pół pokolorowane, pół kontur), jasny wygląd z kolorowymi napisami |

## Uruchomienie lokalnie

```bash
conda create -n kolorowanki python=3.12
conda activate kolorowanki
pip install -r requirements.txt
```

Skopiuj `.env.example` jako `.env` i wpisz swój klucz `OPENAI_API_KEY`, a potem uruchom:

```bash
streamlit run app.py
```

## Wdrożenie (Streamlit Community Cloud)

- Plik główny: `app.py`.
- W Secrets aplikacji: `OPENAI_API_KEY = "sk-..."`.
- `.github/workflows/keep_awake.yml` co 6 godzin odwiedza aplikację, żeby nie zasypiała. Po wdrożeniu wpisz jej adres w `APP_URL`.

## Konfiguracja

Na górze `app.py`: `APP_NAME`, `TEXT_MODEL`, `IMAGE_MODEL`, `DAILY_LIMIT`, `MAX_PER_BATCH` oraz jakość obrazów dla każdego poziomu (`LEVELS[...]["quality"]`).
