import base64
import json
import re
import threading
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import md5
from io import BytesIO
from pathlib import Path

import instructor
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI, OpenAIError
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

#
# KONFIGURACJA
#
APP_NAME = "KredkAI"
APP_SLOGAN = "Twoja wyobraźnia, nasze kontury!"

TEXT_MODEL = "gpt-4o-mini"      # pomysły na kolorowanki (tani, szybki)
IMAGE_MODEL = "gpt-image-2"     # generowanie obrazków
IMAGE_SIZE = "1024x1536"        # pion – pod wydruk A4

DAILY_LIMIT = 15                # ile kolorowanek dziennie dla WSZYSTKICH użytkowników razem
MAX_PER_BATCH = 3               # ile kolorowanek naraz z jednego motywu

TILE_PREFIXES = ("idea", "text", "photo", "gallery")

APP_DIR = Path(__file__).parent
USAGE_FILE = APP_DIR / "usage.json"
ASSETS_DIR = APP_DIR / "assets"

# Nasycone kolory kredek z grafiki tła – do kolorowych napisów i ramek kafelków
PENCIL_COLORS = ["#E23B3B", "#F07C1A", "#E0A800", "#2E9E4F", "#1E88D6", "#7B4FD6", "#D9418C"]

IDEA_COLOR = "#2E9E4F"      # zielona ramka kafelków z pomysłami (tekst do poprawienia)

TEXT_COLOR = "#1E3A6E"      # granat – czytelny na jasnym szkle
MUTED_TEXT = "#55658A"
ACCENT = "#2F6BD8"          # niebieski przycisków

LEVELS = {
    "easy": {
        "label": "Łatwy",
        "emoji": "🐣",
        "color": "#1E90FF",
        "text": "#1479E6",
        "desc": "Grube kontury, duże kształty",
        "quality": "low",
        "idea_hint": (
            "Very simple scene for a toddler: ONE main subject, cute and friendly, "
            "at most 1-2 tiny extra elements, no busy background."
        ),
        "style": (
            "Difficulty: VERY EASY, for toddlers aged 3-5. Very thick, bold, uniform outlines. "
            "Only a few big, simple, rounded shapes (about 10-20 large areas to color). "
            "No small details, no patterns, no texture, plain empty background or a single simple ground line."
        ),
    },
    "medium": {
        "label": "Średni",
        "emoji": "🦊",
        "color": "#8B5CF6",
        "text": "#7C3AED",
        "desc": "Więcej detali, proste tło",
        "quality": "medium",
        "idea_hint": (
            "Scene for a school-age child: a main subject plus a few supporting elements "
            "and a simple background (e.g. trees, clouds, waves)."
        ),
        "style": (
            "Difficulty: MEDIUM, for children aged 6-10. Medium-weight clean outlines. "
            "A moderate amount of detail (about 40-80 areas to color), simple background elements, "
            "some simple decorative details but no dense patterns."
        ),
    },
    "advanced": {
        "label": "Zaawansowany",
        "emoji": "🐢",
        "color": "#EC4899",
        "text": "#DB2777",
        "desc": "Ozdobne wzory, dużo detali",
        "quality": "medium",
        "idea_hint": (
            "Rich, full decorative scene for teenagers and adults: main subject surrounded by many decorative "
            "background elements (leaves, flowers, waves, clouds, small motifs) filling the whole page."
        ),
        "style": (
            "Difficulty: ADVANCED, for teenagers and adults. Thin, clean, clearly readable lines. "
            "A full, richly detailed page: decorative zentangle-style patterns (scales, swirls, petals, paisley, "
            "ornaments) inside the main subject AND inside the background elements; the background is filled "
            "edge to edge with many decorative elements (leaves, flowers, waves, clouds, small motifs), only small "
            "gaps between them. About 250-400 areas to color; every area must still be big enough to color with "
            "a sharp colored pencil. NO stippling, NO tiny dots, NO dense hatching, NO texture noise."
        ),
    },
}

BASE_STYLE = (
    "A printable coloring book page. Pure black line art on a pure white background. "
    "No color, no gray, no shading, no gradients, no filled black areas, no text, no letters, "
    "no signature, no watermark, no frame or border. All shapes are closed outlines that can be colored in."
)


#
# KLUCZ API
#
def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass  # lokalnie zwykle nie ma pliku secrets.toml
    return dotenv_values(APP_DIR / ".env").get("OPENAI_API_KEY")


@st.cache_resource
def get_openai_client(api_key):
    return OpenAI(api_key=api_key)


#
# LIMIT DZIENNY (wspólny dla wszystkich sesji)
#
class UsageCounter:
    def __init__(self, path, limit):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()

    @staticmethod
    def _today():
        return datetime.now(timezone.utc).date().isoformat()

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, count):
        self.path.write_text(json.dumps({self._today(): count}), encoding="utf-8")

    def used_today(self):
        with self.lock:
            return self._load().get(self._today(), 0)

    def remaining(self):
        return max(self.limit - self.used_today(), 0)

    def try_reserve(self):
        with self.lock:
            used = self._load().get(self._today(), 0)
            if used >= self.limit:
                return False
            self._save(used + 1)
            return True

    def release(self):
        with self.lock:
            used = self._load().get(self._today(), 0)
            self._save(max(used - 1, 0))


@st.cache_resource
def _usage_counter(limit):
    return UsageCounter(USAGE_FILE, limit)


def get_usage_counter():
    # limit jest częścią klucza pamięci podręcznej – zmiana DAILY_LIMIT działa od razu
    return _usage_counter(DAILY_LIMIT)


#
# POMYSŁY (text-to-text)
#
class ColoringIdea(BaseModel):
    title: str = Field(description="Krótki, wesoły tytuł kolorowanki po polsku (2-5 słów), poprawny gramatycznie")
    prompt: str = Field(
        description="Opis sceny po polsku (1-3 zdania): co jest na obrazku, kompozycja, poza postaci. "
        "NIE opisuj stylu rysowania, kolorów ani poziomu trudności."
    )


class ColoringIdeas(BaseModel):
    ideas: list[ColoringIdea]


def generate_ideas(client, theme, count, level):
    ideas = []
    for _ in range(3):  # model czasem zwraca mniej pomysłów – dopytujemy o brakujące
        ideas += _ask_for_ideas(client, theme, count - len(ideas), level)
        if len(ideas) >= count:
            break
    return ideas[:count]


def _ask_for_ideas(client, theme, count, level):
    llm = instructor.from_openai(client)
    result = llm.chat.completions.create(
        model=TEXT_MODEL,
        response_model=ColoringIdeas,
        temperature=0.9,
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś ilustratorem kolorowanek. Na podstawie motywu podanego przez użytkownika "
                    f"wymyśl dokładnie {count} różnych pomysłów na kolorowanki. Każdy pomysł ma być inny "
                    "(inna scena, ujęcie lub postać), ale zgodny z motywem. Pisz wyłącznie po polsku. "
                    f"Poziom trudności: {level['idea_hint']}"
                ),
            },
            {"role": "user", "content": theme},
        ],
    )
    return [idea.model_dump() for idea in result.ideas[:count]]


#
# OBRAZKI
#
def clean_page(image_bytes):
    """Wybiela tło i pogłębia czerń – czyste linie do druku."""
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = img.point(lambda p: 255 if p > 200 else (0 if p < 90 else p))
    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def generate_page(client, scene, level):
    prompt = f"{BASE_STYLE}\n{level['style']}\nSubject: {scene}"
    res = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=IMAGE_SIZE,
        quality=level["quality"],
        n=1,
    )
    return clean_page(base64.b64decode(res.data[0].b64_json))


def prepare_photo(photo_bytes):
    """Obraca wg EXIF, zmniejsza (taniej) i zwraca PNG + pasujący format wyjściowy."""
    img = ImageOps.exif_transpose(Image.open(BytesIO(photo_bytes))).convert("RGB")
    img.thumbnail((1536, 1536))
    w, h = img.size
    if w > h * 1.15:
        size = "1536x1024"
    elif h > w * 1.15:
        size = "1024x1536"
    else:
        size = "1024x1024"
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue(), size


def photo_to_page(client, photo_bytes, level, extra=""):
    png, size = prepare_photo(photo_bytes)
    prompt = (
        "Turn this photo into a coloring book page. Keep the same subjects, composition and recognizable "
        "features, but redraw everything as clean outline drawing. "
        f"{BASE_STYLE}\n{level['style']}"
    )
    if extra:
        prompt += f"\nAdditional wishes: {extra}"
    res = client.images.edit(
        model=IMAGE_MODEL,
        image=("photo.png", png, "image/png"),
        prompt=prompt,
        size=size,
        quality=level["quality"],
        n=1,
    )
    return clean_page(base64.b64decode(res.data[0].b64_json))


def run_with_limit(fn, *args):
    """Rezerwuje 1 kolorowankę z limitu dziennego; przy błędzie zwraca ją do puli."""
    counter = get_usage_counter()
    if not counter.try_reserve():
        st.warning("Dzienny limit kolorowanek został wyczerpany. Wróć jutro! 🌙")
        return None
    try:
        return fn(*args)
    except OpenAIError as e:
        counter.release()
        st.error(f"Nie udało się wygenerować obrazka: {e}")
        return None


#
# GALERIA I SESJE (v4)
#
def kolorowanki_word(n):
    """Poprawna odmiana: 1 kolorowankę, 2 kolorowanki, 5 kolorowanek."""
    if n == 1:
        return "1 kolorowankę"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} kolorowanki"
    return f"{n} kolorowanek"


def slugify(text):
    text = text.lower()
    for a, b in zip("ąćęłńóśźż", "acelnoszz"):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "kolorowanka"


def add_to_gallery(title, level_key, source, prompt, png):
    st.session_state["gallery"].append(
        {
            "title": title,
            "level": level_key,
            "source": source,
            "prompt": prompt,
            "created": datetime.now().isoformat(timespec="seconds"),
            "png": png,
        }
    )


def page_filename(i, item):
    return f"{i + 1:02d}_{slugify(item['title'])}.png"


def build_zip(items, with_session=False):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = []
        for i, item in enumerate(items):
            name = page_filename(i, item)
            zf.writestr(name, item["png"])
            meta.append({k: v for k, v in item.items() if k != "png"} | {"file": name})
        if with_session:
            zf.writestr(
                "session.json",
                json.dumps({"app": APP_NAME, "version": 1, "items": meta}, ensure_ascii=False, indent=2),
            )
    return buf.getvalue()


def load_session(zip_bytes):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        data = json.loads(zf.read("session.json"))
        items = []
        for meta in data["items"]:
            item = {k: v for k, v in meta.items() if k != "file"}
            item["png"] = zf.read(meta["file"])
            items.append(item)
    return items


#
# WYGLĄD
#
@st.cache_data
def _read_b64(path, mtime):
    return base64.b64encode(Path(path).read_bytes()).decode()


def asset_b64(name):
    path = ASSETS_DIR / name
    return _read_b64(str(path), path.stat().st_mtime) if path.exists() else ""


def inject_css():
    """Wspólny wygląd obu stron: tło z kredkową krainą, jasne szkło, niebieskie przyciski."""
    hero = asset_b64("hero.jpg")
    bg = f"url(data:image/jpeg;base64,{hero}) center / cover no-repeat fixed" if hero else "#FFF8EE"
    tile_rules = "\n".join(
        f".st-key-tile_{p}_{i} {{ border: 3px solid {IDEA_COLOR if p == 'idea' else PENCIL_COLORS[i % len(PENCIL_COLORS)]} !important; "
        f"border-radius: 20px; padding: 12px; background: rgba(255,255,255,.85); }}"
        for p in TILE_PREFIXES
        for i in range(60)
    )
    level_rules = "\n".join(
        f"[class*='st-key-lvl_'][class*='_{k}'] {{ border: 3px solid {v['color']}; }}\n"
        f"[class*='st-key-lvlsel_'][class*='_{k}'] {{ border: 3px solid {v['color']}; box-shadow: 0 0 0 5px {v['color']}55, 0 10px 28px {v['color']}88; }}\n"
        f"[class*='st-key-lvl'][class*='_{k}'] button {{ border: 2px solid {v['color']} !important; background: #fff !important; }}\n"
        f"[class*='st-key-lvl'][class*='_{k}'] button p {{ color: {v['text']} !important; }}\n"
        f"[class*='st-key-lvlsel_'][class*='_{k}'] button {{ background: {v['color']} !important; }}\n"
        f"[class*='st-key-lvlsel_'][class*='_{k}'] button p {{ color: #fff !important; }}"
        for k, v in LEVELS.items()
    )
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playpen+Sans:wght@600;700;800&family=Nunito:wght@500;600;700;800&display=swap');
html {{ font-size: clamp(12px, calc(100vh / 50), 20px); }}
.stApp {{ background: {bg}; color: {TEXT_COLOR}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stApp *:not([data-testid="stIconMaterial"]) {{ font-family: 'Nunito', sans-serif; }}
.stApp h1, .stApp h2, .stApp h3, .kai-title, .kai-title *, .kai-slogan, .kai-slogan *,
[data-testid="stTab"] p {{ font-family: 'Playpen Sans', cursive !important; }}
.stApp h2, .stApp h3 {{ color: {TEXT_COLOR}; font-weight: 700 !important; }}
.kai-title {{ font-weight: 800; line-height: 1.1; text-shadow: 0 2px 10px rgba(255,255,255,.85); }}

/* czytelne napisy */
.stMarkdown p {{ font-size: 1.2rem; color: {TEXT_COLOR}; }}
[data-testid="stWidgetLabel"] p {{ font-size: 1.3rem !important; font-weight: 700 !important; color: {TEXT_COLOR}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
    font-size: 1.05rem !important; color: {MUTED_TEXT} !important; }}
.stTextInput input, .stTextArea textarea {{ font-size: 1.2rem !important; color: {TEXT_COLOR} !important; }}
.stTextInput [data-baseweb="input"], .stTextArea [data-baseweb="textarea"] {{
    background: #fff !important; border: 2px solid #C9D6F2 !important; border-radius: 14px !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: rgba(255,255,255,.9); border: 2px dashed #9DB6EA; border-radius: 16px; }}
/* polskie napisy w polu wgrywania */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {{ display: none; }}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after {{ color: {MUTED_TEXT}; font-size: 1.05rem; }}
.st-key-photo_upl [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
    content: "Przeciągnij zdjęcie tutaj · JPG, PNG, WEBP do 20 MB"; }}
.st-key-session_upl [data-testid="stFileUploaderDropzoneInstructions"] > div::after {{
    content: "Przeciągnij tutaj plik ZIP zapisany przyciskiem „Zapisz sesję”"; }}
[data-testid="stFileUploaderDropzone"] button > * {{ display: none !important; }}
[data-testid="stFileUploaderDropzone"] button::after {{ font-size: 1.1rem; font-weight: 800; color: {ACCENT}; }}
.st-key-photo_upl [data-testid="stFileUploaderDropzone"] button::after {{ content: "Wybierz zdjęcie"; }}
.st-key-session_upl [data-testid="stFileUploaderDropzone"] button::after {{ content: "Wybierz plik sesji"; }}
[data-testid="stPageLink"] p {{ font-size: 1.2rem !important; font-weight: 700; color: {ACCENT}; }}

/* zakładki */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; }}
[data-testid="stTab"] p {{ font-size: 1.5rem !important; font-weight: 700 !important; color: {TEXT_COLOR}; }}
[data-testid="stTab"] {{ padding: .3rem .3rem .6rem; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {ACCENT}; }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background: {ACCENT}; height: 4px; }}

/* przyciski */
.stButton button, .stDownloadButton button {{ border-radius: 16px !important; min-height: 3rem;
    background: #fff !important; border: 2px solid {ACCENT} !important; }}
.stButton button p, .stDownloadButton button p {{ font-size: 1.2rem !important; font-weight: 800 !important;
    color: {ACCENT} !important; }}
.stButton button[kind="primary"] {{ background: {ACCENT} !important; border: 2px solid {ACCENT} !important;
    box-shadow: 0 8px 22px rgba(47,107,216,.30); }}
.stButton button[kind="primary"]:hover {{ background: #2457B8 !important; }}
.stButton button[kind="primary"] p {{ color: #fff !important; }}
.stButton button:disabled {{ opacity: .45; }}

/* poziomy trudności z podglądem */
[class*='st-key-lvl'] {{ border-radius: 20px; padding: 10px; background: rgba(255,255,255,.75); gap: .5rem; }}
[class*='st-key-lvl'] button {{ min-height: 3rem; }}
[class*='st-key-lvl'] button p {{ font-size: 1.3rem !important; }}
[class*='st-key-lvl'] [data-testid="stCaptionContainer"] {{ text-align: center; white-space: nowrap; }}
[class*='st-key-lvl'] [data-testid="stCaptionContainer"], [class*='st-key-lvl'] [data-testid="stCaptionContainer"] p {{
    color: {TEXT_COLOR} !important; font-weight: 700 !important; font-size: 1.12rem !important; opacity: 1 !important; }}
.kai-label {{ font-size: 1.3rem; font-weight: 700; color: {TEXT_COLOR}; margin: .3rem 0 0; padding-bottom: .85rem; }}
[data-testid="stTextInput"] [data-testid="stWidgetLabel"], [data-testid="stTextArea"] [data-testid="stWidgetLabel"] {{ padding-bottom: 5px; }}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] {{ margin-bottom: 1px; }}
@keyframes kai-spin {{ to {{ transform: rotate(360deg); }} }}
.kai-loader {{ display: flex; align-items: center; gap: 1rem; width: fit-content; margin: 1.2rem auto;
              padding: .9rem 1.6rem; background: rgba(255,255,255,.92); border-radius: 999px;
              box-shadow: 0 8px 26px rgba(40,60,120,.22); }}
.kai-ring {{ width: 46px; height: 46px; border-radius: 50%; flex: none;
            background: conic-gradient({", ".join(PENCIL_COLORS)}, {PENCIL_COLORS[0]});
            -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 8px), #000 calc(100% - 7px));
            mask: radial-gradient(farthest-side, transparent calc(100% - 8px), #000 calc(100% - 7px));
            animation: kai-spin .9s linear infinite; }}
.kai-loader p {{ margin: 0; font-family: 'Playpen Sans', cursive !important; font-size: 1.25rem; font-weight: 700;
                color: {TEXT_COLOR}; }}
.kai-alert {{ font-size: 1.3rem; font-weight: 800; color: #D62828; background: #FFF1F1;
              border: 2px solid #F08A8A; border-radius: 16px; padding: .8rem 1.2rem; margin: .4rem 0; }}
.kai-thumb {{ display: block; width: 78% !important; margin: 0 auto; max-width: none !important; height: 9.75rem !important; object-fit: cover !important; object-position: center 40%; background: transparent; }}

/* suwak "ile kolorowanek" */
[data-testid="stSlider"] [role="group"] > div > div:first-child {{ height: 10px !important; border-radius: 6px; }}
[data-testid="stSlider"] [role="group"] > div > div:nth-child(2) {{ width: 24px !important; height: 24px !important;
    background: #14B8C4 !important; border: 3px solid #fff !important; border-radius: 50% !important;
    box-shadow: 0 2px 10px rgba(20,184,196,.55) !important; cursor: grab; }}
[data-testid="stSliderTickBar"] {{ display: none !important; }}
[data-testid="stSliderThumbValue"] {{ display: none !important; }}
.st-key-count_ticks {{ margin: -.7rem 0 0 -13px; width: calc(100% + 26px) !important; max-width: none !important; }}
.st-key-count_ticks [data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}
.st-key-count_ticks [data-testid="stColumn"]:nth-child(2) [data-testid="stVerticalBlock"] {{ align-items: center; }}
.st-key-count_ticks [data-testid="stColumn"]:nth-child(3) [data-testid="stVerticalBlock"] {{ align-items: flex-end; }}
.st-key-count_ticks .stButton button {{ width: 36px; min-width: 36px; min-height: 0 !important; padding: 0 !important;
    background: transparent !important; border: none !important; box-shadow: none !important;
    display: flex; flex-direction: column; align-items: center; }}
.st-key-count_ticks .stButton button p {{ font-size: 1.15rem !important; font-weight: 800 !important; color: {ACCENT} !important; }}
.st-key-count_ticks .stButton button:hover p {{ text-decoration: underline; }}
.kai-ticks span {{ display: flex; flex-direction: column; align-items: center; width: 24px;
                   color: {ACCENT}; font-weight: 800; font-size: 1.15rem; line-height: 1.1; }}
.kai-ticks i {{ width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; display: block; margin-bottom: 2px; }}

.limit-pill {{ padding: .35rem 1rem; border-radius: 999px; width: fit-content; margin-left: auto; white-space: nowrap;
               border: 2px solid {ACCENT}; color: {ACCENT}; background: rgba(255,255,255,.9);
               font-weight: 800; font-size: 1.1rem; }}
{tile_rules}
{level_rules}
</style>
""",
        unsafe_allow_html=True,
    )


def colorful_html(text, by_word=False):
    """Każda litera (albo słowo) w innym kolorze kredki."""
    parts = text.split(" ") if by_word else list(text)
    out, i = [], 0
    for part in parts:
        if part.strip():
            out.append(f'<span style="color:{PENCIL_COLORS[i % len(PENCIL_COLORS)]}">{part}</span>')
            i += 1
        else:
            out.append(part)
    return (" " if by_word else "").join(out)


def limit_badge():
    left = get_usage_counter().remaining()
    st.markdown(f'<div class="limit-pill">Zostało dziś: {left}/{DAILY_LIMIT}</div>', unsafe_allow_html=True)


def loader_html(text):
    return f'<div class="kai-loader"><div class="kai-ring"></div><p>{text}</p></div>'


@contextmanager
def kai_loader(text):
    """Pokazuje kręcące się kolorowe kółeczko na czas długiej operacji."""
    box = st.empty()
    box.markdown(loader_html(text), unsafe_allow_html=True)
    try:
        yield box
    finally:
        box.empty()


def set_count(n):
    """Kliknięcie liczby pod suwakiem ustawia suwak na tę wartość."""
    st.session_state["count"] = n


def level_picker(prefix, sample_prefix="sample"):
    state_key = f"{prefix}_level"
    if state_key not in st.session_state:
        st.session_state[state_key] = "medium"
    st.markdown('<div class="kai-label">Poziom trudności</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for col, (key, lvl) in zip(cols, LEVELS.items()):
        selected = st.session_state[state_key] == key
        with col, st.container(key=f"{'lvlsel' if selected else 'lvl'}_{prefix}_{key}"):
            sample = asset_b64(f"{sample_prefix}_{key}.webp")
            if sample:
                st.markdown(f'<img class="kai-thumb" src="data:image/webp;base64,{sample}" '
                            f'alt="Przykład – {lvl["label"]}">', unsafe_allow_html=True)
            if st.button(lvl["label"], key=f"btn_{prefix}_{key}", width="stretch"):
                st.session_state[state_key] = key
                st.rerun()
            st.caption(lvl["desc"])
    return st.session_state[state_key]


def show_page(item, idx, key_prefix):
    level = LEVELS[item["level"]]
    with st.container(key=f"tile_{key_prefix}_{idx}"):
        st.image(item["png"], width="stretch")
        st.markdown(f"**{item['title']}**  \nPoziom: {level['label'].lower()}")
        st.download_button(
            "⬇️ Pobierz PNG",
            data=item["png"],
            file_name=page_filename(idx, item),
            mime="image/png",
            key=f"dl_{key_prefix}_{idx}",
            width="stretch",
        )


#
# STRONA STARTOWA
#
def landing_css():
    st.markdown(
        """
<style>
.block-container { max-width: 1120px; padding: 13vh 1rem 2rem; }
.st-key-hero_panel { background: rgba(255,255,255,.38); backdrop-filter: blur(8px) saturate(1.2);
                     -webkit-backdrop-filter: blur(8px) saturate(1.2); max-width: 860px; margin: 0 auto;
                     border: 2px solid rgba(255,255,255,.85); border-radius: 32px; padding: 1.6rem 1.6rem 2rem;
                     box-shadow: 0 16px 50px rgba(40,60,120,.25); align-items: center; }
.st-key-hero_panel [data-testid="stMarkdownContainer"], .st-key-hero_panel .stMarkdown { width: 100%; text-align: center; }
.st-key-hero_panel .kai-title { font-size: 6.4rem; margin: 0; text-align: center; }
.kai-slogan { font-size: 2.1rem; font-weight: 700; text-align: center; margin: .7rem 0 .5rem;
              text-shadow: 0 2px 10px rgba(255,255,255,.85); }
.st-key-hero_panel .kai-tagline { font-size: 1.4rem; line-height: 1.45; margin: 0 auto 1.1rem; text-align: center;
                                  max-width: 1000px; color: #1E3A6E; font-weight: 600; }
.st-key-hero_panel .stButton { display: flex; justify-content: center; }
.st-key-hero_panel .stButton button[kind="primary"] { min-height: 4rem; padding: 0 2.6rem; }
.st-key-hero_panel .stButton button[kind="primary"] p { font-size: 1.6rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )


def landing_page():
    landing_css()
    with st.container(key="hero_panel"):
        st.markdown(
            f'<div class="kai-title">{colorful_html(APP_NAME)}</div>'
            f'<div class="kai-slogan">{colorful_html(APP_SLOGAN, by_word=True)}</div>'
            '<div class="kai-tagline">Kolorowanki z Twojej wyobraźni albo z Twojego zdjęcia –<br>'
            "narysowane przez AI w kilka chwil.</div>",
            unsafe_allow_html=True,
        )
        if st.button("🖍️ Stwórz kolorowankę", type="primary"):
            st.switch_page(COLOR_PAGE)


def coloring_css():
    """Strona kolorowania: cała treść na jasnym szklanym panelu nad grafiką."""
    st.markdown(
        """
<style>
.block-container { max-width: min(1560px, calc(100vw - 5rem)); margin: .6rem auto 1.2rem; padding: .6rem 2.4rem 1.6rem !important;
                   min-height: calc(100vh - 1.8rem); height: auto !important; flex-shrink: 0 !important;
                   background: rgba(255,255,255,.66); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
                   border: 2px solid rgba(255,255,255,.9); border-radius: 32px;
                   box-shadow: 0 16px 50px rgba(40,60,120,.25); }
.kai-title.small { font-size: 2.8rem; text-align: center; margin: 0; }
.stTabs { margin-top: 1.8rem; }
.block-container [data-testid="stVerticalBlock"] { gap: 1rem; }
.st-key-count_row { margin-top: 1.1rem; }
.st-key-photo_btn { align-items: center; }
.kai-warn { font-size: 1.2rem; font-weight: 800; color: #9A5B00; background: #FFF6E0; border: 2px solid #F4C064;
            border-radius: 16px; padding: .7rem 1.1rem; margin: .3rem 0 .2rem; }
.kai-section { font-size: 1.5rem; font-weight: 800; color: #1E3A6E; margin: 2rem 0 .3rem; }
.stTabs [data-baseweb="tab-list"] { gap: 0 !important; }
[data-testid="stTab"] { flex: 1 1 0; justify-content: center; }
.stApp h3 { font-size: 1.6rem; margin: .2rem 0 .1rem; padding: 0; }
</style>
""",
        unsafe_allow_html=True,
    )


#
# APLIKACJA
#
def coloring_page():
    coloring_css()
    head_l, head_c, head_r = st.columns([1.3, 2, 1.3], vertical_alignment="center")
    with head_l:
        st.page_link(LANDING_PAGE, label="Strona główna", icon=":material/arrow_back:")
    with head_c:
        st.markdown(f'<div class="kai-title small">{colorful_html(APP_NAME)}</div>', unsafe_allow_html=True)
    with head_r:
        limit_badge()

    api_key = get_api_key()
    if not api_key:
        st.error("Brak klucza OpenAI. Dodaj `OPENAI_API_KEY` do pliku `.env` (lokalnie) albo do Secrets na Streamlit Cloud.")
        st.stop()
    client = get_openai_client(api_key)

    for key, default in (("gallery", []), ("ideas", []), ("last_batch", [])):
        if key not in st.session_state:
            st.session_state[key] = default

    no_limit_left = get_usage_counter().remaining() == 0

    text_tab, photo_tab, gallery_tab = st.tabs(["✏️ Z tekstu", "📷 Ze zdjęcia", "🗂️ Moja galeria"])

    # --- Z TEKSTU ---
    with text_tab:
        theme = st.text_input(
            "Co ma być na kolorowance?",
            placeholder="np. żółw pływający w oceanie, dinozaury na pikniku, jednorożec w kosmosie…",
        )
        level_key = level_picker("text")
        with st.container(key="count_row"):
            _, slider_col, _, button_col, _ = st.columns([0.8, 0.85, 0.3, 0.8, 0.8], vertical_alignment="center")
            with slider_col:
                if "count" not in st.session_state:
                    st.session_state["count"] = 1
                count = st.slider("Ile kolorowanek?", 1, MAX_PER_BATCH, key="count")
                with st.container(key="count_ticks"):
                    for n, tick_col in enumerate(st.columns(MAX_PER_BATCH), start=1):
                        with tick_col:
                            st.button(str(n), key=f"tick_{n}", on_click=set_count, args=(n,), type="tertiary")
            with button_col:
                ideas_clicked = st.button("💡 Wymyśl pomysły", type="primary", disabled=not theme.strip())

        if ideas_clicked:
            with kai_loader("Wymyślam pomysły…"):
                try:
                    st.session_state["ideas"] = generate_ideas(client, theme.strip(), count, LEVELS[level_key])
                    st.session_state["ideas_for"] = (theme.strip(), count)
                    st.session_state["ideas_gen"] = st.session_state.get("ideas_gen", 0) + 1
                    st.session_state["last_batch"] = []
                except Exception as e:
                    st.error(f"Nie udało się wymyślić pomysłów: {e}")

        ideas = st.session_state["ideas"]
        if ideas:
            st.markdown('<div class="kai-section">Pomysły – możesz je poprawić przed rysowaniem</div>',
                        unsafe_allow_html=True)
            gen = st.session_state.get("ideas_gen", 0)
            cols = st.columns(min(len(ideas), 4))
            for i, idea in enumerate(ideas):
                with cols[i % len(cols)], st.container(key=f"tile_idea_{i}"):
                    idea["title"] = st.text_input("Tytuł", idea["title"], key=f"idea_title_{gen}_{i}")
                    idea["prompt"] = st.text_area("Opis sceny", idea["prompt"], key=f"idea_prompt_{gen}_{i}",
                                                  height=150)

            batch_level = level_key
            stale = st.session_state.get("ideas_for") != (theme.strip(), count)
            left = get_usage_counter().remaining()
            if stale:
                st.markdown('<div class="kai-alert">⚠️ Zmieniono liczbę kolorowanek lub temat – kliknij '
                            '„Wymyśl pomysły”, żeby pomysły do nich pasowały.</div>', unsafe_allow_html=True)
            elif len(ideas) > left:
                st.info(f"Dziś można jeszcze wygenerować {left} kolorowanek – narysuję tyle, ile się zmieści.")
            if st.button("🎨 Generuj kolorowanki", type="primary", disabled=no_limit_left or stale):
                batch = []
                with kai_loader("Rysuję…") as box:
                    for i, idea in enumerate(ideas):
                        box.markdown(loader_html(f"Rysuję {i + 1}/{len(ideas)}: {idea['title']} – ok. 30–60 s"),
                                     unsafe_allow_html=True)
                        png = run_with_limit(generate_page, client, idea["prompt"], LEVELS[batch_level])
                        if png is None:
                            break
                        add_to_gallery(idea["title"], batch_level, "text", idea["prompt"], png)
                        batch.append(len(st.session_state["gallery"]) - 1)
                st.session_state["last_batch"] = batch
                st.rerun()

        if st.session_state["last_batch"]:
            st.markdown('<div class="kai-section">Gotowe kolorowanki</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for n, idx in enumerate(st.session_state["last_batch"]):
                with cols[n % 4]:
                    show_page(st.session_state["gallery"][idx], idx, "text")

    # --- ZE ZDJĘCIA ---
    with photo_tab:
        st.markdown('<div class="kai-label">Wgraj zdjęcie</div>', unsafe_allow_html=True)
        with st.container(key="photo_upl"):
            photo = st.file_uploader("Wgraj zdjęcie", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        photo_level = level_picker("photo", sample_prefix="sample_photo")
        extra_col, title_col = st.columns([1.6, 1])
        with extra_col:
            extra = st.text_input("Dodatkowe życzenia (opcjonalnie)", placeholder="np. pomiń tło, dodaj kwiatki wokół")
        with title_col:
            photo_title = st.text_input("Tytuł kolorowanki", value="Moje zdjęcie")

        if photo:
            left_col, right_col = st.columns(2, gap="large")
            with left_col:
                st.image(photo, caption="Twoje zdjęcie", width="stretch")
                with st.container(key="photo_btn"):
                    make_clicked = st.button("🖍️ Zrób kontury", type="primary", disabled=no_limit_left)
            with right_col:
                if make_clicked:
                    with kai_loader("Odrysowuję kontury… ok. 30–60 s"):
                        png = run_with_limit(photo_to_page, client, photo.getvalue(), LEVELS[photo_level],
                                             extra.strip())
                    if png:
                        add_to_gallery(photo_title or "Moje zdjęcie", photo_level, "photo", extra, png)
                        st.session_state["last_photo"] = len(st.session_state["gallery"]) - 1
                        st.rerun()
                if st.session_state.get("last_photo") is not None:
                    idx = st.session_state["last_photo"]
                    show_page(st.session_state["gallery"][idx], idx, "photo")

    # --- GALERIA ---
    with gallery_tab:
        gallery = st.session_state["gallery"]
        st.markdown('<div class="kai-warn">⚠️ Pobierz kolorowanki przed odświeżeniem strony lub zamknięciem '
                    'aplikacji – potem znikną z galerii.</div>', unsafe_allow_html=True)
        gallery_msg = st.session_state.pop("gallery_msg", None)
        if gallery_msg:
            st.success(gallery_msg)
        if not gallery:
            st.info("Tu pojawią się wszystkie kolorowanki z tej sesji.")
        else:
            c1, c2 = st.columns(2)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            c1.download_button("📦 Pobierz wszystkie (ZIP)", build_zip(gallery),
                               file_name=f"kolorowanki_{stamp}.zip", mime="application/zip",
                               width="stretch")
            c2.download_button("💾 Zapisz sesję", build_zip(gallery, with_session=True),
                               file_name=f"sesja_kolorowanek_{stamp}.zip", mime="application/zip",
                               width="stretch")
            cols = st.columns(4)
            for idx, item in enumerate(gallery):
                with cols[idx % 4]:
                    show_page(item, idx, "gallery")

        st.divider()
        with st.container(key="session_upl"):
            session_file = st.file_uploader("📂 Wczytaj zapisaną sesję (plik ZIP z przycisku „Zapisz sesję”)",
                                            type=["zip"], key="session_upload")
        if session_file and st.button("Wczytaj sesję"):
            try:
                loaded = load_session(session_file.getvalue())
                # dopisujemy do galerii (nie zastępujemy jej); pomijamy kolorowanki, które już w niej są
                known = {md5(item["png"]).hexdigest() for item in st.session_state["gallery"]}
                new_items = [item for item in loaded if md5(item["png"]).hexdigest() not in known]
                st.session_state["gallery"].extend(new_items)
                skipped = len(loaded) - len(new_items)
                msg = f"Dodano do galerii {kolorowanki_word(len(new_items))} z pliku"
                if skipped:
                    msg += f" (pominięto {skipped} – już były w galerii)"
                st.session_state["gallery_msg"] = msg + "."
                st.rerun()
            except (KeyError, ValueError, zipfile.BadZipFile):
                st.error("To nie jest plik sesji z tej aplikacji.")


#
# MAIN
#
st.set_page_config(page_title=APP_NAME, page_icon="🖍️", layout="wide")
inject_css()

LANDING_PAGE = st.Page(landing_page, title=APP_NAME, icon="🖍️", default=True)
COLOR_PAGE = st.Page(coloring_page, title=f"{APP_NAME} – koloruj", icon="🎨", url_path="koloruj")
st.navigation([LANDING_PAGE, COLOR_PAGE], position="hidden").run()
