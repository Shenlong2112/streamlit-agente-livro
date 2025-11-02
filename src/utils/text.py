import re
from unidecode import unidecode

def slugify(text: str, max_len: int = 80) -> str:
    text = unidecode(text).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len].rstrip("-")

def safe_basename_from_filename(name: str) -> str:
    base = name.rsplit(".", 1)[0]
    return slugify(base)

def build_transcript_filename(title: str, date_utc: str, extra_kw: str | None = None) -> str:
    # date_utc no formato YYYYmmddTHHMMSSZ
    parts = [date_utc, slugify(title)]
    if extra_kw:
        parts.append(slugify(extra_kw, max_len=30))
    return "-".join([p for p in parts if p]) + ".md"
