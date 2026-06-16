import re
import sys
from collections import Counter

CORPUS_PATH = "real_data/full_corpus_ru.txt"
OUT_PATH = "real_data/full_corpus_ru_clean.txt"
REPORT_PATH = "_filter_report.txt"

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
LATIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

RE_CYRILLIC_WORD = re.compile(r'[а-яёА-ЯЁ]{2,}')
RE_NON_CYR_CHARS = re.compile(r'[^а-яёА-ЯЁa-zA-Z\s]')

RE_DOMAIN_SUFFIX = re.compile(
    r'(?:\.(?:ru|com|org|net|am|by|kz|ua|su|tatar|moscow)\b)(?!\s*[а-яё])',
    re.IGNORECASE
)
RE_URL_LIKE = re.compile(
    r'(?:https?://|www\.)[^\s]+',
    re.IGNORECASE
)
RE_FILE_PREFIX = re.compile(r'^(?:Файл|File):', re.IGNORECASE)
RE_FILE_EXT = re.compile(
    r'\b\w+\.(?:jpg|jpeg|png|gif|svg|bmp|tiff?|webp|pdf|djvu|docx?|xlsx?)\b',
    re.IGNORECASE
)
RE_REPEAT = re.compile(r'(.)\1{5,}')
RE_CYR = re.compile(r'[а-яёА-ЯЁ]')


def has_real_junk(line: str) -> str | None:
    s = line.strip()
    if not s:
        return "empty"

    if len(s) < 4:
        return "too_short"

    has_cyr = bool(RE_CYR.search(s))
    has_lat = bool(re.search(r'[a-zA-Z]', s))

    if not has_cyr and not has_lat:
        return "no_letters"

    # File artifacts
    if RE_FILE_PREFIX.search(s):
        return "file_artifact"
    if RE_FILE_EXT.search(s):
        cyr_words = RE_CYRILLIC_WORD.findall(s)
        if len(cyr_words) <= 2:
            return "file_artifact"

    # URL patterns
    if RE_URL_LIKE.search(s):
        return "url"
    m = RE_DOMAIN_SUFFIX.search(s)
    if m:
        before = s[:m.start()]
        non_cyr = len(RE_NON_CYR_CHARS.findall(before))
        if non_cyr < 3:
            pass
        elif len(RE_CYRILLIC_WORD.findall(before)) >= 2:
            pass
        else:
            return "url"

    # Non-Cyrillic scripts (Greek, Georgian, Arabic, CJK)
    foreign_script_chars = 0
    for ch in s:
        cp = ord(ch)
        if 0x0370 <= cp <= 0x03FF:
            foreign_script_chars += 1
        elif 0x10A0 <= cp <= 0x10FF:
            foreign_script_chars += 1
        elif 0x2D00 <= cp <= 0x2D2F:
            foreign_script_chars += 1
        elif 0x0600 <= cp <= 0x06FF:
            foreign_script_chars += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            foreign_script_chars += 1
        elif 0xAC00 <= cp <= 0xD7AF:
            foreign_script_chars += 1
    cyr_count = sum(1 for c in s if c in CYRILLIC)
    if foreign_script_chars >= 10 and cyr_count == 0:
        return "non_cyrillic_script"

    # Repetitive chars (ааааа, хахахаха, 123456)
    if RE_REPEAT.search(s):
        cyr_words = RE_CYRILLIC_WORD.findall(s)
        if len(cyr_words) <= 1:
            return "excessive_repetition"

    return None


def is_bibliography_junk(line: str) -> str | None:
    s = line.strip().strip('—–-− ').strip()
    if not s:
        return None
    cyr_words = RE_CYRILLIC_WORD.findall(s)
    if len(cyr_words) <= 2 and len(s) < 80:
        has_isbn = bool(re.search(r'[Ii][Ss][Bb][Nn]', s))
        has_other_marker = any(w in s for w in ['тир', 'илл', 'экз', 'стр', 'с.', 'Изд'])
        if has_isbn and has_other_marker:
            return "bibliography"
        if has_isbn and len(cyr_words) <= 1:
            return "bibliography"
    return None


def is_digit_row(line: str) -> str | None:
    s = line.strip()
    digits = sum(1 for c in s if c.isdigit())
    if digits > 0 and digits / max(len(s), 1) > 0.5:
        cyr_words = RE_CYRILLIC_WORD.findall(s)
        if len(cyr_words) <= 2:
            return "digit_row"
    return None


def has_orphan_suffix(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None

    for tld in ['.ru', '.com', '.am', '.by', '.kz', '.ua', '.su', '.reggi']:
        if s.lower().endswith(tld) and len(s) > len(tld) + 2:
            before_dot = s[:-len(tld)]
            if before_dot[-1].isalpha() and before_dot[-2].isalpha():
                rest = before_dot.strip()
                cyr_words = RE_CYRILLIC_WORD.findall(rest)
                if len(cyr_words) <= 1:
                    return "orphan_suffix"
    return None


def main():
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        all_lines = [l.rstrip('\n\r') for l in f]

    print(f"Read {len(all_lines)} lines from {CORPUS_PATH}")

    stats = Counter()
    kept = []
    removed = []

    for i, line in enumerate(all_lines):
        reason = has_real_junk(line)
        if reason is None:
            reason = has_orphan_suffix(line)
        if reason is None:
            reason = is_bibliography_junk(line)
        if reason is None:
            reason = is_digit_row(line)

        if reason:
            stats[reason] += 1
            removed.append((i, reason, line[:150]))
        else:
            kept.append(line)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for line in kept:
            f.write(line + '\n')

    total_removed = len(all_lines) - len(kept)

    with open(REPORT_PATH, 'w', encoding='utf-8') as r:
        r.write("CORPUS FILTER REPORT\n")
        r.write(f"Source: {CORPUS_PATH}\n")
        r.write(f"Total lines: {len(all_lines)}\n")
        r.write(f"Kept: {len(kept)}\n")
        r.write(f"Removed: {total_removed} ({100*total_removed/max(len(all_lines),1):.1f}%)\n\n")
        r.write("Removal breakdown:\n")
        for k, v in sorted(stats.items(), key=lambda x: -x[1]):
            if v > 0:
                r.write(f"  {k}: {v}\n")
        r.write("\nRemoved samples (up to 50):\n")
        for idx, (line_no, reason, snippet) in enumerate(removed):
            if idx >= 50:
                r.write(f"  ... and {len(removed) - 50} more\n")
                break
            r.write(f"  [{reason}] L{line_no}: {snippet}\n")

    print(f"Kept: {len(kept)} | Removed: {total_removed} ({100*total_removed/max(len(all_lines),1):.1f}%)")
    print(f"Saved: {OUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == '__main__':
    main()
