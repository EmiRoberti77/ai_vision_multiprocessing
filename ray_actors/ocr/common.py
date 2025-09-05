# ---- at top (replace your LOT/EXP regexes) ----
import re

LOT_KEYS = [
    r"LOT", r"BATCH", r"LOT\s*NO\.?", r"BATCH\s*NO\.?", r"PARTI\s*NO\.?", r"PART\s*NO\.?"
]
# accept A–Z0–9 and hyphen/underscore, length >= 3
LOT_VALUE_RE = re.compile(r"[A-Z0-9][A-Z0-9\-_]{2,}")

# include Turkish + common English variants for expiry
EXP_KEYS = [
    r"EXP(?:I?RY)?", r"USE\s*BY", r"BEST\s*BEFORE", r"EXPIRES?",
    r"SON\s*KULL", r"S\.?K\.?T\.?"
]
MONTH = {'JAN':'01','FEB':'02','MAR':'03','APR':'04','MAY':'05','JUN':'06',
         'JUL':'07','AUG':'08','SEP':'09','SEPT':'09','OCT':'10','NOV':'11','DEC':'12'}

# YYYY-MM / MM-YYYY / MM-YY / "01 Dec 2026" / "Dec 2026"
EXP_DATE_RE = re.compile(r"""(?ix)
(?:
  (?P<Y1>20\d{2})[.\-/](?P<M1>0?[1-9]|1[0-2])         # 2026-08
 |(?P<M2>0?[1-9]|1[0-2])[.\-/](?P<Y2>20\d{2})         # 08/2026
 |(?P<M3>0?[1-9]|1[0-2])[.\-/](?P<Y3>\d{2})           # 08/26
 |(?P<MN1>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(?P<Y4>20\d{2})
 |(?P<D1>\d{1,2})\s+(?P<MN2>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(?P<Y5>20\d{2})
)
""")

SPACEY_DIGITS = re.compile(r"(?<!\d)(\d)\s+(?=\d)")  # collapse separated digits: '2 0 2 6' -> '2026'

def collapse_spaced_digits(s: str) -> str:
    # repeatedly collapse '2 0 2 6' -> '2026'
    prev = None
    while prev != s:
        prev = s
        s = SPACEY_DIGITS.sub(r"\1", s)
    return s

def clean_line(txt: str) -> str:
    t = txt.upper()
    t = t.replace("；", ":").replace(";", ":").replace("—", "-").replace("`", " ").replace("’", "'")
    t = t.replace("SN:", "SN:").replace("SM:", "SM:").replace("PC:", "PC:")
    t = collapse_spaced_digits(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def norm_month(tok: str) -> str | None:
    return MONTH.get(tok.upper())

def parse_expiry_from_text(up: str) -> str:
    m = EXP_DATE_RE.search(up)
    if not m:
        return ""
    yyyy, mm = None, None
    if m.group('Y1') and m.group('M1'):
        yyyy, mm = m.group('Y1'), f"{int(m.group('M1')):02d}"
    elif m.group('M2') and m.group('Y2'):
        yyyy, mm = m.group('Y2'), f"{int(m.group('M2')):02d}"
    elif m.group('M3') and m.group('Y3'):
        yyyy = f"20{int(m.group('Y3')):02d}"
        mm = f"{int(m.group('M3')):02d}"
    elif m.group('MN1') and m.group('Y4'):
        yyyy, mm = m.group('Y4'), norm_month(m.group('MN1'))
    elif m.group('D1') and m.group('MN2') and m.group('Y5'):
        yyyy, mm = m.group('Y5'), norm_month(m.group('MN2'))
    if yyyy and mm and 2000 <= int(yyyy) <= 2100 and 1 <= int(mm) <= 12:
        return f"{yyyy}-{mm}"
    return ""

def find_lot_on_line(line: str) -> str:
    # look for any key, then capture a reasonable code to the right
    for key in LOT_KEYS:
        m = re.search(rf"\b{key}\b\s*[:\-]?\s*(?P<val>[A-Z0-9\-_]{{3,}})", line)
        if m:
            val = m.group("val")
            # guard against common false positives like EXP
            if not re.match(r"^EXP$", val):
                return val
    return ""

def has_exp_key(line: str) -> bool:
    return any(re.search(rf"\b{k}\b", line) for k in EXP_KEYS)
