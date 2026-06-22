import re
import json
import os


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

CATALOG_FILE = "preliminary_course_catalog.json"

REJECT_KEYWORDS = [
    "TERM:",
    "CUMULATI",
    "Entry",
    "Exit",
    "Credit",
    "School",
    "Official",
    "State ID",
    "Grd"
]

CREDIT_PATTERN = re.compile(r"^\d+\.\d{2}$")
GRADE_PATTERN = re.compile(r"^[ABCDF][\+\-t]?$")


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def clean_course_code(token):
    digits = re.sub(r"[^\d]", "", token)
    if len(digits) >= 4:
        return digits
    return None


def is_credit(token):
    return bool(CREDIT_PATTERN.match(token))


def is_grade(token):
    return bool(GRADE_PATTERN.match(token))


# ---------------------------------------------------
# LOAD / SAVE
# ---------------------------------------------------

def load_catalog():
    if os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_catalog(catalog):
    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=4)


# ---------------------------------------------------
# EXTRACTION
# ---------------------------------------------------

def extract_course_catalog(filename, rows, existing_catalog):
    catalog = existing_catalog

    for row in rows:
        tokens = row.get("text", [])
        if not tokens:
            continue

        joined_row = " ".join(tokens)

        if any(keyword in joined_row for keyword in REJECT_KEYWORDS):
            continue

        # Find course code
        code_index = None
        course_code = None

        for i, token in enumerate(tokens):
            cleaned = clean_course_code(token)
            if cleaned:
                code_index = i
                course_code = cleaned
                break

        if course_code is None:
            continue

        # Find credit
        credit_index = None
        credit_value = None

        for i, token in enumerate(tokens):
            if is_credit(token):
                credit_index = i
                credit_value = token
                break

        if credit_value is None:
            continue

        # Extract title
        middle_tokens = tokens[code_index + 1 : credit_index]
        middle_tokens = [t for t in middle_tokens if not is_grade(t)]

        title = " ".join(middle_tokens).strip()
        title = re.sub(r"\s+", " ", title)

        if not title:
            continue
        





        '''
        Course Catalog structure:


        catalog
        │
        ├── course_code
        │   │
        │   ├── titles
        │   │   ├── observed_title : [frequency, [filename, filename, ...]]
        │   │   └── observed_title : [frequency, [filename, filename, ...]]
        │   │
        │   ├── credits
        │   │   ├── observed_credit : [frequency, [filename, filename, ...]]
        │   │   └── observed_credit : [frequency, [filename, filename, ...]]
        │   │
        │   └── count
        │
        └── course_code
        '''



        if course_code not in catalog:
            catalog[course_code] = {
                "titles": {},
                "credits": {},
                "count": 0
            }

        entry = catalog[course_code]

        entry["count"] += 1

        # Track title frequency
        if title not in entry["titles"]:
            entry["titles"][title] = [0, []]

        entry["titles"][title][0] += 1
        entry["titles"][title][1].append(filename)

        # Track credit frequency
        if credit_value not in entry["credits"]:
            entry["credits"][credit_value] = [0, []]

        entry["credits"][credit_value][0] += 1
        entry["credits"][credit_value][1].append(filename)

    return catalog