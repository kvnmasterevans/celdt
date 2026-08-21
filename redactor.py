from rapidfuzz import fuzz
import re


redact_threshold = .85
review_threshold = .75



def get_protected_terms_from_filename(filename):
    """
    Extract student ID, last name, and first name from a transcript filename.

    Expected format:

    AD High School Transcript for 005188868  CISNEROS, JOCELYN (Term ...)
    """

    pattern = (
        r"AD High School Transcript for\s+"
        r"(\d+)\s+"
        r"(.+?),\s+"
        r"(.+?)\s+"
        r"\(Term"
    )

    match = re.search(pattern, filename)

    if not match:
        raise ValueError(f"Could not parse filename: {filename}")

    student_id = match.group(1)
    last_name = match.group(2).strip()
    first_name = match.group(3).strip()

    return [student_id, last_name, first_name]



def compare_strings(target: str, candidate: str) -> float:
    """
    Return a 0-100 similarity score between two strings.

    RapidFuzz's ratio is based on normalized edit similarity and is
    appropriate for OCR-corrupted versions of a known string.
    """

    if not target or not candidate:
        return 0.0

    return fuzz.ratio(target, candidate)


# def redact_protected_info_from_rows(protected_terms, rows):

#     for row in rows:
#         print("redact row")
#         for protected_term in protected_terms:
#             similarity = compare_strings(row, protected_term)
#             if similarity > redact_threshold:
#                 print("redact")
#             elif similarity > review_threshold:
#                 print("review")


#     print("do the redaction")






def find_best_match(protected_term, row):
    """
    Find the substring of `row` that most closely matches
    `protected_term`.

    Returns:
        (start_index, end_index, similarity)
        or None if no candidate is found.
    """

    protected_term = protected_term.strip()

    if not protected_term:
        return None

    best_match = None
    best_similarity = 0.0

    # For now, compare against substrings approximately the same
    # length as the protected term.
    target_length = len(protected_term)

    # Allow OCR to make the candidate somewhat longer/shorter.
    min_length = max(1, target_length - 2)
    max_length = target_length + 2

    for length in range(min_length, max_length + 1):

        for start in range(0, len(row) - length + 1):

            end = start + length
            candidate = row[start:end]

            similarity = compare_strings(
                protected_term,
                candidate
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (start, end, similarity)

    return best_match


# def redact_protected_info_from_rows(protected_terms, rows):

#     redacted_rows = []

#     for row in rows:

#         row_text = row   # ["text"]
#         new_row = row_text

#         for protected_term in protected_terms:

#             match = find_best_match(
#                 protected_term,
#                 new_row
#             )

#             if match is None:
#                 continue

#             start, end, similarity = match

#             if similarity >= redact_threshold:

#                 print(
#                     f"REDACT: '{new_row[start:end]}' "
#                     f"matched '{protected_term}' "
#                     f"({similarity:.3f})"
#                 )

#                 new_row = (
#                     new_row[:start]
#                     + "[REDACTED]"
#                     + new_row[end:]
#                 )

#             elif similarity >= review_threshold:

#                 print(
#                     f"REVIEW: '{new_row[start:end]}' "
#                     f"possibly matches '{protected_term}' "
#                     f"({similarity:.3f})"
#                 )

#         redacted_rows.append(new_row)

#     return redacted_rows


def redact_protected_info_from_rows(protected_terms, rows):

    redacted_rows = []

    for row in rows:

        # Make a copy so we don't modify the original OCR data
        new_row = row.copy()

        # Copy the text list
        new_row["text"] = row["text"].copy()

        # Examine each individual OCR string
        for i, text in enumerate(new_row["text"]):

            for protected_term in protected_terms:

                match = find_best_match(
                    protected_term,
                    text
                )

                if match is None:
                    continue

                start, end, similarity = match

                if similarity >= redact_threshold:

                    print(
                        f"REDACT: '{text[start:end]}' "
                        f"matched '{protected_term}' "
                        f"({similarity:.3f})"
                    )

                    # Replace ONLY the matching portion
                    new_row["text"][i] = (
                        text[:start]
                        + "[REDACTED]"
                        + text[end:]
                    )

                    # Don't keep trying to redact this same text
                    # after we've already replaced it.
                    break

        redacted_rows.append(new_row)

    return redacted_rows