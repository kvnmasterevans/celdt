# English Learner Detector / CELDT Project

This repository contains scripts for detecting English learner information from student transcripts in image or PDF format.  
The program reads PDFs and image files, extracts relevant text using OCR, and identifies CELDT and ELPAC data.

---


## 🚀 How to Use

To run the program, execute the following command from your terminal:

```bash
python main.py run all [folder_name]
```
usage example:
python main.py run all Transcripts

If you want to do student name & ID redaction add the --redact-pii argument to the call.
usage example:
python main.py run all Transcripts --redact-pii

Note:
Each run overwrites the existing text_output.txt and csv_output.csv files.
If you want to preserve results from previous runs, move or rename those files before running the program again.

## Version History

### 🧩 Version 0.01 — *March 3, 2025*
**Repository:** `English_Learner_Detector`  
- Supported formats: **PDF, PNG, JPG** (page one only)  
- Used simple string matching to detect English learner status  
- Did **not** support TIFF files  
- Search strings stored in `English_Learner_Detector.py` (easy to modify)

---

### 🧩 Version 0.02 — *March 24, 2025*
**Repository:** `celdt`  
- Supported format: **PDF only**  
- Added basic CELDT and ELPAC detection  
- Did not search for any other class-related strings  
- Specialized search method stored in `check_for_CELDT.py` (not yet user-modifiable)  
- The initial commit works; later attempts to extend to PNG/JPG (April 11–14) introduced errors

---

### 🧩 Version 0.03 — *October 14, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information  
- **Known Bug:**  
  - Occasional `list index` error on certain transcripts

---

### 🧩 Version 0.04 — *December 18, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- Fixed a bug in one of the loops in FinalizeColumns.py that would access an out-of-range index

---

### 🧩 Version 0.05 — *December 22, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- Fixed a variety of bugs including accessing NONE data type, index out-of-range errors, corrected header detection method and updated .png standardization method
- Got the new Algorithm Detection method to work by increasing the minimum pixel count threshold to 10,000
- Has simple file-processing time tracker

---

### 🧩 Version 0.05.1 — *December 23, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- Improved per-file time tracking output structure

---

### 🧩 Version 0.05.2 — *December 23, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- Can now use either new or old column algorithm by setting bool at top of utils.py, USE_NEW_COLUMN_ALGORITHM
- **Known Bug:**  
  - Running program the first time CAN cause a whole slew of errors that disapper upon the second run. (Presumably this has to do with initializing EasyOCR)
---

### 🧩 Version 0.05.3 — *December 23, 2025*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- More detailed time output
- **Known Bug:**  
  - Running program the first time CAN cause a whole slew of errors that disapper upon the second run. (Presumably this has to do with initializing EasyOCR)
---

### 🧩 Version 0.05.4 — *January 11, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- **Renamed: **
        check_for_CELDT.py => check_for_CELDT_and_ELPAC.py
        rowUtilsNew.py => Row_Utilities.py
        AltFinColsHolder.py => New_Column_Algorithm.py
        FinalizeColumns.py => Old_Column_Algorithm.py
- cleaned up the code removing many print lines and removing unused commented-out code
- removed the datetime import from the utils.py file
- Moved the EasyOCR initialization line to outside of the loop in utils.py to stop redundant calls and also to hopefully fix the presumed initialization bug

---

### 🧩 Version 0.05.5 — *January 13, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Extracts detailed CELDT/ELPAC date and score information as well as entry / exit dates  
- Now also produces a tsv file, tsv_output.txt (has identical info to the csv file)

### 🧩 Version 0.06.0 — *February 27, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Now creates a course catalog from transcripts

---

### 🧩 Version 0.06.1 — *Mar 22, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Added location-based entry and exit date detection

---

### 🧩 Version 0.07.0 — *Apr 6, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Added a cache-only run version

---

### 🧩 Version 0.07.1 — *Apr 15, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Fixed code so that if you sepecify a cache directory other than OCR_Cache then it actually runs

---

### 🧩 Version 0.07.2 — *Apr 20, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Fixed subtle difference between cache-less and cached version of the script so that both produce identical outputs the way the cache-less original version formats everything

---

### 🧩 Version 0.07.3 — *Apr 26, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Updated Readme

---

### 🧩 Version 0.07.4 — *May 4, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Altered utils.py cache method so that files with different names but duplicate contents properly create different caches

---

### 🧩 Version 0.08.0 — *May 22, 2026*
**Repository:** `celdt`  
- Supports **PDF, PNG, and JPG**  
- Now outputs preliminary_course_catalog.json instead of course_catalog.json, which is formatted to function as input for the Course_Catalog_Finalizer script

---

### 🧩 Version 0.08.1 — *May 24, 2026*
**Repository:** `celdt`  
- Updated the README.md file

---

### 🧩 Version 0.09 — *June 22, 2026*
**Repository:** `celdt`  
- changed preliminary_course_catalog.json output to also list the files that each observation came from

---

### 🧩 Version 0.09 — *June 23, 2026*
**Repository:** `celdt`  
- changed preliminary_course_catalog.json output to also list the files that each observation came from

### 🧩 Version 0.10 — *August 20, 2026*
**Repository:** `celdt`  
- Added string based redaction (reads student name and ID from the filename and redacts that in the rows)
    - * The redaction is far too broad at the moment and redacts much more than just PII

---

## 🏗️ Basic Architecture (Version 0.05.3)

The project consists of **six main functional files:**

---

### `main.py`
- Handles command-line input  
- Iterates through target folder files  
- Calls `process_image()` from `utils.py`
- tracks time stamps for each file

---

### `utils.py`
- Contains the main `process_image()` method  
- Imports utilities from `Row_Utilities.py`, `Old_Column_Algorithm.py`, and `check_for_CELDT_and_ELPAC.py`  
      (previously named `rowUtilsNew.py`, `FinalizeColumns.py`, and `check_for_CELDT.py`)
- Determines file type and routes it to `process_png()`, `process_jpg()`, or `process_pdf()`  

**Process Flow:**  
1. Converts each file to a standardized `.png` format using `pdf_to_png()`, `standardize_png()`, or `jpg_to_png()`  
2. Sends the standardized PNG to `extract_data()`  
3. For multi-page PDFs, also processes the final page and appends its results  

**`extract_data()` workflow:**  
- Runs OCR via `run_OCR()` and stores results in JSON (`convert_OCR_page_result_to_JSON()`)  
- Detects transfer worksheets using `check_for_transfer_worksheet()`  
- Cleans the image (removes text/top/bottom regions) using `removeText()`, `removeTop()`, `remove_img_bottom()`  
- Computes vertical black pixel projection (`blackPixProjProfile`)  
- Determines column positions using `check_predicted_column_values()` from `Old_Column_Algorithm.py` or `detect_four_columns` from `New_Column_Algorithm.py` (Chooses which method to use based on `USE_NEW_COLUMN_ALGORITHM` bool)
- Finds text rows via `findTextRows()`  
- Extracts entry and exit dates with `extract_entry_and_exit_dates()`  

**Returns:**  
`rows, OCR_Data_Path, transfer_worksheet_found, entry_date, exit_date`

---

### `Row_Utilities.py` formerly `rowUtilsNew.py`
- Core module for row and column analysis  
- Exports:
  - `check_header_rows_2_and_3()` *(legacy, can remove)*
  - `findTextRows()`
  - `findMatchingRowPatterns()`
- `findMatchingRowPatterns()` locates row boundaries based on course headers  
- `findTextRows()` uses OCR JSON + column positions to assign text into columns

---

### `Old_Column_Algorithm.py` formerly `FinalizeColumns.py`
- Contains `check_predicted_column_values()`  
- Uses black pixel projection profiles to identify and verify column boundaries

### `New_Column_Algorithm.py` formerly `AltFinColsHolder.py`
- Contains `check_predicted_column_values()`  
- Uses black pixel projection profiles to identify and verify column boundaries
- More efficient AND better written than the `Old_Column_Algorithm.py` counterpart

---

### `check_for_CELDT_and_ELPAC.py` formerly `check_for_CELDT.py`
- Contains `check_CELDT_ELPAC_status()`  
- Extracts CELDT and ELPAC data from text rows

---

### `extract_course_catalog.py` 
- Contains logic for extracting and saving a preliminary course catalog in a format that can later be run through the Course_Catalog_Finalizer program

---

## 🐞 Known Issues
- Redaction method is too broad and redacts far more than ideal

---

*Last Updated: January 11, 2025 (Version 0.05.3)*
