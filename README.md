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

## 🏗️ Basic Architecture (Version 0.03)

The project consists of **five main functional files:**

---

### `main.py`
- Handles command-line input  
- Iterates through target folder files  
- Calls `process_image()` from `utils.py`

---

### `utils.py`
- Contains the main `process_image()` method  
- Imports utilities from `rowUtilsNew.py`, `FinalizeColumns.py`, and `check_for_CELDT.py`  
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
- Determines column positions using `check_predicted_column_values()`  
- Finds text rows via `findTextRows()`  
- Extracts entry and exit dates with `extract_entry_and_exit_dates()`  

**Returns:**  
`rows, OCR_Data_Path, transfer_worksheet_found, entry_date, exit_date`

---

### `rowUtilsNew.py`
- Core module for row and column analysis  
- Exports:
  - `check_header_rows_2_and_3()` *(legacy, can remove)*
  - `findTextRows()`
  - `findMatchingRowPatterns()`
- `findMatchingRowPatterns()` locates row boundaries based on course headers  
- `findTextRows()` uses OCR JSON + column positions to assign text into columns

---

### `FinalizeColumns.py`
- Contains `check_predicted_column_values()`  
- Uses black pixel projection profiles to identify and verify column boundaries

---

### `check_for_CELDT.py`
- Contains `check_CELDT_status()`  
- Extracts CELDT and ELPAC data from text rows

---

## 🐞 Known Issues
- Some transcripts cause a `list index out of range` error during data extraction 
  - This error occurs in the FinalizeColumns.py in the ColumnConfirm3 method
  - where the index is greater than the length of proj_profile via proj_profile[index]

---

*Last Updated: October 14, 2025 (Version 0.03)*
