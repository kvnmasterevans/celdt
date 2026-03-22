import os
import json

def save_debug_json(data, original_filename, page_number, debug_dir="ocr_debug"):
    # create folder if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)

    # strip extension from original filename
    base_name = os.path.splitext(os.path.basename(original_filename))[0]

    # build unique filename
    debug_filename = f"{base_name}_page{page_number}.json"

    debug_path = os.path.join(debug_dir, debug_filename)

    # write json
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return debug_path



# import os
# import json
# import shutil

# def save_debug_json(OCR_Data_Path, original_filename, page_number):
    
#     debug_dir = "ocr_debug"
#     os.makedirs(debug_dir, exist_ok=True)

#     base_name = os.path.splitext(os.path.basename(original_filename))[0]

#     debug_filename = f"{base_name}_page{page_number}.json"
#     debug_path = os.path.join(debug_dir, debug_filename)
    
#     debug_dir = "ocr_debug"
#     os.makedirs(debug_dir, exist_ok=True)

#     base_name = os.path.splitext(os.path.basename(original_filename))[0]
#     debug_filename = f"{base_name}_page{page_number}.json"
#     debug_path = os.path.join(debug_dir, debug_filename)

#     shutil.copy(OCR_Data_Path, debug_path)