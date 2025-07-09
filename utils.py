import fitz
import difflib
from PIL import Image
import cv2
import copy
import numpy as np
import easyocr
import json
import os
from datetime import datetime
from rowUtilsNew import check_header_rows_2_and_3, findTextRows, findMatchingRowPatterns
from FinalizeColumns import check_predicted_column_values
from check_for_CELDT import check_CELDT_status



def log_message(filename, message):
    with open(filename, 'a') as file:
        file.write(message + '\n')

def save_image(image, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(file_path, image)

    
def get_unique_filename(directory, base_filename, extension):
    """
    Generates a unique file name by appending the current date and time,
    and if necessary, a number to avoid duplicates.
    """
    # Get the current date and time
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{base_filename}{extension}"
    full_path = os.path.join(directory, filename)
    
    # Check if the file already exists and append a number if it does
    counter = 1
    while os.path.exists(full_path):
        filename = f"{base_filename}({counter}){extension}"
        full_path = os.path.join(directory, filename)
        counter += 1
    
    return full_path

def openJpgImage(jpg_path):
    image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    return image

def remove_extension(file_name):
    root, ext = os.path.splitext(file_name)
    return root

# converts an OCR result into json format
# saves the json file
# and returns the path to the json file
def convert_OCR_results_to_json(result1, result2, fileName):
    output_data1 = []
    for detection in result1:
        bounding_box, text, confidence = detection


        # Reformat the bounding box information
        bounding_box = [{'x': float(x), 'y': float(y)} for x, y in bounding_box]

        output_data1.append({
            'text': text,
            'bounding_box': bounding_box,
            'confidence': confidence
        })
    output_data2 = []
    for detection in result2:
        bounding_box, text, confidence = detection


        # Reformat the bounding box information
        bounding_box = [{'x': float(x), 'y': float(y)} for x, y in bounding_box]

        output_data2.append({
            'text': text,
            'bounding_box': bounding_box,
            'confidence': confidence
        })

    
    fileName = os.path.basename(remove_extension(fileName))
    output_file_path1 = "OCR_Data/" + fileName + "1.json"
    output_file_path2 = "OCR_Data/" + fileName + "2.json"
    os.makedirs("OCR_Data", exist_ok=True)
    with open(output_file_path1, 'w') as json_file:
        json.dump(output_data1, json_file)
    with open(output_file_path2, 'w') as json_file:
        json.dump(output_data2, json_file)

    return output_file_path1, output_file_path2


# // NEW LOGIC BEING IMPLEMENTED
# //
# // UNDER CONSTRUCTION
# //
def convert_OCR_page_result_to_json(result, fileName, page_number):
    output_data = []
    for detection in result:
        bounding_box, text, confidence = detection


        # Reformat the bounding box information
        bounding_box = [{'x': float(x), 'y': float(y)} for x, y in bounding_box]

        output_data.append({
            'text': text,
            'bounding_box': bounding_box,
            'confidence': confidence
        })
    fileName = os.path.basename(remove_extension(fileName))
    output_file_path = "OCR_Data/" + fileName + str(page_number) + ".json"
    os.makedirs("OCR_Data", exist_ok=True)
    with open(output_file_path, 'w') as json_file:
        json.dump(output_data, json_file)
    return output_file_path


# runs easy OCR on the provided image and returns the results
def run_ocr(png_img_path):
    
    # # Store jpg in temorary file for OCR read
    # OUTPUT_IMAGE_PATH = "Temp/temp.png"
    # pdf_document = fitz.open(jpg_img_path)
    # pdf_page = pdf_document.load_page(0)
    # pix = pdf_page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # Convert to image with 300 DPI
    # pix.save(OUTPUT_IMAGE_PATH)
    print("starting ocr ...")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(png_img_path) # OUTPUT_IMAGE_PATH
    print("ocr finished")

    return result


# convert image to jpg 
# save the jpg
# and return path of that jpg file along with width and height values of the jpg
def pdf_to_jpg_path(pdf_path):

    SCALING_FACTOR = 300 / 72
    
    height = pix.height
    width = pix.width
    OUTPUT_IMAGE_PATH = "Temp/temp.jpg"
    os.makedirs("Temp", exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    pdf_page = pdf_document.load_page(0)  # Load the first page (index 0) 
    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR))  # Convert to image with 300 DPI
    pix.save(OUTPUT_IMAGE_PATH)

    
    return OUTPUT_IMAGE_PATH, width, height


def standardize_png(file_path):
    print("Standardizing", file_path)
    image = Image.open(file_path)
    
    # Define the desired width
    desired_width = 2550

    # Calculate the scaling factor based on the desired width
    scaling_factor = desired_width / image.width
    new_height = int(image.height * scaling_factor)
    
    # Resize the image
    resized_image = image.resize((desired_width, new_height), Image.LANCZOS)

    width, height = resized_image.size

    # Generate output path based on the original filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_image_path = "Temp/temp.png"
    
    # Save the resized image
    resized_image.save(output_image_path)

    return output_image_path, width, height




def standardize_image(file_path):
    print("standardizing image")
    file_extension = os.path.splitext(file_path)[1].lower()
    os.makedirs("Temp", exist_ok=True)

    if file_extension == ".png":
        return standardize_png(file_path)
    if file_extension == ".pdf":
        print("... is a pdf")
        return pdf_to_png(file_path)
    elif file_extension in [".jpg", ".jpeg"]:
        return image_to_png(file_path)
    elif file_extension == ".tiff":
        raise ValueError(f"Unimplemented file type: {file_extension}")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def pdf_to_png(pdf_path):
    print("pdf to png...")
    SCALING_FACTOR = 300 / 72  # Standardize PDFs to 300 DPI
    OUTPUT_IMAGE_PATH1 = "Temp/temp1.png"
    OUTPUT_IMAGE_PATH2 = "Temp/temp2.png"

    pdf_document = fitz.open(pdf_path)
    pdf_page1 = pdf_document.load_page(0)  # Load the first page
    print("page 1 loaded....")
    pix1 = pdf_page1.get_pixmap(matrix=fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR))
    print("pixmap")
    pix1.save(OUTPUT_IMAGE_PATH1)
    print("page 1...")


    if len(pdf_document) > 1:
        pdf_page2 = pdf_document.load_page(1)  # Load the second page
        pix2 = pdf_page2.get_pixmap(matrix=fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR))
        pix2.save(OUTPUT_IMAGE_PATH2)
        print("page 2...")
        height2 = pix2.height
        width2 = pix2.width
    else:
        OUTPUT_IMAGE_PATH2 = None
        height2 = None
        width2 = None
    height1 = pix1.height
    width1 = pix1.width
    
    print("end pdf to png...")
    return OUTPUT_IMAGE_PATH1, width1, height1, OUTPUT_IMAGE_PATH2, width2, height2

def single_pdf_to_png():
    print("new logic?")


def jpg_to_png(image_path):
    OUTPUT_IMAGE_PATH = "Temp/temp.png"
    
    # Open the image (TIFF, JPG, or JPEG) and check its DPI
    img = Image.open(image_path)
    
    # Target size and DPI for 300 DPI on a 2550 x 3300 image
    target_width, target_height = 2550, 3300
    target_dpi = 300
    
    # Get current DPI, default to 72 if not specified
    current_dpi = img.info.get('dpi', (72, 72))[0]  # Usually, both X and Y DPI are the same
    
    # Adjust image size based on current DPI if not already at 300 DPI
    if current_dpi != target_dpi:
        scaling_factor = target_dpi / current_dpi
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        
        # Resize image gradually in steps if needed
        while img.width > new_width * 1.5 or img.height > new_height * 1.5:
            img = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
        
        # Final resize to reach exact target DPI dimensions
        img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Final resize to standardize to 2550 x 3300 pixels if needed
    if img.width != target_width or img.height != target_height:
        img = img.resize((target_width, target_height), Image.BILINEAR)
    
    # Convert to PNG and save
    img = img.convert("RGBA")  # Ensure transparency support if needed
    img.save(OUTPUT_IMAGE_PATH, "PNG", dpi=(target_dpi, target_dpi))
    
    # Explicitly close the image to free memory
    img.close()
    
    # Return the output path and the final dimensions
    return OUTPUT_IMAGE_PATH, img.width, img.height


def image_to_png(image_path):
    OUTPUT_IMAGE_PATH1 = "Temp/temp1.png"
    OUTPUT_IMAGE_PATH2 = "Temp/temp2.png"
    
    # Open the image (TIFF, JPG, or JPEG) and check its DPI
    img = Image.open(image_path)
    
    # Target size and DPI for 300 DPI on a 2550 x 3300 image
    target_width, target_height = 2550, 3300
    target_dpi = 300
    
    # Get current DPI, default to 72 if not specified
    current_dpi = img.info.get('dpi', (72, 72))[0]  # Usually, both X and Y DPI are the same
    
    # Adjust image size based on current DPI if not already at 300 DPI
    if current_dpi != target_dpi:
        scaling_factor = target_dpi / current_dpi
        new_width = int(img.width * scaling_factor)
        new_height = int(img.height * scaling_factor)
        
        # Resize image gradually in steps if needed
        while img.width > new_width * 1.5 or img.height > new_height * 1.5:
            img = img.resize((img.width // 2, img.height // 2), Image.BILINEAR)
        
        # Final resize to reach exact target DPI dimensions
        img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Final resize to standardize to 2550 x 3300 pixels if needed
    if img.width != target_width or img.height != target_height:
        img = img.resize((target_width, target_height), Image.BILINEAR)
    
    # Convert to PNG and save
    img = img.convert("RGBA")  # Ensure transparency support if needed
    img.save(OUTPUT_IMAGE_PATH1, "PNG", dpi=(target_dpi, target_dpi))
    img.save(OUTPUT_IMAGE_PATH2, "PNG", dpi=(target_dpi, target_dpi))
    
    # Explicitly close the image to free memory
    img.close()
    
    # Return the output path and the final dimensions
    width, height = target_width, target_height
    return OUTPUT_IMAGE_PATH1, width, height, OUTPUT_IMAGE_PATH2, width, height











# def check_for_transfer_worksheet(OCRData, threshold=0.9):
#     print("~~~ 1")
#     target = "transfer admission worksheet"

#     for i in range(len(OCRData)):
#         combined_text = ""
#         for j in range(i, min(i + 3, len(OCRData))):
#             chunk = OCRData[j]
#             if chunk and chunk["text"]:
#                 combined_text += " " + chunk["text"].lower()

#         # Clean up extra spaces
#         combined_text = combined_text.strip()

#         # Get similarity ratio
#         similarity = difflib.SequenceMatcher(None, combined_text, target).ratio()

#         if similarity >= threshold:
#             print(f"Match found (similarity={similarity:.2f}): {combined_text}")
#             return True

#     return False


def check_for_transfer_worksheet(OCRData):
    target = "transfer admission worksheet"
    for chunk in OCRData:
        if chunk and "text" in chunk:
            if target in chunk["text"].lower():
                print(f"Found in single chunk: {chunk['text']}")
                return True
    return False







# draws white rectangles over all text (as detected by the OCR read)
# in order to "erase" that text
def removeText(textImagePath, OCRData):
    textImage = cv2.imread(textImagePath, cv2.IMREAD_GRAYSCALE)
    hei, wid = textImage.shape
    blank_image3 = np.ones((hei, wid, 3), dtype=np.uint8) * 255
    for textChunk in OCRData: 
        top_left_coord = (int(textChunk["bounding_box"][0]["x"]),int(textChunk["bounding_box"][0]["y"]))
        bot_right_coord = (int(textChunk["bounding_box"][2]["x"]),int(textChunk["bounding_box"][2]["y"]))
        # "erase" all text by drawing white rectangle over it
        textless_img = cv2.rectangle(textImage, top_left_coord, bot_right_coord, (255,255,255), thickness=cv2.FILLED)
    return textless_img


# draws a white rectangle over any portion of the image
# that's above the class data
def removeTop(textImage, OCRData, CourseStrings):
    
    hei, wid = textImage.shape

    

    stringDetected = False
    for textChunk in OCRData:
        for course in CourseStrings:
            if(course.lower() in textChunk["text"].lower()):
                if stringDetected == False:
                    stringDetected = True
                    rowVal = textChunk["bounding_box"][0]["y"] # pixel location for bottom of "couse id" line
                    # "erase" everything above classes table
                    topless_img = cv2.rectangle(textImage, (int(0),int(0)), (int(wid), int(rowVal)), (255,255,255), cv2.FILLED)

    if stringDetected == False:
        return textImage, 0
    else:           
        return topless_img, rowVal
    
def removeTop2(textImage1, textImage2, OCRData, CourseStrings):
    
    hei, wid = textImage1.shape

    

    stringDetected = False
    rowVals = []
    for textChunk in OCRData:
        for course in CourseStrings:
            if(course.lower() in textChunk["text"].lower()):
                # if stringDetected == False:
                stringDetected = True
                rowVals.append(textChunk["bounding_box"][0]["y"]) # pixel location for bottom of "couse id" line

    rowVals



    # "erase" everything above classes table
    topless_img1 = cv2.rectangle(textImage1, (int(0),int(0)), (int(wid), int(rowVals[0])), (255,255,255), cv2.FILLED)
    if len(rowVals) > 1:
        topless_img2 = cv2.rectangle(textImage2, (int(0),int(0)), (int(wid), int(rowVals[-1])), (255,255,255), cv2.FILLED)
    

    if stringDetected == False:
        return textImage1, textImage1, 0
    else:           
        return topless_img1, topless_img2, rowVals
    


# Find instances of "state id" appearing in the data
#   *Still need to do something about transcripts 6 and 9
def removeStateID(image, OCRData, CourseHeaderRow):
    stateIDStrings = ["state id"]
    detectedStateIdPositions = []
    hei, wid = image.shape

    stringDetected = False
    for textChunk in OCRData:
        for spelling in stateIDStrings:
            if(spelling in textChunk["text"].lower()):
                if stringDetected == False:
                    stringDetected = True
                    rowVal = [textChunk["bounding_box"][0]["y"], textChunk["bounding_box"][0]["x"]] # pixel location for bottom of "couse id" line
                    detectedStateIdPositions.append(rowVal)


    numStateId = len(detectedStateIdPositions)
    img_state_id_removed = image # initialize with original image
    if numStateId == 0:
        img_state_id_removed = image
    else:
        for state_id_position in detectedStateIdPositions:
            y_val = state_id_position[0]
            x_val = state_id_position[1]

            if ( x_val + int(wid / 5) ) > wid:
               censor_right_side = wid
            else:
                censor_right_side = x_val + int(wid / 5)
            
            if ( y_val + int(wid / 8) ) > hei:
               censor_bottom = hei
            else:
                censor_bottom = y_val + int(wid / 8)

            censor_bottom_right = ( int(censor_right_side), int(censor_bottom) )

            if y_val > CourseHeaderRow:
                # "erase" everything "state id" through bottom of image
                # img_state_id_removed = cv2.rectangle(img_state_id_removed, (int(0),int(y_val)), (int(wid), int(hei)), (255,255,255), cv2.FILLED)
                img_state_id_removed = cv2.rectangle(img_state_id_removed, (int(x_val),int(y_val)), (censor_bottom_right), (255,255,255), cv2.FILLED)

    return img_state_id_removed    
    


# draws a white rectangle a short (but arbitrary) distance below
# the top of the student course columns
def remove_img_bottom(img, topOfClasses, imgHeight, imgWidth):
    # pixel location for bottom of "couse id" line
    fifthOfImage = imgHeight / 5
    fifthBelowClasses = topOfClasses + fifthOfImage
    columnsOnlyImage = cv2.rectangle(img, (int(0),int(fifthBelowClasses)), (int(imgWidth),int(imgHeight)), (255,255,255), thickness=cv2.FILLED)
    return columnsOnlyImage



# creates a projection profile from a white pixel projection profile
# by subtracting the current white pixel value from the
# maximum possible white pixel value
# leaving us with only black pixels
# (because image is binary)
def createBlackPixelProjectionProfile(proj_profile):


    # find max # of white pixel density in profile
    most = 0
    for colDensity in proj_profile:
        if colDensity > most:
            most = colDensity
   
    blackPixels_proj_profile = []
    for pixel_cnt in proj_profile:
         # get max white pixel minus current white pixel to find black pixel
        blackPixels_proj_profile.append(most - pixel_cnt)
    
    return blackPixels_proj_profile



def drawColumnEdges(columns, image, height, width, coursesHeaderRow, OCR_Data):


    edge1 = columns[0]
    edge2 = columns[1]
    edge3 = columns[2]
    edge4 = columns[3]

    col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data, columns, coursesHeaderRow)

    row1Bot = findMatchingRowPatterns(col1Rows, coursesHeaderRow, edge2)

    row2Bot = findMatchingRowPatterns(col2Rows, coursesHeaderRow, edge3)

    row3Bot = findMatchingRowPatterns(col3Rows, coursesHeaderRow, edge4)
    

    imageCopy = copy.deepcopy(image)

    # draw vertical column edges
    cv2.rectangle(imageCopy, (int(edge1-4), int(height)),
                (int(edge1+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge2-4), int(height)),
                    (int(edge2+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge3-4), int(height)),
                    (int(edge3+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    cv2.rectangle(imageCopy, (int(edge4-4), int(height)),
                    (int(edge4+4), int(0)), (0,0,255), thickness=cv2.FILLED)
    
    # draw top boundary line
    cv2.rectangle(imageCopy, (0, int(coursesHeaderRow - 4)),
                (int(width), int(coursesHeaderRow + 4)), (0,0,255), thickness=cv2.FILLED)
    

    # draw column bottoms:
    col2matches, col3matches = check_header_rows_2_and_3(coursesHeaderRow, columns, OCR_Data)
    # draw row 1 bottom
    cv2.rectangle(imageCopy, (int(edge1), int(row1Bot+4)),
                    (int(edge2), int(row1Bot-4)), (0,0,255), thickness=cv2.FILLED)
    # draw row 2 bottom if it exists
    if col2matches:
        cv2.rectangle(imageCopy, (int(edge2), int(row2Bot+4)),
                        (int(edge3), int(row2Bot-4)), (0,0,255), thickness=cv2.FILLED)
    # draw row 3 bottom if it exists
    if col3matches:
        cv2.rectangle(imageCopy, (int(edge3), int(row3Bot+4)),
                        (int(edge4), int(row3Bot-4)), (0,0,255), thickness=cv2.FILLED)


    return imageCopy



























































def process_image(filename, input_folder_path):

    def remove_temporary_files():
        # Remove temporary image file
        OUTPUT_IMAGE_PATH1 = "Temp/temp1.png"
        OUTPUT_IMAGE_PATH2 = "Temp/temp2.png"
        OUTPUT_IMAGE_PATH = "Temp/temp.png"
        if os.path.exists(OUTPUT_IMAGE_PATH1):
            os.remove(OUTPUT_IMAGE_PATH1)
        if os.path.exists(OUTPUT_IMAGE_PATH2):
            os.remove(OUTPUT_IMAGE_PATH2)
        if os.path.exists(OUTPUT_IMAGE_PATH):
            os.remove(OUTPUT_IMAGE_PATH)
        # Remove temporary OCR_DATA .png
        if os.path.exists(OCR_Data_Path):
            os.remove(OCR_Data_Path)
        if OCR_Data_Path2 is not None:
            if os.path.exists(OCR_Data_Path2):
                os.remove(OCR_Data_Path2)

    def extract_data(png_path, height, width, page_number):
        # do ocr read if necessary
            result = run_ocr(png_path)
            OCR_Data_Path = convert_OCR_page_result_to_json(result, filename, page_number)

            #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            with open(OCR_Data_Path, 'r') as json_file:
                OCR_Data = json.load(json_file)


            # # # check for transfer worksheet
            transfer_worksheet_found = check_for_transfer_worksheet(OCR_Data)



            # originalImg = openJpgImage(standardized_png_path1)
            textlessImg = removeText(png_path, OCR_Data)
            CutoffStrings = ["standardized", "tests"]
            toplessImg, coursesHeaderRow = removeTop(textlessImg, OCR_Data, CutoffStrings)

            # column row stuff:
            bottomlessImg = remove_img_bottom(toplessImg, coursesHeaderRow, height, width)

            projection_profile = np.sum(bottomlessImg, axis=0)

            blackPixProjProfile = createBlackPixelProjectionProfile(projection_profile)

            columns = check_predicted_column_values(blackPixProjProfile, png_path, height) # stand png path var????

            col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data, columns, coursesHeaderRow)

            rows = []
            print("row 1:")
            for row in col1Rows:
                rows.append(row)
                print(f"/t{row}")
            print("row 2:")   
            for row in col2Rows:
                rows.append(row)
                print(f"/t{row}")
            print("row 3:")
            for row in col3Rows:
                rows.append(row)
                print(f"/t{row}")

            return rows, OCR_Data_Path, transfer_worksheet_found

    def process_pdf():
        # standardize image format
        standardized_png_path1, width1, height1, standardized_png_path2, width2, height2 = pdf_to_png(file_path)
        
        # STANDARD LOGIC
        rows, OCR_Data_Path, transfer_worksheet_found = extract_data(standardized_png_path1, height1, width1, 1)
        celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows = check_CELDT_status(rows)

        OCR_Data_Path2 = None
        if standardized_png_path2 is not None:
            print("logic for the second one...")
            
            # PAGE 2 ONLY LOGIC 
            rows_page_2, OCR_Data_Path2, _ = extract_data(standardized_png_path1, height1, width1, 2)

            celdt_detected_page_2, confirmed_celdt_rows_page_2, elpac_detected_page_2, elpac_rows_page_2 = check_CELDT_status(rows_page_2)
            celdt_detected = celdt_detected or celdt_detected_page_2
            elpac_detected = elpac_detected or elpac_detected_page_2
            for row in confirmed_celdt_rows_page_2:
                confirmed_celdt_rows.append(row)
            for elpac_row in elpac_rows_page_2:
                elpac_rows.append(elpac_row)

        return celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows, OCR_Data_Path, OCR_Data_Path2, transfer_worksheet_found
    
    
    def process_png():
        standardized_png, width, height = standardize_png(file_path)
        
        rows, OCR_Data_Path, transfer_worksheet_found = extract_data(standardized_png, height, width, 1)
        celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows = check_CELDT_status(rows)

        return celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows, OCR_Data_Path, transfer_worksheet_found
        

    def process_jpg():
        standardized_png, width, height = jpg_to_png(file_path)
        
        rows, OCR_Data_Path, transfer_worksheet_found = extract_data(standardized_png, height, width, 1)
        celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows = check_CELDT_status(rows)

        return celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows, OCR_Data_Path, transfer_worksheet_found




    print("processing transcript " + str(filename) + " in folder " + str(input_folder_path))

    file_path = os.path.join(input_folder_path, filename)
    file_extension = os.path.splitext(filename)[1]


    file_extension = os.path.splitext(file_path)[1].lower()
    OCR_Data_Path2 = None
    if file_extension == ".png":
        celdt_detected, confirmed_rows, elpac_detected, elpac_rows, OCR_Data_Path, transfer_worksheet_found = process_png()
    elif file_extension == ".pdf":
        celdt_detected, confirmed_rows, elpac_detected, elpac_rows, OCR_Data_Path, OCR_Data_Path2, transfer_worksheet_found = process_pdf()
    elif file_extension in [".jpg", ".jpeg"]:
        celdt_detected, confirmed_rows, elpac_detected, elpac_rows, OCR_Data_Path, transfer_worksheet_found = process_jpg()
    elif file_extension == ".tiff":
        raise ValueError(f"Unimplemented file type: {file_extension}")
    else:
        raise ValueError(f"file type {file_extension} not recognized")

    


    print("still going 1")
    remove_temporary_files()
    print("still going 1.1")
    return celdt_detected, confirmed_rows, elpac_detected, elpac_rows, transfer_worksheet_found    # dates, scores, score_types