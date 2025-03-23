import fitz
from PIL import Image
import cv2
import copy
import numpy as np
import easyocr
import json
import os
from datetime import datetime
from rowUtilsNew import check_header_rows_2_and_3, findTextRows, findMatchingRowPatterns



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


# def standardize_image_file(image_path):
#     print("import image")
#     #filter for extension...k
#     _, extension = os.path.splitext(image_path)
    
#     # Convert extension to lowercase for consistent comparison
#     extension = extension.lower()

#     if extension == ".pdf":
#         print(f"Processing PDF file: {image_path}")
#         # Add PDF processing logic here
#     elif extension == ".tiff":
#         print(f"Processing TIFF file: {image_path}")
#         # Add TIFF processing logic here
#     elif extension in [".jpg", ".jpeg"]:
#         print(f"Processing JPEG file: {image_path}")
#         # Add JPEG processing logic here
#     else:
#         print(f"Unsupported file type: {image_path}")

# def standardize_png(file_path):
#     print("standardize")
#     image = Image.open(file_path)
    

#     # Define the new size (width, height)
#     SCALING_FACTOR = (300 / 72 )  # Replace with your desired dimensions

#     # Resize the image
#     resized_image = image.resize(SCALING_FACTOR)
#     width, height = resized_image.size

#     OUTPUT_IMAGE_PATH = "Temp/temp.png"
#     # Save or show the resized image
#     resized_image.save(OUTPUT_IMAGE_PATH)  # Save the resized image

#     return OUTPUT_IMAGE_PATH, width, height


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

    # if file_extension == ".png":
    #     return standardize_png(file_path)
    if file_extension == ".pdf":
        print("... is a pdf")
        return pdf_to_png(file_path)
    # elif file_extension in [".jpg", ".jpeg"]:
    #     return image_to_png(file_path)
    # elif file_extension == ".tiff":
    #     raise ValueError(f"Unimplemented file type: {file_extension}")
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

    pdf_page2 = pdf_document.load_page(1)  # Load the second page
    pix2 = pdf_page2.get_pixmap(matrix=fitz.Matrix(SCALING_FACTOR, SCALING_FACTOR))
    pix2.save(OUTPUT_IMAGE_PATH2)
    print("page 2...")
    
    height1 = pix1.height
    width1 = pix1.width
    height2 = pix2.height
    width2 = pix2.width
    print("end pdf to png...")
    return OUTPUT_IMAGE_PATH1, width1, height1, OUTPUT_IMAGE_PATH2, width2, height2

# def image_to_png(image_path):
#     OUTPUT_IMAGE_PATH = "Temp/temp.png"
    
#     # Open the image (TIFF, JPG, or JPEG) and check its DPI
#     img = Image.open(image_path)
    
#     # Get current DPI and dimensions
#     dpi = img.info.get('dpi', (300, 300))  # Default to 72 DPI if not specified
#     current_dpi = dpi[0]  # Usually, both X and Y DPI are the same

#     # # Scale image to match 300 DPI if it's not already
#     # scaling_factor = 300 / current_dpi
#     # new_width = int(img.width * scaling_factor)
#     # new_height = int(img.height * scaling_factor)

#     #-----
#     if img.width > 2600:
#         scale_fact = 2550 / img.width
#         new_width = int(img.width * scale_fact)
#         new_height = int(img.height * scale_fact)
#         img = img.resize((new_width, new_height), Image.LANCZOS)

#     # # Resize image if scaling is needed
#     # if scaling_factor != 1:
#     #     img = img.resize((new_width, new_height), Image.LANCZOS)
    
#     # Convert to PNG and save
#     img = img.convert("RGBA")  # Ensure transparency support if needed
#     img.save(OUTPUT_IMAGE_PATH, "PNG")
    
#     width, height = img.size
#     return OUTPUT_IMAGE_PATH, width, height


# from PIL import Image

# def image_to_png(image_path):
#     OUTPUT_IMAGE_PATH = "Temp/temp.png"
    
#     # Open the image (TIFF, JPG, or JPEG) and check its DPI
#     img = Image.open(image_path)
    
#     # Target size and DPI for 300 DPI on a 2550 x 3300 image
#     target_width, target_height = 2550, 3300
#     target_dpi = 300
    
#     # Get current DPI, default to 72 if not specified
#     current_dpi = img.info.get('dpi', (72, 72))[0]  # Usually, both X and Y DPI are the same
    
#     # Adjust image size based on current DPI if not already at 300 DPI
#     if current_dpi != target_dpi:
#         scaling_factor = target_dpi / current_dpi
#         new_width = int(img.width * scaling_factor)
#         new_height = int(img.height * scaling_factor)
        
#         # Resize image to reach target DPI dimensions
#         img = img.resize((new_width, new_height), Image.LANCZOS)
    
#     # Final resize to standardize to 2550 x 3300 pixels if needed
#     if img.width != target_width or img.height != target_height:
#         img = img.resize((target_width, target_height), Image.LANCZOS)
    
#     # Convert to PNG and save
#     img = img.convert("RGBA")  # Ensure transparency support if needed
#     img.save(OUTPUT_IMAGE_PATH, "PNG", dpi=(target_dpi, target_dpi))
    
#     # Return the output path and the final dimensions
#     width, height = img.size
#     return OUTPUT_IMAGE_PATH, width, height

def image_to_png(image_path):
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
    width, height = target_width, target_height
    return OUTPUT_IMAGE_PATH, width, height


































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