import os
import csv
import argparse
import json
from utils import standardize_image, standardize_image2, removeText, removeTop, removeTop2, removeStateID, \
    run_ocr, convert_OCR_results_to_json, \
        openJpgImage, remove_extension, \
        save_image, \
        createBlackPixelProjectionProfile, remove_img_bottom
from rowUtilsNew import findTextRows
from FinalizeColumns import check_predicted_column_values
from check_for_CELDT import check_CELDT_status
import cv2
import numpy as np

def process_single_image(file_name, input_folder_path):
    redacted_image = process_image(file_name, input_folder_path)

    filename = os.path.basename(remove_extension(file_name))
    redacted_img_path = "Redacted/" + str(filename) + '.png'
    save_image(redacted_image, redacted_img_path)

    


# should return boolean for english learner status - ( turn into .txt file / files later... )
# return un-tilted image  with tilt degrees in name
# return column edges image
#   col top, col edges, col bottoms?
def process_image(filename, input_folder_path):

    def remove_temporary_files():
        # Remove temporary image file
        OUTPUT_IMAGE_PATH1 = "Temp/temp1.png"
        OUTPUT_IMAGE_PATH2 = "Temp/temp2.png"
        if os.path.exists(OUTPUT_IMAGE_PATH1):
            os.remove(OUTPUT_IMAGE_PATH1)
        if os.path.exists(OUTPUT_IMAGE_PATH2):
            os.remove(OUTPUT_IMAGE_PATH2)
        # Remove temporary OCR_DATA .png
        if os.path.exists(OCR_Data_Path1):
            os.remove(OCR_Data_Path1)
        if os.path.exists(OCR_Data_Path2):
            os.remove(OCR_Data_Path2)

    def process_pdf():
        # standardize image format
        standardized_png_path1, width1, height1, standardized_png_path2, width2, height2 = standardize_image(file_path)


        # do ocr read if necessary
        result1 = run_ocr(standardized_png_path1)
        result2 = run_ocr(standardized_png_path2)
        OCR_Data_Path1, OCR_Data_Path2 = convert_OCR_results_to_json(result1, result2, filename)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        with open(OCR_Data_Path1, 'r') as json_file:
            OCR_Data1 = json.load(json_file)
        with open(OCR_Data_Path2, 'r') as json_file:
            OCR_Data2 = json.load(json_file)


        # originalImg = openJpgImage(standardized_png_path1)
        textlessImg1 = removeText(standardized_png_path1, OCR_Data1)
        CutoffStrings1 = ["standardized", "tests"]
        toplessImg1, coursesHeaderRow1 = removeTop(textlessImg1, OCR_Data1, CutoffStrings1)
        textlessImg2 = removeText(standardized_png_path2, OCR_Data2)
        CutoffStrings2 = ["standardized", "tests"]
        toplessImg2, coursesHeaderRow2 = removeTop(textlessImg2, OCR_Data2, CutoffStrings2)
        # column row stuff:
        bottomlessImg1 = remove_img_bottom(toplessImg1, coursesHeaderRow1, height1, width1)
        bottomlessImg2 = remove_img_bottom(toplessImg2, coursesHeaderRow2, height1, width1)
        # cv2.imshow("bottomless image", bottomlessImg)
        # cv2.waitKey(0)

        projection_profile1 = np.sum(bottomlessImg1, axis=0)
        projection_profile2 = np.sum(bottomlessImg2, axis=0)

        blackPixProjProfile1 = createBlackPixelProjectionProfile(projection_profile1)
        blackPixProjProfile2 = createBlackPixelProjectionProfile(projection_profile2)

        columns1 = check_predicted_column_values(blackPixProjProfile1, standardized_png_path1, height1) # stand png path var????
        columns2 = check_predicted_column_values(blackPixProjProfile2, standardized_png_path2, height2)

        col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data1, columns1, coursesHeaderRow1)
        col4Rows, col5Rows, col6Rows = findTextRows(OCR_Data2, columns2, coursesHeaderRow2)

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
        for row in col4Rows:
            rows.append(row)
            print(f"/t{row}")
        for row in col5Rows:
            rows.append(row)
            print(f"/t{row}")
        for row in col6Rows:
            rows.append(row)
            print(f"/t{row}")
        return rows, OCR_Data_Path1, OCR_Data_Path2


    def process_other():
        # standardize image format
        standardized_png_path1, width1, height1, standardized_png_path2, width2, height2 = standardize_image2(file_path)


        # do ocr read if necessary
        result1 = run_ocr(standardized_png_path1)
        result2 = run_ocr(standardized_png_path2)
        OCR_Data_Path1, OCR_Data_Path2 = convert_OCR_results_to_json(result1, result2, filename)

        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        with open(OCR_Data_Path1, 'r') as json_file:
            OCR_Data1 = json.load(json_file)
        with open(OCR_Data_Path2, 'r') as json_file:
            OCR_Data2 = json.load(json_file)


        # originalImg = openJpgImage(standardized_png_path1)
        textlessImg1 = removeText(standardized_png_path1, OCR_Data1)
        textlessImg2 = removeText(standardized_png_path2, OCR_Data2)
        CutoffStrings1 = ["standardized", "tests"]
        toplessImg1, toplessImg2, coursesHeaderRows = removeTop2(textlessImg1, textlessImg2, OCR_Data1, CutoffStrings1)
        
        # CutoffStrings2 = ["standardized", "tests"]
        # toplessImg2, coursesHeaderRow2 = removeTop(textlessImg2, OCR_Data2, CutoffStrings2)
        # column row stuff:
        if len(coursesHeaderRows) > 0:
            bottomlessImg1 = remove_img_bottom(toplessImg1, coursesHeaderRows[0], height1, width1)
            projection_profile1 = np.sum(bottomlessImg1, axis=0)
            blackPixProjProfile1 = createBlackPixelProjectionProfile(projection_profile1)
            columns1 = check_predicted_column_values(blackPixProjProfile1, standardized_png_path1, height1) # stand png path var????
            col1Rows, col2Rows, col3Rows = findTextRows(OCR_Data1, columns1, coursesHeaderRows[0])
        if len(coursesHeaderRows) > 1:
            bottomlessImg2 = remove_img_bottom(toplessImg2, coursesHeaderRows[-1], height1, width1)
            projection_profile2 = np.sum(bottomlessImg2, axis=0)
            blackPixProjProfile2 = createBlackPixelProjectionProfile(projection_profile2)
            columns2 = check_predicted_column_values(blackPixProjProfile2, standardized_png_path2, height2)
            col4Rows, col5Rows, col6Rows = findTextRows(OCR_Data2, columns2, coursesHeaderRows[-1])
        else:
            col4Rows, col5Rows, col6Rows = [], [], []
        # cv2.imshow("bottomless image", bottomlessImg)
        # cv2.waitKey(0)  

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
        for row in col4Rows:
            rows.append(row)
            print(f"/t{row}")
        for row in col5Rows:
            rows.append(row)
            print(f"/t{row}")
        for row in col6Rows:
            rows.append(row)
            print(f"/t{row}")
        return rows, OCR_Data_Path1, OCR_Data_Path2








    print("processing transcript " + str(filename) + " in folder " + str(input_folder_path))

    file_path = os.path.join(input_folder_path, filename)
    file_extension = os.path.splitext(filename)[1]


    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".png":
        rows, OCR_Data_Path1, OCR_Data_Path2 = process_other()
    if file_extension == ".pdf":
        rows, OCR_Data_Path1, OCR_Data_Path2 = process_pdf()
    elif file_extension in [".jpg", ".jpeg"]:
        rows, OCR_Data_Path1, OCR_Data_Path2 = process_other()
    elif file_extension == ".tiff":
        raise ValueError(f"Unimplemented file type: {file_extension}")

    
    
  

    # celdt_detected, dates, scores, score_types = check_CELDT_status(rows)
    celdt_detected, confirmed_rows, elpac_detected, elpac_rows = check_CELDT_status(rows)

    print("still going 1")
    remove_temporary_files()
    print("still going 1.1")
    return celdt_detected, confirmed_rows, elpac_detected, elpac_rows    # dates, scores, score_types




def process_images_in_folder(folder_path):
    print("processing all images in " + str(folder_path) + " folder")
    # scan in image
     # List all files in the folder
    
    output_text_file_path = "text_output.txt"
    output_csv_file_path = "csv_output.csv"

    with open(output_text_file_path, "w") as text_file, open(output_csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        text_file.write("English Learner Statuses: \n\n")
        for filename in os.listdir(folder_path):
            try:
                print()
                print(filename)
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                
                # celdt_detected, dates, scores, score_types = process_image(filename, folder_path)
                celdt_detected, confirmed_rows, elpac_detected, elpac_rows = process_image(filename, folder_path)
                print("still going 2")
                # for match in matches:
                #     print(match)
                text_file.write(f"{filename}\n\tCELDT results found = {celdt_detected}\n")
                text_file.write(f"\tELPAC results found = {elpac_detected}\n")
                print("still going 3")


                if elpac_detected or celdt_detected:
                    text_file.write("\t details:\n")

                if elpac_detected == True:
                    text_file.write(f"\t\tELPAC data\n")
                    print("elpac detected")
                    for row in elpac_rows:
                        text_file.write(f"\t\t{row}\n")
                if celdt_detected == True: # and dates and scores and score_types:
                    print("celdt_detected")
                    text_file.write(f"\t\tCELDT data\n")
                    for row in confirmed_rows:
                        text_file.write(f"\t\t{row}\n")
                
                
                    print("still going 7")
                text_file.flush()
                print("still going 8")
                csv_writer.writerow([filename, (celdt_detected or elpac_detected)])
                # Flush to ensure data is written to disk
                csv_file.flush()
                print(f"finished with {filename}")
                print("still going 9")
                

                # fileName = os.path.basename(remove_extension(filename))
                # redacted_img_path = "Redacted/" + str(fileName) + ".png"
                
                # save_image(redacted_image, redacted_img_path)
            except:
                print(f"some problem in processing {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run OCR on images.")
    parser.add_argument('command', choices=['run'], help="The command to run.")
    parser.add_argument('target', help="The target to process: 'all' for all images in the folder or the specific image filename.")
    parser.add_argument('folder', help="The path to the folder containing images.")
    
    args = parser.parse_args()
    
    if args.command == 'run':
        if args.target == 'all':
            process_images_in_folder(args.folder)
        else:
            print("Error: Can only run for all files. haven't implemented single file yet")

if __name__ == "__main__":
    main()