import os
import csv
import argparse
import json
from utils import process_image
        
# from rowUtilsNew import findTextRows
# from FinalizeColumns import check_predicted_column_values
# from check_for_CELDT import check_CELDT_status
# import cv2
import numpy as np
import sys




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

                csv_writer.writerow([filename, (celdt_detected or elpac_detected)])
                if elpac_detected or celdt_detected:
                    text_file.write("\t details:\n")

                if elpac_detected == True:
                    text_file.write(f"\t\tELPAC data\n")
                    print("elpac detected")
                    for row in elpac_rows:
                        text_file.write(f"\t\t{row}\n")
                        csv_writer.writerow([filename, row])
                if celdt_detected == True: # and dates and scores and score_types:
                    print("celdt_detected")
                    text_file.write(f"\t\tCELDT data\n")
                    for row in confirmed_rows:
                        text_file.write(f"\t\t{row}\n")
                        csv_writer.writerow([filename, row])
                
                
                    print("still going 7")
                text_file.flush()
                print("still going 8")
                
                # Flush to ensure data is written to disk
                csv_file.flush()
                print(f"finished with {filename}")
                print("still going 9")
                

                # fileName = os.path.basename(remove_extension(filename))
                # redacted_img_path = "Redacted/" + str(fileName) + ".png"
                
                # save_image(redacted_image, redacted_img_path)
            except KeyboardInterrupt:
                print("Interrupted by user. Exiting.")
                raise  # ← this re-raises it so the program stops
            except Exception as e:
                print(f"some problem in processing {filename}")
                print(f"Error: {e}")


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
    with open("log_output.txt", "w", buffering=1) as f:
        sys.stdout = f
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
            print("finished running")