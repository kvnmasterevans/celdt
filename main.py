import os
import csv
import argparse
import json
from utils import process_image, USE_NEW_COLUMN_ALGORITHM
import traceback
import time
import numpy as np
import sys

def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type == IndexError:
        tb = traceback.extract_tb(exc_traceback)
        # Get the last frame (where the error happened)
        filename, lineno, func, text = tb[-1]
        print(f"\n❌ IndexError in {filename}, line {lineno}: {text}")
        print(f"  locals: {exc_traceback.tb_frame.f_locals}\n")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = custom_excepthook


def process_images_in_folder(folder_path):
    print("processing all images in " + str(folder_path) + " folder")
    print(f"Using new algorithm is set to {USE_NEW_COLUMN_ALGORITHM}")
    # scan in image
     # List all files in the folder
    
    output_text_file_path = "text_output.txt"
    output_csv_file_path = "csv_output.csv"

    with open(output_text_file_path, "w") as text_file, open(output_csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        text_file.write("English Learner Statuses: \n\n")
        for filename in os.listdir(folder_path):
            try:
                start = time.perf_counter()
                print()
                print(filename)
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                
                # celdt_detected, dates, scores, score_types = process_image(filename, folder_path)
                celdt_detected, confirmed_celdt_rows, elpac_detected, elpac_rows, transfer_worksheet_found, entry_date, exit_date, celdt_date, elpac_date = process_image(filename, folder_path)
                print("still going 2")
                print(f"Dates:    entry: {entry_date}, exit: {exit_date}, elpac: {elpac_date}, celdt: {celdt_date}")
                # for match in matches:, entry_date, exit_date
                #     print(match)
                text_file.write(f"{filename}\n\tTransfer Admission Worksheet = {transfer_worksheet_found}\n")
                text_file.write(f"{filename}\n\tCELDT results found = {celdt_detected}\n")
                text_file.write(f"\tELPAC results found = {elpac_detected}\n")
                print("still going 3")

                csv_writer.writerow([filename, f" CELDT or ELPAC Detected = {(celdt_detected or elpac_detected)}"])
                if elpac_detected or celdt_detected:
                    text_file.write("\t details:\n")

                if elpac_detected == True:
                    text_file.write(f"\t\tELPAC data\n")
                    print("elpac detected")
                    for row in elpac_rows:
                        text_file.write(f"\t\t{row}\n")
                        csv_writer.writerow([filename, row])
                print(f"CELDT Detected = {celdt_detected}")
                if celdt_detected == True: # and dates and scores and score_types:
                    print("celdt_detected")
                    text_file.write(f"\t\tCELDT data\n")
                    for row in confirmed_celdt_rows:
                        text_file.write(f"\t\t{row}\n")
                        csv_writer.writerow([filename, row])


                if entry_date != None:
                    print("ENTRY DATE FOUND!!!!   " + entry_date)
                    csv_writer.writerow([filename, f" entry date: {entry_date}"])
                else:
                    csv_writer.writerow([filename, " entry date not found"])
                if exit_date != None:
                    print("EXIT DATE FOUND!!!!   " + exit_date)
                    csv_writer.writerow([filename, f" exit date: {exit_date}"])
                else:
                    csv_writer.writerow([filename, " exit date not found"])
                
                
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
                traceback.print_exc()
            finally:
                elapsed = time.perf_counter() - start
                text_file.write(f"{filename} took {elapsed:.3f} seconds to complete")

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