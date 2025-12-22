import re

def check_CELDT_status(rows):

    print("Checking for CELDT status...")

    celdt_confirmed_rows = []
    celdt_detected = False
    elpac_detected = False
    elpac_rows = []
    elpac_data_remaining = 0
    for row in rows:
        current_row = ""
        for text in row["text"]:
            current_row += " " +str(text)
        if "CELDT" in current_row:
            print("************** CELDT DETECTED !!! *****************")
            celdt_detected = True
            print("regular expression for date: dd/dd")
            celdt_confirmed_rows.append(current_row)


        # ELPAC logic
        if elpac_data_remaining > 0:
            elpac_rows.append(current_row)
            elpac_data_remaining = elpac_data_remaining - 1
        if "ELPAC" in current_row:
            elpac_detected = True
            elpac_data_remaining = 3


    print("still going *initial*")    
    return celdt_detected, celdt_confirmed_rows, elpac_detected, elpac_rows