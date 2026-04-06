# import re

# def check_CELDT_ELPAC_status(rows):

#     print("Checking for CELDT status and ELPAC data...")

#     celdt_confirmed_rows = []
#     celdt_detected = False
#     elpac_detected = False
#     elpac_rows = []
#     elpac_data_remaining = 0
#     for row in rows:
#         current_row = ""
#         for text in row["text"]:
#             current_row += " " +str(text)
#         if "CELDT" in current_row:
#             celdt_detected = True
#             celdt_confirmed_rows.append(current_row)


#         # ELPAC logic
#         if elpac_data_remaining > 0:
#             elpac_rows.append(current_row)
#             elpac_data_remaining = elpac_data_remaining - 1
#         if "ELPAC" in current_row:
#             elpac_detected = True
#             elpac_data_remaining = 3
   
#     return celdt_detected, celdt_confirmed_rows, elpac_detected, elpac_rows


import re

def check_CELDT_ELPAC_status(rows):

    print("Checking for CELDT status and ELPAC data...")

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
            celdt_detected = True
            celdt_confirmed_rows.append(current_row)


        # ELPAC logic
        if elpac_data_remaining > 0:
            elpac_rows.append(current_row)
            elpac_data_remaining = elpac_data_remaining - 1
        if "ELPAC" in current_row:
            elpac_detected = True
            elpac_data_remaining = 3
   
    return celdt_detected, celdt_confirmed_rows, elpac_detected, elpac_rows



def Generic_CELDT_ELPAC_String_Check(OCR_Data):


    CELDT_Found = False
    ELPAC_Found = False

    for chunk in OCR_Data:
        print("check for celdt and elpac string")
        if "CELDT".lower() in chunk["text"].lower():
            CELDT_Found = True
        if "ELPAC".lower() in chunk["text"].lower():
            ELPAC_Found = True
    
    return CELDT_Found, ELPAC_Found
