import nltk
nltk.download('punkt')
from difflib import SequenceMatcher




# approximate comparison function
def withinTolerance(firstVal, secondVal, tolerance):
    if abs(firstVal - secondVal) < tolerance:
        return True
    else:
        return False
    



def _withinPotentialColumnRange(detectedText, course_id_line, LEdge, REdge):
    # if below top cut-off line...
    # ... and to left of first 1/3 column marker...
    if detectedText["bounding_box"][0]["y"] > course_id_line \
    and detectedText["bounding_box"][0]["x"] < REdge \
    and detectedText["bounding_box"][0]["x"] > LEdge  :    # ------------------- was index 1 -> changed it to check LEFT side instead  
                                                                # ---------------------could possibly check for CENTER of chunk instead?                      
        return True
    else:
        return False  

# checks if text chunk from OCR read overhangs the next column
def _TextOverhangsNextColumn(text_box, right_column_edge):
    if text_box["bounding_box"][1]["x"] > right_column_edge:
        return True
    else:
        return False

# approximates the portion of string contained within
# the column range by using a ratio between the bounding box length
# and the distance from the left-side of the bounding box
# to the right-side column boundary
# returns the shortened string
def _split_extruding_text(text_box, right_column_edge):

    box_length = text_box["bounding_box"][1]["x"] - text_box["bounding_box"][0]["x"]
    box_length_contained_within_column = right_column_edge - text_box["bounding_box"][0]["x"]
    if box_length > 0:
        ratio_of_box_contained_in_column = box_length_contained_within_column / box_length
    else:
        return "", ""
    string_length = len(text_box["text"])
    number_of_chars_in_preserved_string = int( string_length * ratio_of_box_contained_in_column )

    remaining_string = text_box["text"][:number_of_chars_in_preserved_string]

    excess_string = text_box["text"][number_of_chars_in_preserved_string:]

    return remaining_string, excess_string


def find_rows(detectedText, left_string_segment):
    possible_rows = []
    rowThreshold = 20
    rowCenter = ( detectedText["bounding_box"][0]["y"] + detectedText["bounding_box"][3]["y"] ) / 2
    rowDetected = False
    for poss_row in possible_rows:
        if (rowCenter < (poss_row["row"] + rowThreshold) )and (rowCenter >  (poss_row["row"] - rowThreshold) ): # matches a known row

            rowDetected = True
            if len(left_string_segment) > 0:
                poss_row["x's"].append( detectedText["bounding_box"][0]["x"] ) # store necessary data.... y is row,now we need x vals???
                poss_row["text"].append(left_string_segment )
            break
            # if found do logic to add and then 'break'

    # if not found then add to rows    
    if rowDetected == False:
        # this part needs to change
        if len(left_string_segment) > 0:
            possible_rows.append({"row":rowCenter, "x's":[ detectedText["bounding_box"][0]["x"] ], \
                                    "text": [ left_string_segment ]}) # when row not found (new row)
            # num_poss_rows_col1 += 1 # not necessarily used....





# def extract_entry_and_exit_dates(json_ocr_data, course_id_line, entry_date_coord, exit_date_coord):
#     print("entry and exit dates from x_coordinate...")
#     rowThreshold = 20  # ??? need to choose appropriate threshold
#     possible_rows = [] # should this be a list of dicts with another list embedded in dict to store x pos's??? - yes??? 
#     # [{"row": num, "x's": [num, num, num]}]
#     num_poss_rows_col1 = 0
#     for detectedText in json_ocr_data:
#         # determine location of rows: 
#         # find possible rows & store the x-pattern data
        
#         # find rows:
#         rowCenter = ( detectedText["bounding_box"][0]["y"] + detectedText["bounding_box"][3]["y"] ) / 2
#         rowDetected = False
#         for poss_row in possible_rows:
#             if (rowCenter < (poss_row["row"] + rowThreshold) )and (rowCenter >  (poss_row["row"] - rowThreshold) ): # matches a known row

#                 rowDetected = True
#                 if len(detectedText["text"]) > 0:
#                     poss_row["x's"].append( [ detectedText["bounding_box"][0]["x"], detectedText["bounding_box"][1]["x"] ] ) # store necessary data.... y is row,now we need x vals???
#                     poss_row["text"].append(detectedText["text"] )
#                 break
#                 # if found do logic to add and then 'break'

#         # if not found then add to rows    
#         if rowDetected == False:
#             # this part needs to change
#             if len(detectedText["text"]) > 0:
#                 possible_rows.append({"row":rowCenter, "x's":[ detectedText["bounding_box"][0]["x"], detectedText["bounding_box"][1]["x"] ], \
#                                         "text": [ detectedText["text"] ]}) # when row not found (new row)
#                 num_poss_rows_col1 += 1

#     for row in possible_rows:
#         if row




# finds rows of text within the first column below the course-id line 
# (*though goes to bottom of document, because column bottoms not yet known)
# assembles all text on each line into a single 'row' value
# along with the bounding box x values for each chunk of text within each row
# (*currently only stores LEFT side of each bounding box - may need to include right sides as well...)


def findTextRows(json_ocr_data, columnPositions, course_id_line):


    colEdge1 = columnPositions[0]
    colEdge2 = columnPositions[1]
    colEdge3 = columnPositions[2]
    colEdge4 = columnPositions[3]

    # determine location of rows: 
    # find possible rows & store the x-pattern data
    rowThreshold = 20  # ??? need to choose appropriate threshold
    possible_rows_col1 = [] # should this be a list of dicts with another list embedded in dict to store x pos's??? - yes??? 
    # [{"row": num, "x's": [num, num, num]}]
    num_poss_rows_col1 = 0

    overhanging_text_boxes_1 = []
    # First Major Column
    for detectedText in json_ocr_data:
        if _withinPotentialColumnRange(detectedText, course_id_line, colEdge1, colEdge2):

            # CHECK IF HANGING OVER SECOND COLUMN AND CUTOFF EXTRA----------------------
            if _TextOverhangsNextColumn(detectedText, colEdge1):
                # cutoff extra:
                left_string_segment, right_string_segment = _split_extruding_text(detectedText, colEdge2)
                bounding_box = [ {"x": colEdge2 + 1, "y": box["y"]} if i in [0, 3] else box \
                    for i, box in enumerate(detectedText["bounding_box"]) ]
                # bounding_box = [{"x": , "y":},
                #                 {},
                #                 {},
                #                 {}]
                separated_row = {"text": right_string_segment, 
                                 "bounding_box": bounding_box, 
                                 "confidence": detectedText["confidence"]}
                overhanging_text_boxes_1.append(separated_row)

            else:
                left_string_segment = detectedText["text"]
                right_string_segment = ""


            # find rows:
            rowCenter = ( detectedText["bounding_box"][0]["y"] + detectedText["bounding_box"][3]["y"] ) / 2
            rowDetected = False
            for poss_row in possible_rows_col1:
                if (rowCenter < (poss_row["row"] + rowThreshold) )and (rowCenter >  (poss_row["row"] - rowThreshold) ): # matches a known row

                    rowDetected = True
                    if len(left_string_segment) > 0:
                        poss_row["x's"].append( detectedText["bounding_box"][0]["x"] ) # store necessary data.... y is row,now we need x vals???
                        poss_row["text"].append(left_string_segment )
                    break
                    # if found do logic to add and then 'break'

            # if not found then add to rows    
            if rowDetected == False:
                # this part needs to change
                if len(left_string_segment) > 0:
                    possible_rows_col1.append({"row":rowCenter, "x's":[ detectedText["bounding_box"][0]["x"] ], \
                                            "text": [ left_string_segment ]}) # when row not found (new row)
                    num_poss_rows_col1 += 1

#
##
###                  COLUMN 2           
##
#
    overhanging_text_boxes_2 = []
    possible_rows_col2 = []
    num_poss_rows_col2 = 0 # unused???
    for text_chunk in overhanging_text_boxes_1:
        json_ocr_data.append(text_chunk)
    for detectedText in json_ocr_data:
        if _withinPotentialColumnRange(detectedText, course_id_line, colEdge2, colEdge3):

            # CHECK IF HANGING OVER SECOND COLUMN AND CUTOFF EXTRA----------------------
            if _TextOverhangsNextColumn(detectedText, colEdge2):
                # cutoff extra:
                left_string_segment, right_string_segment = _split_extruding_text(detectedText, colEdge3)
                bounding_box = [ {"x": colEdge2 + 1, "y": box["y"]} if i in [0, 3] else box \
                    for i, box in enumerate(detectedText["bounding_box"]) ]
                # bounding_box = [{"x": , "y":},
                #                 {},
                #                 {},
                #                 {}]
                separated_row = {"text": right_string_segment, 
                                 "bounding_box": bounding_box, 
                                 "confidence": detectedText["confidence"]}
                overhanging_text_boxes_2.append(separated_row)

            else:
                left_string_segment = detectedText["text"]
                right_string_segment = ""


            # find rows:
            rowCenter = ( detectedText["bounding_box"][0]["y"] + detectedText["bounding_box"][3]["y"] ) / 2
            rowDetected = False
            for poss_row in possible_rows_col2:
                if (rowCenter < (poss_row["row"] + rowThreshold) )and (rowCenter >  (poss_row["row"] - rowThreshold) ): # matches a known row

                    rowDetected = True
                    if len(left_string_segment) > 0:
                        poss_row["x's"].append( detectedText["bounding_box"][0]["x"] ) # store necessary data.... y is row,now we need x vals???
                        poss_row["text"].append(left_string_segment )
                    break
                    # if found do logic to add and then 'break'

            # if not found then add to rows    
            if rowDetected == False:
                # this part needs to change
                if len(left_string_segment) > 0:
                    possible_rows_col2.append({"row":rowCenter, "x's":[ detectedText["bounding_box"][0]["x"] ], \
                                            "text": [ left_string_segment ]}) # when row not found (new row)
                    num_poss_rows_col2 += 1


#
##
###                  COLUMN 3           
##
#
    overhanging_text_boxes_3 = []
    possible_rows_col3 = []
    num_poss_rows_col3 = 0 # unused???
    for text_chunk in overhanging_text_boxes_2:
        json_ocr_data.append(text_chunk)
    for detectedText in json_ocr_data:
        if _withinPotentialColumnRange(detectedText, course_id_line, colEdge3, colEdge4):

            # CHECK IF HANGING OVER SECOND COLUMN AND CUTOFF EXTRA----------------------
            if _TextOverhangsNextColumn(detectedText, colEdge3):
                # cutoff extra:
                left_string_segment, right_string_segment = _split_extruding_text(detectedText, colEdge4)
                bounding_box = [ {"x": colEdge3 + 1, "y": box["y"]} if i in [0, 3] else box \
                    for i, box in enumerate(detectedText["bounding_box"]) ]
                # bounding_box = [{"x": , "y":},
                #                 {},
                #                 {},
                #                 {}]
                separated_row = {"text": right_string_segment, 
                                 "bounding_box": bounding_box, 
                                 "confidence": detectedText["confidence"]}
                overhanging_text_boxes_3.append(separated_row)

            else:
                left_string_segment = detectedText["text"]
                right_string_segment = ""


            # find rows:
            rowCenter = ( detectedText["bounding_box"][0]["y"] + detectedText["bounding_box"][3]["y"] ) / 2
            rowDetected = False
            for poss_row in possible_rows_col3:
                if (rowCenter < (poss_row["row"] + rowThreshold) )and (rowCenter >  (poss_row["row"] - rowThreshold) ): # matches a known row

                    rowDetected = True
                    if len(left_string_segment) > 0:
                        poss_row["x's"].append( detectedText["bounding_box"][0]["x"] ) # store necessary data.... y is row,now we need x vals???
                        poss_row["text"].append(left_string_segment )
                    break
                    # if found do logic to add and then 'break'

            # if not found then add to rows    
            if rowDetected == False:
                # this part needs to change
                if len(left_string_segment) > 0:
                    possible_rows_col3.append({"row":rowCenter, "x's":[ detectedText["bounding_box"][0]["x"] ], \
                                            "text": [ left_string_segment ]}) # when row not found (new row)
                    num_poss_rows_col3 += 1


    





    # sort rows:
    # Sort each row's x values and reorder the text accordingly
    for row in possible_rows_col1:
        combined = list(zip(row["x's"], row["text"]))
        combined.sort(key=lambda pair: pair[0])
        row["x's"], row["text"] = zip(*combined)
        row["x's"] = list(row["x's"])
        row["text"] = list(row["text"])


    for row in possible_rows_col2:
        combined = list(zip(row["x's"], row["text"]))
        combined.sort(key=lambda pair: pair[0])
        row["x's"], row["text"] = zip(*combined)
        row["x's"] = list(row["x's"])
        row["text"] = list(row["text"])

    for row in possible_rows_col3:
        combined = list(zip(row["x's"], row["text"]))
        combined.sort(key=lambda pair: pair[0])
        row["x's"], row["text"] = zip(*combined)
        row["x's"] = list(row["x's"])
        row["text"] = list(row["text"])

    return possible_rows_col1, possible_rows_col2, possible_rows_col3


#
#
#
#
#
#
#
#
#                                       DETERMINE BOTTOMS
#
#
#
#
#
#
#
#
#
#

# accepts 2 x-patterns and checks if they are
# similar enough to count as matched
def _comparePatterns(list1, list2, debugging=False):
    thresh = 12 # pixel threshold for comparisons - 12 - 25 too much

    # ---------------------------------------------------------
    len_1 = len(list1)
    len_2 = len(list2)

    iter_1 = 0
    iter_2 = 0
    continue_loop = True
    loopcounter = 0
    while (iter_1 < len_1) and (iter_2 < len_2) and continue_loop:
        # if list 1 item greater than lower threshold and smaller than higher threshold 
        # aka both values match (within threshold)
        if (list2[iter_2] - thresh <= list1[iter_1] and list1[iter_1] <= list2[iter_2] + thresh):
            iter_1 += 1
            iter_2 += 1

        # if not a match then check if text chunks have been combined 
        # and if so check for a match given those conditions
        elif (list2[iter_2] - thresh > list1[iter_1]):

            if iter_1 + 1 < len_1:
                combined_val = list1[iter_1] + list1[iter_1 + 1]

                if (list2[iter_2] - thresh * 2 <= combined_val and combined_val <= list2[iter_2] + thresh * 2):
                    iter_1 += 2
                    iter_2 += 1

                else:
                    return False
            else:
                return False
                
        # checking the other list for combined values...
        elif list1[iter_1] > list2[iter_2] + thresh:
            if iter_2 + 1 < len_2:
                combined_val = list2[iter_2] + list2[iter_2 + 1]
                if (combined_val - thresh <= list1[iter_1] and list1[iter_1] <= combined_val + thresh):
                    iter_1 += 1
                    iter_2 += 2
                else:
                    return False
            else:
                return False
    # check conditions like iter1 and iter2 matching?????
    if (iter_1 == len_1) and (iter_2 == len_2):
        return True 
    else:
        return False  
    



# accepts 2 strings and compares their similarity
def _compareStrings(string1, string2):
    tokens1 = nltk.word_tokenize(string1.lower())
    tokens2 = nltk.word_tokenize(string2.lower())
    match = SequenceMatcher(None, tokens1, tokens2)
    similarity = match.ratio()
    return similarity


# determines the x position pattern for the x vals of a row
# as the difference between each x value as well
# as the difference between the rightmost x value and
# the right side of the column for the final value in the pattern
def _findXPattern(xVals, columnREdge):
    pattern = []
    for i in range(1, len(xVals)):
        xDiff = xVals[i] - xVals[i-1]
        pattern.append(xDiff)
        if i == len(xVals) - 1: # if final value append distance to column edge
            columnDiff = columnREdge - xVals[i]
            pattern.append(columnDiff)
    return pattern

# accepts 2 lines(*should create line object...)
# and checks whether they are similar enough to match or not
def _compareRows(row1, row2, right_column_edge, debugging = False):
    
    
    STRING_SIMILARITY_THRESHOLD = .4
    if len(row1["x's"]) == 1:
        string1 = " ".join(row1["text"])
        string2 = " ".join(row2["text"])
        similarity = _compareStrings(string1, string2)
        x1 = row1["x's"][0]
        x2 = row2["x's"][0]
        x_similarity = withinTolerance(x1, x2, 12)
        if (similarity > STRING_SIMILARITY_THRESHOLD) and (len(string1) > 0) and x_similarity:
            return True
        else:
            return False

    else:
        pattern1 = _findXPattern(row1["x's"], right_column_edge)
        pattern2 = _findXPattern(row2["x's"], right_column_edge)
        return _comparePatterns(pattern1, pattern2)


# might need to run check to see if they're bleeding text across columns....
def check_header_rows_2_and_3(header_row_pos, column_edges, OCR_Data):
    CourseStrings = ["course", "crs", "coun5", "coursc", "courec", "cour60", "cotise", "couno", "courso"]
    header1 = ""
    header2 = ""
    header3 = ""
    head2matches = False
    head3matches = False
    for data in OCR_Data:
        if withinTolerance(data["bounding_box"][0]["y"], header_row_pos, 15):
            if data["bounding_box"][0]["x"] < column_edges[1]:  
                header1 += " " + str(data["text"])
            elif data["bounding_box"][0]["x"] < column_edges[2]:
                header2 += " " + str(data["text"])
            elif data["bounding_box"][0]["x"] < column_edges[3]:
                header3 += " " + str(data["text"])
            else:
                print(" ~ ERROR MATCHING HEADER ROW ~ ")

    for string in CourseStrings:
        if string in header2.lower():
            head2matches = True
        if string in header3.lower():
            head3matches = True

    return head2matches, head3matches 




















#
#
#
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~ APPLY ROW BOT SEARCH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
#
#
#
#   





# finds rows with patterns matching the course data
# col1Rows is the rows found in findTextRows()
# such that rows are beneath the "course ID" courses_header_line
# and contained within the width of the row
def findMatchingRowPatterns(colRows, courses_header_line, column_right_edge):



    # find all rows with patterns matching the first 5 rows
    def find_initial_matching_pattern_indexes():
        pat1Matches = []
        for i, row in enumerate(colRows): # will always detect first rows...
            if _compareRows(row, row1, column_right_edge) \
                or _compareRows(row, row2, column_right_edge) \
                or _compareRows(row, row3, column_right_edge) \
                or _compareRows(row, row4, column_right_edge) \
                or _compareRows(row, row5, column_right_edge):
                pat1Matches.append(i)
        return pat1Matches

    # uses the list of indexes which match the initial patterns
    # from find_initial_matching_pattern_indexes() in order to
    # find lowest pattern-1 match's y coordinate 
    # (which is the final index)
    def find_lowest_matching_coordinate_from_matching_indexes():
        
        if len(colRows) > 0 and len(pat1MatchIndexes) > 0:
            lowestPat1 = colRows[pat1MatchIndexes[-1]]["row"]
        else:
            print("ERROR: no pattern 1 matches found (rowUtils.py findMatchingRowPatterns())")
            print("or no pat1MatchIndexes")
        return lowestPat1

    # find internal patterns between top and lowest matched patterns
    def find_encapsulated_patterns(lowestPattern):  
        foundPatterns = []
        for row in colRows:
            patternFound = False
            for pattern in foundPatterns:
                if _compareRows(pattern, row, column_right_edge):
                    patternFound = True
            if patternFound == False:
                foundPatterns.append(row) # append row instead
            if row["row"] == lowestPattern:
                break
        return foundPatterns



    def find_largest_course_data_vertical_gap_between_lines():
        largest_gap = 0
        for i, row in enumerate(colRows):
            if row["row"] < lowestPat1:
                if i > 0:
                    row_difference = abs(row["row"] - colRows[i-1]["row"])
                    if row_difference > largest_gap:
                        largest_gap = row_difference
            else:
                break
        return largest_gap
                    



    # using found patterns determine the bottom of column 1
    # reaturns lowest matched pattern
    def find_row_bottom(right_boundary, pattern_types):
        largest_vertical_gap = find_largest_course_data_vertical_gap_between_lines()
        unfound_counter = 0
        row1Bottom = courses_header_line # initialize row1Bottom to top of possible course lines
        for i, row in enumerate(colRows):
            patternFound = False
            if i == 0:
                dist_from_header = abs(row["row"] - courses_header_line)
                if dist_from_header > largest_vertical_gap * 5:
                    return courses_header_line
            if i > 0:
                row_diff = abs(colRows[i]["row"] - colRows[i-1]["row"])
                if row_diff > (largest_vertical_gap * 1.2):
                    break

            for j, pat in enumerate(pattern_types):
                if _compareRows(pat, row, right_boundary):
                    patternFound = True
                    break

            if patternFound:
                unfound_counter = 0
                if i > 0:
                    row1Bottom = colRows[i]["row"]
            else:
                unfound_counter += 1
                _compareRows(pat, row, right_boundary, True) # just to display print comparisons within function...
                if unfound_counter >= 3: # if too many non-matches then quit looking for lower match to not find abberant match by accident
                    return row1Bottom

        return row1Bottom







    row1 = colRows[0]
    row2 = colRows[1]
    row3 = colRows[2]
    row4 = colRows[3]
    row5 = colRows[4]
    first_patterns = [row1, row2, row3, row4, row5]
    
    pat1MatchIndexes = find_initial_matching_pattern_indexes()


    lowestPat1 = find_lowest_matching_coordinate_from_matching_indexes()

    pattern_types = find_encapsulated_patterns(lowestPat1)

    row1Bottom = find_row_bottom(column_right_edge, pattern_types)

    # run algorithm again with new row1 bottom to find MORE course data patterns
    expanded_pattern_types = find_encapsulated_patterns(row1Bottom)

    row_1_bottom = find_row_bottom(column_right_edge, expanded_pattern_types)            

    return row1Bottom