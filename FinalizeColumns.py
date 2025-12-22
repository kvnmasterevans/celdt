# FinalizeColumns.py
# from productionUtils import displayColumnEdges as showEdges
# from utils import openJpgImage
import cv2

def openJpgImage(jpg_path):
    image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    return image

#To Do:
# make tolerance a static file-wide variable



# approximate comparison function
def withinTolerance(firstVal, secondVal, tolerance):
    if abs(firstVal - secondVal) < tolerance:
        return True
    else:
        return False
    

def initialCheck(potentialColumns):

    diff1 = abs(potentialColumns[0] - potentialColumns[1])
    diff2 = abs(potentialColumns[1] - potentialColumns[2])
    diff3 = abs(potentialColumns[2] - potentialColumns[3])
    if withinTolerance(diff1, diff2, 10): # was .6 tolerance
        if withinTolerance(diff2, diff3, 10):
            return True
        
    else:
        return False
    

# check (*approximates) local maximum heuristic (3 away in each direction)
def columnConfirm(index, proj_profile):
    def greater_than_neighboring_indexes():
        print("cycle through indexes")
    profile_len = len(proj_profile)

    upperCheckValid = False
    lowerCheckValid = False
    columnConfirmed = False

    if index + 3 < profile_len:
        if proj_profile[index] > proj_profile[index+3]:
            upperCheckValid = True
    else:
        upperCheckValid = True


    if index - 3 >= 0:
        if proj_profile[index] > proj_profile[index-3]:
            lowerCheckValid = True
    else:
        lowerCheckValid = True


    if upperCheckValid and lowerCheckValid:
        columnConfirmed = True

    return columnConfirmed


# actually does nothing because its all commented out ...
# looks like the intention was to collect the average location and standard deviation
# of each of the quadrants' possible lines, and then
# check if values differed by more than 2 standard deviations from those averages
# but I suppose wasn't useful because its all commented out
def columnConfirm2(index1, index2, index3, index4):
    # def checkColumnDiffs():
    #     if diff1 < abs(avg1 - 2*stDev1) \
    #         and diff2 < abs(avg2 - 2*stDev2) \
    #         and diff3 < abs(avg3 - 2*stDev3) \
    #         and diff4 < abs(avg4 - 2*stDev4):
    #         return True
    #     else:
    #         return False
        
    # avg1 = 213
    # avg2 = 913
    # avg3 = 1613
    # avg4 = 2311

    # stDev1 = 110
    # stDev2 = 40
    # stDev3 = 51
    # stDev4 = 120

    # diff1 = abs(index1 - avg1)
    # diff2 = abs(index2 - avg2)
    # diff3 = abs(index3 - avg3)
    # diff4 = abs(index4 - avg4)




    # return checkColumnDiffs()
    return True


# checks local maximum 625 indices in each direction 
# (if those indices exist)
def columnConfirm3(index, proj_profile):
    def greater_than_neighboring_indexes():
        print("cycle through indexes")
    profile_len = len(proj_profile)

    upperCheckValid = True
    lowerCheckValid = True
    columnConfirmed = False


    for i in range(625):
        if index + i < profile_len:
            if proj_profile[index] < proj_profile[index + i]:
                upperCheckValid = False



    for i in range(625):
        if index - i >=0 and index < ( len(proj_profile) - 1 ):
            try:
                # print(f" test first index {proj_profile[index - i]} ")
                # print(f" test second index {proj_profile[index]} ")
                if proj_profile[index] < proj_profile[index - i]:
                    upperCheckValid = False
            except IndexError:
                print(f"IndexError at index={index}, i={i}, len={len(proj_profile)}")
                raise  # re-raise to stop or show traceback if you want


    if upperCheckValid and lowerCheckValid:
        columnConfirmed = True

    return columnConfirmed
    


    



# original_jpg_path and height parameters are temporary for testing!
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


# Takes black pixel projection profile
def check_predicted_column_values(proj_profile, original_jpg_path, height):

    # For storing the pixel values within each quadrant of the image
    Quad1 = []
    Quad2 = []
    Quad3 = []
    Quad4 = []

    profile_length = len(proj_profile)

        # set initial elements in possible_columns
    for i, pixelcount in enumerate(proj_profile):
        if pixelcount > 0:
            if i < profile_length / 4:
                Quad1.append([pixelcount, i])
            if (i < profile_length / 2) and (i >= profile_length / 4):
                Quad2.append([pixelcount, i])
            if (i < profile_length * 3 / 4) and (i >= profile_length / 2):
                Quad3.append([pixelcount, i])
            if (i >= profile_length * 3 / 4):
                Quad4.append([pixelcount, i])

    print("Profile enumerated over ...")

    # how to store unsorted?
    orig1 = Quad1
    orig2 = Quad2
    orig3 = Quad3
    orig4 = Quad4

    # sort columns:
    Quad1.sort(reverse = True, key = lambda x: x[0])
    Quad2.sort(reverse = True, key = lambda x: x[0])
    Quad3.sort(reverse = True, key = lambda x: x[0])
    Quad4.sort(reverse = True, key = lambda x: x[0])
    print("Sorted ...")




    tolerance = .5 # tolerance for closeness of comparison values
    # iterators to keep track of progress through "Quad" lists
    # start at 1 so that the first value from each quad aleady has 1 initial element for checking against
    iter1 = 1
    iter2 = 1
    iter3 = 1
    iter4 = 1

    # determine max size of each list
    size1 = len(Quad1)
    size2 = len(Quad2)
    size3 = len(Quad3)
    size4 = len(Quad4)


    originalImg = openJpgImage(original_jpg_path)
    # showEdges(Quad1[0][1], Quad2[0][1], Quad3[0][1], Quad4[0][1], originalImg, height, "initial check")
    if initialCheck([Quad1[0][1], Quad2[0][1], Quad3[0][1], Quad4[0][1]]):
        return [Quad1[0][1], Quad2[0][1], Quad3[0][1], Quad4[0][1]]
    print("Initial check ...")
    def potentialColumnsRemain():
        return iter1 < size1 or iter2 < size2 or iter3 < size3 or iter4 < size4

    while potentialColumnsRemain():



        # set potential next column values if they exist, or -1 if all values have been exhausted
        # the negative 1 value ensures the column cannot be selected from the max() function
        potentialNextVal1 = -1
        potentialNextVal2 = -1
        potentialNextVal3 = -1
        potentialNextVal4 = -1
        if iter1 < size1:
            potentialNextVal1 = Quad1[iter1][0]
        if iter2 < size2:
            potentialNextVal2 = Quad2[iter2][0]
        if iter3 < size3:
            potentialNextVal3 = Quad3[iter3][0]
        if iter4 < size4:
            potentialNextVal4 = Quad4[iter4][0]

        potentialNextColVals = [potentialNextVal1, potentialNextVal2, potentialNextVal3, potentialNextVal4]

        index_of_max, max_value = max(enumerate(potentialNextColVals), key=lambda x: x[1])
        if index_of_max ==0:
            current_x_pos = Quad1[iter1][1]
        if index_of_max ==1:
            current_x_pos = Quad2[iter2][1]
        if index_of_max ==2:
            current_x_pos = Quad3[iter3][1]
        if index_of_max ==3:
            current_x_pos = Quad4[iter4][1]


        # determine which value to check by index_of_max
        if index_of_max == 0: # when value comes from first column
            iter1 += 1 # increment list iterator for current column
            loopIter = 0 # set loop iterator ... for iterating through anchor values????

            # "anchor" with col2 vals
            while loopIter < iter2: # loop through values in anchor column
                diff1 = abs(current_x_pos - Quad2[loopIter][1]) # set "anchor"
                
            #       and check col3 then col4
                i = 0 # set iterator
                while i < iter3: 
                    diff2 = abs(current_x_pos - Quad3[i][1])
                    if withinTolerance(diff1 * 2, diff2, tolerance):
                        # found!
                        # calculate 4th index prediction
                        index1 = current_x_pos
                        index2 = Quad2[loopIter][1]
                        index3 = Quad3[i][1]
                        index4 = index3 + diff1
                        # check if index 4 is greater than surrounding values
                        # if columnConfirm(index4, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm3(index4, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1


                i = 0 # set iterator
                while i < iter4: 
                    diff2 = abs(current_x_pos - Quad4[i][1])
                    if withinTolerance(diff1 * 3, diff2, tolerance):
                        # found!
                        # calculate 3rd index prediction
                        index1 = current_x_pos
                        index2 = Quad2[loopIter][1]
                        index4 = Quad4[i][1]
                        index3 = index2 + diff1
                        # if columnConfirm(index3, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm3(index3, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1

                
                loopIter += 1 # incement anchor loop iterator





        # if max value is from column 2
        if index_of_max == 1:

            iter2 += 1 # increment list iterator for current column
            loopIter = 0 # set loop iterator ... for iterating through anchor values????

            while loopIter < iter1: # loop through values in anchor column
                diff1 = abs(current_x_pos - Quad1[loopIter][1]) # set "anchor" with column 1 vals

                i = 0 # set iterator
                while i < iter3: 
                    diff2 = abs(current_x_pos - Quad3[i][1])
                    if withinTolerance(diff1, diff2, tolerance):
                        # found!
                        # calculate 4th index prediction
                        index1 = Quad1[loopIter][1]
                        index2 = current_x_pos 
                        index3 = Quad3[i][1]
                        index4 = index3 + diff1
                        # if columnConfirm(index4, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm3(index4, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1


                i = 0 # set iterator
                while i < iter4: 
                    diff2 = abs(current_x_pos - Quad4[i][1])
                    if withinTolerance(diff1 * 2, diff2, tolerance):
                        # found!
                        # calculate 3rd index prediction
                        index1 = Quad1[loopIter][1]
                        index2 =  current_x_pos
                        index4 = Quad4[i][1]
                        index3 = index2 + diff1
                        # if columnConfirm(index3, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm3(index3, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1

                
                loopIter += 1 # incement anchor loop iterator





        # when max value is found in column 3
        if index_of_max == 2:
            iter3 += 1 # increment list iterator for current column
            loopIter = 0 # set loop iterator ... for iterating through anchor values????

            # "anchor" with col4 vals
            while loopIter < iter4: # loop through values in anchor column
                diff1 = abs(current_x_pos - Quad4[loopIter][1]) # set "anchor"
                
            #       and check col3 then col4
                i = 0 # set iterator
                while i < iter1: 
                    diff2 = abs(current_x_pos - Quad1[i][1])
                    if withinTolerance(diff1 * 2, diff2, tolerance):
                        # found!
                        # calculate 2nd index prediction
                        index1 = Quad1[i][1]
                        index3 = current_x_pos
                        index4 = Quad4[loopIter][1]
                        index2 = index1 + diff1
                        # if columnConfirm(index2, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm3(index2, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1


                i = 0 # set iterator
                while i < iter2: 
                    diff2 = abs(current_x_pos - Quad2[i][1])
                    if withinTolerance(diff1, diff2, tolerance):
                        # found!
                        # calculate 3rd index prediction
                        
                        index2 = Quad2[i][1]
                        index3 = current_x_pos
                        index4 = Quad4[loopIter][1]
                        index1 =  index2 - diff1
                        # if columnConfirm(index1, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm(index1, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1

                
                loopIter += 1 # incement anchor loop iterator
        
        
        
        
        
        
        
        
        # when biggest value is from column 4
        if index_of_max == 3:
            iter4 += 1 # increment list iterator for current column
            loopIter = 0 # set loop iterator ... for iterating through anchor values????

            # "anchor" with col3 vals
            while loopIter < iter3: # loop through values in anchor column
                diff1 = abs(current_x_pos - Quad3[loopIter][1]) # set "anchor"
                
            #       and check col3 then col4
                i = 0 # set iterator
                while i < iter1: 
                    diff2 = abs(current_x_pos - Quad1[i][1])
                    if withinTolerance(diff1 * 3, diff2, tolerance):
                        # found!
                        # calculate 2nd index prediction
                        index1 = Quad1[i][1]
                        index3 = Quad3[loopIter][1]
                        index4 = current_x_pos
                        index2 = index1 + diff1
                        # if columnConfirm(index2, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm(index2, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1


                i = 0 # set iterator
                while i < iter2: 
                    diff2 = abs(current_x_pos - Quad2[i][1])
                    if withinTolerance(diff1, diff2, tolerance):
                        # found!
                        # calculate 3rd index prediction
                        
                        index2 = Quad2[i][1]
                        index3 = current_x_pos
                        index4 = Quad4[loopIter][1]
                        index1 =  index2 - diff1
                        # if columnConfirm(index1, proj_profile):
                        #     # return column indexes
                        #     return [index1, index2, index3, index4]
                        # else:
                        #     i += 1
                        if columnConfirm2(index1, index2, index3, index4) \
                            and columnConfirm(index1, proj_profile):
                            return [index1, index2, index3, index4]
                        else:
                            i += 1
                    else:
                        i += 1

                
                loopIter += 1 # incement anchor loop iterator
    
    else:
        return [-1,-1,-1,-1]