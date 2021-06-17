# find . -name "*.DS_Store" -type f -delete
import cv2
import pickle
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import clock
import imutils
import multiprocessing as mp
import mysql.connector
import pymysql.cursors
import pandas as pd

def save_desc_file(train_path, method, condition):
    # Start the stopwatch / counter
    t1_start = clock()
    # Get name path of subfolder train
    training_names = os.listdir(train_path)

    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths 
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
        image_classes += [class_id]*len(class_path)
        class_id += 1

    des_list = []
    if(condition == 1):
        for i, filename in enumerate(image_paths):
            img = cv2.imread(filename)
            grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = detector.detectAndCompute(
                grey_img, None)
            des_list.append(desc) #SAVE KP AND DESC HERE

        # Open des_list and dump des_list to des_list.txt
        with open('des_list_{}.txt'.format(method), 'wb') as fp: 
            pickle.dump(des_list, fp)
        # Open des_list.txt and save data in des_list.txt to des_list
        with open('des_list_{}.txt'.format(method), 'rb') as fp:
            des_list = pickle.load(fp)
    elif (condition == 0):
        # Open des_list.txt and save data in des_list.txt to des_list
        with open('des_list_{}.txt'.format(method), 'rb') as fp:
            des_list = pickle.load(fp)
    
    # Stop the stopwatch / counter
    t1_stop = clock()
    # # Print Elapsed time during t1_stop-t1_start
    print("Save desc file time during in seconds:", t1_stop-t1_start)
    return des_list ,image_paths

def countkeypointgood_imgtest(method, i, des_list, desctest):
    if method == "orb":
        # Defind desc from each array of des_list
        desc = des_list[i]

        # FLANN algorithm matching
        FLANN_INDEX_KDTREE = 6
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks=32)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # FLANN use knnMatch desc and desctest
        matches = flann.knnMatch(desc, desctest, 2) # k=2 is dimension

        ratio = 0.75
        good = [m[0] for m in matches \
                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]
    elif method == "sift" or method == "surf":
        # Defind desc from each array of des_list
        desc = des_list[i]

        # FLANN algorithm matching
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # FLANN use knnMatch desc and desctest
        matches = flann.knnMatch(desc, desctest, k=2) # k=2 is dimension

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # Ratio test as per Lowe's paper
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)
    return len(good)

def capture_frame(frame):
    # Start the stopwatch / counter
    t1_start = clock()
    # Convert imgtest BGR to GRAYSCALE
    imgtest = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the Keypoints and Descriptors with SIFT in Image test
    kptest, desctest = detector.detectAndCompute(imgtest, None)
    # Stop the stopwatch / counter
    t1_stop = clock()
    # # Print Elapsed time during t1_stop-t1_start
    print("Detect desc frame Elapsed time during in seconds:", t1_stop-t1_start)
    return desctest

def select_method(method):
    # Start the stopwatch / counter
    t1_start = clock()
    print("Method used: "+method)
    if method == "orb":
        detector = cv2.ORB_create(1000)
    elif method == "sift":
        detector = cv2.xfeatures2d.SIFT_create()
    elif method == "surf":
        detector = cv2.xfeatures2d.SURF_create()
    # Stop the stopwatch / counter
    t1_stop = clock()
    # # Print Elapsed time during t1_stop-t1_start
    print("Select method Elapsed time during in seconds:", t1_stop-t1_start)
    return detector

def parallel_detect(method,des_list,desctest):
    # Parallel to count good match points
    # Start the stopwatch / counter
    t1_start = clock()
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(countkeypointgood_imgtest,args=(method,i,des_list,desctest)).get() for i, filename in enumerate(image_paths)]
    pool.close()
    # Stop the stopwatch / counter
    t1_stop = clock()
    # # Print Elapsed time during t1_stop-t1_start
    print("Count Match Parallel Elapsed time during in seconds:", t1_stop-t1_start)
    return results

if __name__ == "__main__":
     # Sometimes reading CSV for Excel need encoding
    df = pd.read_csv('productnew.csv', encoding="ISO-8859-1")
    
    # connect Database "cal_object_detection"
    connection = pymysql.connect(
        host="194.59.164.43",
        user="u489851921_root",
        passwd="0431A3b5cf",
        db="u489851921_calobj",
        cursorclass=pymysql.cursors.DictCursor
    )

    print("================ Detail =================")
    method = "orb" # Initiate detector  // orb // sift // surf
    print("Use CPU CORE: ",mp.cpu_count())
    MIN_MATCH_COUNT = 25
    condition = 0 # Create des_list for save Descriptors of each images train
    train_path = "./models" # Defind of subfolder train
    detector = select_method(method)
    des_list, image_paths = save_desc_file(train_path, method, condition)
    print("==============================================")

    # Create cam by Video file or Capture moniter
    cam = cv2.VideoCapture(0) #Capture moniter
    cv2.namedWindow("Capture video with parallel orb") # Create Windows of text to detect

    # Loop of Detection
    while True:
        # ret is a boolean variable that returns true if the frame is available.
        # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly.
        ret, frame = cam.read() 
        cv2.imshow("test", frame) # Show windows of frame

        if not ret: # Check (boolean) ret
            break

        k = cv2.waitKey(1) # k is wait key from keybroad

        # If ESC pressed then break the loop
        if k % 256 == 27: 
            print("Escape hit, closing...")
            break
        
        # If SPACE pressed
        elif k % 256 == 32:
            # Start the stopwatch / counter
            t1_start = clock()
            # Capture frame
            desctest = capture_frame(frame)
            
            # Feature matching parallel
            results = parallel_detect(method,des_list,desctest)

            # Start the stopwatch / counter
            t1_start = clock()
            # Draw Matches Homography if good matches more then MIN_MATCH_COUNT What's 25
            if max(results) > MIN_MATCH_COUNT:
                # Add product sql IMPORT CSV to MATCH PRODUCT
                filename = image_paths[results.index(max(results))]
                # Add product sql IMPORT CSV to MATCH PRODUCT
                filename = os.path.basename(filename)
                filename = os.path.splitext(filename)[0]
                print("filename", filename)
                product_item = str(df[df['product_name']==filename].product_id.sum())
                print("product_item", product_item)

                # connect DB query
                with connection.cursor() as cursor:
                    # Select pamount from cart where pid like product_item to pInTable
                    sql = "SELECT `pamount` FROM `cart` WHERE `pid` LIKE %s"
                    cursor.execute(sql, (product_item,))
                    pInTable = cursor.fetchone()
                    print(pInTable)
                
                    # If pInTable doesn't have
                    if not pInTable:
                        # Add product to cart by Insert
                        sql = "INSERT INTO `cart` (`pid`, `pamount`) VALUES (%s, %s)"
                        cursor.execute(sql, (product_item, "1"))
                        connection.commit()
                        # connection.close()
                    
                    # Else pInTable already have product in cart
                    else:
                        # Get amount of product from cart where pid like product_item
                        sql = "SELECT `pamount` FROM `cart` WHERE `pid` LIKE %s"
                        cursor.execute(sql, (product_item, ))
                        amountp = cursor.fetchone()
                        amountp = amountp[0]+1

                        # Add product to cart by update from old amount
                        sql = "UPDATE `cart` SET `pamount`=%s WHERE `pid` = %s"
                        cursor.execute(sql, (amountp, product_item))
                        connection.commit()
                        print(cursor.rowcount, "record(s) affected")
                        # connection.close()

                # # Display the resulting frame
                img = cv2.imread(image_paths[results.index(max(results))])
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),), plt.show()

            # Stop the stopwatch / counter
            t1_stop = clock()
            # # Print Elapsed time during t1_stop-t1_start
            print("Check good match Elapsed time during in seconds:", t1_stop-t1_start)

            # Stop the stopwatch / counter
            t1_stop = clock()
            # # Print Elapsed time during t1_stop-t1_start
            print("Total matching Elapsed time during in seconds:", t1_stop-t1_start)
            print("==============================================")

    cam.release()
    cv2.destroyAllWindows()