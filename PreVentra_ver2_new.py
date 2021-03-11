import sys
import os
from ctypes import *                                               # Import libraries
import math
import random
import cv2
from PIL import Image
import numpy as np
import time
import darknet
from sklearn.cluster import MeanShift
import socketio
import connect_socket 
import cam_stream
import cluster
import PreVentra_maintain as mt
from datetime import datetime

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from os.path import dirname, join
import operator
import time
cluster_id = 0

def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def anonymize_face_simple(image, factor=1.0):
   # automatically determine the size of the blurring kernel based
   # on the spatial dimensions of the input image
   (h, w) = image.shape[:2]
   kW = int(w / factor)
   kH = int(h / factor)
   # ensure the width of the kernel is odd
   if kW % 2 == 0:
      kW -= 1
   # ensure the height of the kernel is odd
   if kH % 2 == 0:
      kH -= 1
   # apply a Gaussian blur to the input image using our computed
   # kernel size
   return cv2.GaussianBlur(image, (kW, kH), 0)

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def find_old_risk(arr_key, arr, n_people, n_mask, nomask):
    if n_people == 0:
        return 0
    low_1 = 0
    low_2 = 0
    low_3 = 0
    up_3 = 0
    #n_dist = 0

    for i in arr_key:
        for j in arr_key:
            if i==j:
                continue
            dist = np.sqrt(pow(arr[i][0]-arr[j][0],2) + pow(arr[i][1]-arr[j][1],2) + pow(arr[i][2]-arr[j][2],2))
            dist = dist/100 #m 단위
            if dist<1:
                low_1 += 1
            elif dist<2:
                low_2 += 1
            elif dist<3:
                low_3 += 1
            else:
                up_3 += 1
            #n_dist += 1

    dist_risk = (12.8*low_1 + 5.3*low_2 + 2.6*low_3 + 1.0*up_3)/n_people
    mask_risk = 17.4 - 0.143*(n_mask/n_people*100) 
    total_risk = dist_risk * mask_risk

    if nomask/n_people*100 >= 80:
        total_risk = 450
    #print("%f 퍼센트"%(nomask/n_people*100))
    #effective_people = total_risk * n_people/100

    return total_risk

def find_risk(no_mask_detection, social_dist_X, peo_num, congestion):
    if peo_num == 0:
        risk = 0
    else:
        risk = (3 * no_mask_detection / peo_num + social_dist_X / peo_num) * congestion
    return risk
    
def cvDrawBoxes(detections, img, space_size, pro_dist, pro_peo, cluster_list, old_cluster_list, maintain_info, blurring):
    # Focal length
    #F = 800
    global cluster_id
    F = 1080
    space = space_size  
    red = (255, 30, 30)
    green =(50, 255, 50)
    blue = (5, 154, 255)

    person_detection = 0
    mask_detection = 0
    no_mask_detection = 0
    incorrect_detection = 0
    mask_unknown_detection = 0
    risk = 0
    congestion = 0
    n_cluster = 0
    cluster_info = []
    social_dist_x = 0

    if True: #len(detections) > 0:  					     
        pos_dict = dict()
        pos_box = dict()
        pos2d_dict = dict()
        pos_mon2D = np.empty((1, 2))
        centroid_dict = dict() 						
        objectId = 0							
        no_mask_list = []

        for detection in detections:			
            # Check for the only person name tag 
            name_tag = str(detection[0].decode())   
            x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]      
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h)) 
            
            x_mid = round((xmin+xmax)/2, 4)
            y_mid = round((ymin+ymax)/2, 4)
            
            height = round(ymax-ymin,4)
            distance = (27 * F)/height   # average human head size = 22 
            
            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F
            pos_dict[objectId] = (x_mid_cm,y_mid_cm,distance)
            pos2d_dict[objectId] = (x_mid,y_mid)
            pos_box[objectId] = (xmin, ymin, xmax, ymax)
            objectId += 1 
            
        #remove not right annotation
        list_not_right = []
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i<j:
                    if IOU(pos_box[i], pos_box[j]) > 0.5:                                 
                        if detections[i][1] > detections[j][1]:
                            list_not_right.append(j)
                        else:
                            list_not_right.append(i)        

        pos_key = list(pos_dict.keys())
        for i in pos_key:
            if i in list_not_right:
                del pos_dict[i]
                del pos2d_dict[i]
            
        #remove low detection
        list_no = []
        for i in pos_dict.keys():
            if detections[i][1]*100<40:  
                list_no.append(i)
        pos_key = list(pos_dict.keys())
        for i in pos_key:
            if i in list_no:
                del pos_dict[i]
                del pos2d_dict[i]

        #show label box and put 2D point
        pos_list = []
        for index in pos_dict.keys():
            detection = detections[index]
            name_tag = str(detection[0].decode())   
            x, y, w, h = detection[2][0],\
                        detection[2][1],\
                        detection[2][2],\
                        detection[2][3]      
                
            xmin, ymin, xmax, ymax = pos_box[index]   #convertBack(float(x), float(y), float(w), float(h)) 
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            pos_list.append((xmin, ymin))
            x_mid = round((xmin+xmax)/2, 4)
            y_mid = round((ymin+ymax)/2, 4)
                
            if name_tag == 'mask_weared':                
                cv2.rectangle(img, pt1, pt2, green, 1)
                mask_detection += 1 
                no_mask_list.append(False)
                
            if name_tag == 'mask_off':                
                cv2.rectangle(img, pt1, pt2, red, 1)
                no_mask_detection += 1
                no_mask_list.append(True)
                    
            if name_tag == 'mask_incorrect':
                cv2.rectangle(img, pt1, pt2, red, 1)    
                incorrect_detection += 1
                no_mask_list.append(True)
                    
            if name_tag == 'mask_unknown':                
                cv2.rectangle(img, pt1, pt2, blue, 1)
                no_mask_list.append(False)
                mask_unknown_detection += 1
                
            img_h, img_w = img.shape[:2]
            if blurring == True:
                if xmin < 0:   xmin = 0
                if xmax < 0:   xmax = 0
                if ymax < 0:   ymax = 0
                if ymin < 0:   ymin = 0
                face = img[ymin+1:ymax, xmin+1:xmax] 
                face = anonymize_face_simple(face, factor=1.0) 
                img[ymin+1:ymax, xmin+1:xmax] = face 

            pos_mon2D = np.append(pos_mon2D, [[x_mid, y_mid]],axis = 0)         
            person_detection += 1  
            
        a = 0
        for index in pos_dict.keys():
            pt1 = pos_list[a]
            a += 1
            detection = detections[index]
            name_tag = str(detection[0].decode())
            if name_tag == 'mask_weared':                
                cv2.rectangle(img, (pt1[0], pt1[1]-15), (pt1[0]+60, pt1[1]), green, -1)
                cv2.putText(img, "mask on",(pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX , 0.4, [0,0,0], 1)

            if name_tag == 'mask_off':                
                cv2.rectangle(img, (pt1[0], pt1[1]-15), (pt1[0]+60, pt1[1]), red, -1)
                cv2.putText(img, "mask off",(pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX , 0.4, [0,0,0], 1)  #round(w/130, 1)
                    
            if name_tag == 'mask_incorrect':
                cv2.rectangle(img, (pt1[0], pt1[1]-15), (pt1[0]+60, pt1[1]), red, -1)
                cv2.putText(img, "incorrect",(pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX , 0.4, [0,0,0], 1)     
                    
            if name_tag == 'mask_unknown':                
                cv2.rectangle(img, (pt1[0], pt1[1]-15), (pt1[0]+60, pt1[1]), blue, -1)
                cv2.putText(img, "unknown",(pt1[0], pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX , 0.4, [0,0,0], 1)
        pos_mon2D = np.delete(pos_mon2D, [0, 0], axis=0)

        #clustering
        detect_list = []
        band = pro_dist*100   # bandwidth = 100(1m)
        
        if len(pos_dict) != 0:            
            pos_arr = np.zeros((len(pos_dict), 3))
            s = 0
            for d in pos_dict.keys():
                pos_arr[s] = pos_dict[d]
                s += 1

            clust = MeanShift(bandwidth=band).fit(pos_arr)   
            labels = clust.labels_
            centers = clust.cluster_centers_

            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            max_center_2D = np.zeros(n_clusters_)
            cluster_risk = np.zeros(n_clusters_)

            #found cluster risk
            for k in range(n_clusters_):
                my_members = labels == k
                member_no_mask = np.logical_and(my_members, no_mask_list)
                n_no_mask = np.sum(member_no_mask)
                member = pos_arr[my_members]
                member_2d = pos_mon2D[my_members]
                n_member = len(member)

                if n_member == 1:
                    cluster_risk[k] = 0.0
                else:
                    center = centers[k]
                    center_2d = (center*F)/center[2]
                    distance_to_center = np.zeros(n_member)
                    distance_to_center_2D = np.zeros(n_member)
                    for q in range(n_member):
                        distance_to_center[q] =np.power((member[q][0]-center[0]),2)+ np.power((member[q][1]-center[1]),2) + np.power((member[q][2]-center[2]),2) # x_mid, y_mid, distance.
                        distance_to_center_2D[q] = np.power((member_2d[q][0]-center_2d[0]),2)+ np.power((member_2d[q][1]-center_2d[1]),2)
                    max_distance = np.max(distance_to_center)                    
                    max_center_2D[k] = np.max(distance_to_center_2D)
                    if np.sqrt(max_distance) < band:        # 너무 큰 cluster들은 넣지 말기!          
                        cluster_risk[k] = find_old_risk(range(n_member), member, n_member, (n_member-n_no_mask), n_no_mask)/5
                        detect_cluster = cluster.Cluster(center, n_member, np.sqrt(max_distance), np.sqrt(max_center_2D[k]), cluster_risk[k])
                        detect_list.append(detect_cluster)
                        
        img = cluster.update_cluster(cluster_list, detect_list, band, img, old_cluster_list)            # cluster 정보 update하기
        img = cv2.putText(img, str(len(detect_list)), (0,200), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
        
        """
        #show distance base 2d line
        for i in pos2d_dict.keys():
            for j in pos2d_dict.keys():
                if i > j:
                    dist = np.sqrt(pow(pos2d_dict[i][0]-pos2d_dict[j][0],2) + pow(pos2d_dict[i][1]-pos2d_dict[j][1],2))
                    if dist < 100:
                        x1, y1 = pos2d_dict[i]
                        x2, y2 = pos2d_dict[j]
                        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
        """
        #found congestion      
        total_den = 0
        total_risk = 0
        sd_none = []
        for i in pos_dict.keys():
            one_den = 0
            for j in pos_dict.keys():
                if i == j:
                    continue
                dist = np.sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))
                dist = dist/100 #(m)     

                #show distance base 3d line
                if dist < pro_dist and i < j:
                    xmin, ymin, xmax, ymax = pos_box[i]
                    x1 = int((xmin+xmax) / 2)
                    y1 = int((ymin+ymax) / 2)
                    xmin, ymin, xmax, ymax = pos_box[j]
                    x2 = int((xmin+xmax) / 2)
                    y2 = int((ymin+ymax) / 2)
                    img = cv2.line(img, (x1, y1), (x2, y2), red, 2)
                    img = cv2.putText(img, str(round(dist,2)), (int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                    sd_none.append(i)
                    sd_none.append(j)
        
        #found social distance X
        sd_none = set(sd_none)
        sd_none = list(sd_none)
        social_dist_x = len(sd_none)
        
        density = person_detection / pro_peo * 100
        if density <= 40: total_den = 20
        elif density <= 80: total_den = 40
        elif density <= 100: total_den = 60
        elif density <= 120: total_den = 80
        else: total_den = 100
        
        total_den += 5 * social_dist_x

        #found total_risk
        real_no_mask = np.sum(no_mask_list)
        total_risk = find_risk(no_mask_detection, social_dist_x, person_detection, total_den)
        # old_version: total_risk = find_old_risk(pos_dict.keys(), pos_dict, person_detection, (person_detection-real_no_mask), real_no_mask)   
        
        #opencv에 나타내기
        img, n_cluster, cluster_info, cluster_id = cluster.show_cluster(cluster_list, img, F, 0, cluster_id)
        
        #maintain
        congestion = total_den  
        risk = total_risk      
        print("total_congestion: %s"%str(congestion))
        print("total_risk: %s "%str(risk))
        print("\n")
        person_detection = maintain_info.maintain_peo(person_detection)
        risk = maintain_info.maintain_risk(risk)
        congestion = maintain_info.maintain_congestion(congestion)

        img = cv2.putText(img, "risk: " + str(maintain_info.pre_risk_level), (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
        img = cv2.putText(img, "congestion: " + str(maintain_info.pre_level), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)

    img = cv2.putText(img, "no mask: " + str(no_mask_detection), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2)
    
    dict_result = {}
    dict_result["mask_weared"] = mask_detection
    dict_result["mask_off"] = no_mask_detection
    dict_result["mask_incorrect"] = incorrect_detection
    dict_result["mask_unknown"] = mask_unknown_detection
    dict_result["congestion"] = congestion
    dict_result["risk"] = risk
    dict_result["n_cluster"] = n_cluster
    dict_result["cluster_info"] = cluster_info
    dict_result["n_not_keep_dist"] = social_dist_x
    
    print(dict_result)
    return img, dict_result, maintain_info

netMain = None
metaMain = None
altNames = None
net = None
meta = None


def YOLO(is_cam, camera_id, address, space, pro_dist, pro_peo, blurring):
    global metaMain, netMain, altNames, net, meta
    configPath = "./cfg/yolov4-mask.cfg"                                 
    weightPath = "./yolov4-mask_final.weights"                                 # 나중에 train한 weight 넣기  -  backup에서 가져오면 됨              
    metaPath = "./cfg/mask.data" 
    tracking = False

    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" + os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # cctv 카메라 영상, 저장된 영상 중 하나 선택
    if is_cam:
        cap = cam_stream.CamStream(address)
        cap.set_frame_rate(".4")  
        cap.start()
    else:
        cap = cv2.VideoCapture(address) 

    frame_width = 854                                 # Returns the width and height of capture video
    frame_height = 480

    print("Starting the YOLO loop...")
    socket = connect_socket.SocketClient("http://115.145.212.100:51122", camera_id)

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    cluster_list = []
    old_cluster_list = []
    
    maintain_info = mt.Maintain()

    #tracking 여부에 따라 선택하기
    if tracking:
        track_c = for_tracking.Track(frame_width, frame_height)
        track_c.start_track()

    try:
        if is_cam:
            prev_time = time.time()
            elapsed = cap.get_frame_rate()
            after_show = False
            image_to_show = None
            while True:                                                      # Load the input frame and write output frame.        
                if after_show and (cv2.waitKey(20) & 0xFF == ord('q')):
                    break

                im = cap.get_latest_frame()
                
                if elapsed >= cap.get_frame_rate():
                    image_to_show = im
                else:
                    elapsed = time.time() - prev_time
                    continue

                if image_to_show is None:
                    continue

                prev_time = time.time()
        
                frame_rgb = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
                frame_resized = cv2.resize(frame_rgb,(frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

                darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image
                now = datetime.now()
                cap_time = now.strftime("%Y-%m-%d %H:%M:%S")
                # yolov4를 사용하여 마스크 착용여부 분류하기
                detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   

                dict_result = {}
                # 이미지에 결과 표시 및 감염 위험도 및 혼잡도 계산하기
                image, dict_result, maintain_info = cvDrawBoxes(detections, frame_resized, space, pro_dist, pro_peo, cluster_list, old_cluster_list, maintain_info, blurring)         
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                   
                dict_result["camera_id"] = camera_id
                dict_result["time"] = cap_time 
                
                #tracking을 할 경우 사람 feature 정보 저장
                if tracking:      
                    track_c.now_tracking(image_to_show)
                    track_c.show_track(image)
                socket.send_data(image, dict_result)

                elapsed = time.time() - prev_time
                after_show = True

        else:         
            #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))  
            while True:
                now = datetime.now() 
                now_mic = now.microsecond
                if now_mic % 30000 == 0:  # 1 second
                    ret, frame_read = cap.read()
                    if not ret:                                                  # Check if frame present otherwise he break the while loop
                        break
                    if tracking:      
                        track_c.now_tracking(frame_read)                                  
        
                if now_mic % 600000 == 0:  # 2 frame: 600000 / 4 frame: 300000
                    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
                    frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes()) 
                    now = datetime.now()
                    cap_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)                    
                    dict_result = {}
                    image, dict_result, maintain_info = cvDrawBoxes(detections, frame_resized, space, pro_dist, pro_peo, cluster_list, old_cluster_list, maintain_info, blurring)  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if tracking:
                        track_c.show_track(image)
                    dict_result["camera_id"] = camera_id
                    dict_result["time"] = cap_time                     
                    socket.send_data(image, dict_result)
                    
                    #out.write(image)

    except KeyboardInterrupt:
        print("cancelled")
        
    finally:
        if is_cam:       
            cap.stop()                                                    # For releasing cap and out. 
        else:
            cap.release() 
            out.release()
        print(":::Video Write Completed")
        socket.disconnect()


if __name__ == "__main__":
    is_cam = False
    camera_id = 1
    address = "/home/seungho/darknet/2020-sw-skku-GIT/video-gumin/co-working01.mp4" #"rtsp://gumin1:123412341234@192.168.0.14:554/stream1"  #"rtsp://skkutapo:skkuproject@192.168.0.9:554/stream1" #"/home/seungho/darknet/2020-sw-skku-GIT/video/cafe.mp4" 
    space = 39
    pro_dist = 2
    pro_peo = 6
    blurring = False
    YOLO(is_cam, camera_id, address, space, pro_dist, pro_peo, blurring)                                                           # Calls the main function YOLO()