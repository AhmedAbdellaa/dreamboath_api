import cv2
import numpy as np
import os 
import mediapipe as mp
import base64

def readb64(uri):
    try :
        nparr = np.fromstring(base64.b64decode(uri), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except : 
        return np.array([])
    

def get_new_box(src_w, src_h, bbox):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        # to make box square
        if box_h > box_w :
            box_w = box_h
        else :
            box_h = box_w

        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = x
        left_top_y = y
        right_bottom_x = center_x+box_w/2
        right_bottom_y = center_y+box_h/2

        left_distance = left_top_x
        right_dist = src_w - right_bottom_x
        top_dist = left_top_y
        bottom_dist = src_h - right_bottom_y


        #if image width less than 3 boxes of face size
        if src_w <box_w *3 :
            left_top_x =0 #left point
            right_bottom_x = src_w #right point
            # if distination betwen top of face and top of images less than 1.5 face height 
            if top_dist < 1.5*box_h : 
                left_top_y =0 # top point
                right_bottom_y += src_w - box_h - top_dist # bottom point 
            else:
                if np.abs(bottom_dist) <1.5*box_h :
                    right_bottom_y = src_h
                    left_top_y -= src_w - box_h - np.abs(bottom_dist)
                #both top and down are bigger than 1.5 face height 
                else :
                    left_top_y -= box_h
                    right_bottom_y +=box_h
        
        
        elif left_distance >= box_w and top_dist >= box_h \
            and right_dist >= box_w and bottom_dist >= box_h :

            left_top_x -=box_w
            left_top_y -=box_w
            right_bottom_x += box_w
            right_bottom_y +=box_w
        else : 
            min_dist = min(abs(np.array([left_distance ,right_dist ,top_dist ,bottom_dist])))
            left_top_x -=min_dist
            left_top_y -=min_dist
            right_bottom_x += min_dist
            right_bottom_y +=min_dist
            if left_top_x < 0:
            # right_bottom_x -= left_top_x
                diff = 0 - left_top_x
                right_bottom_x = right_bottom_x - diff   
                left_top_x = 0

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)
def crop(org_img, bbox, out_w, out_h):

        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, \
            right_bottom_x, right_bottom_y = get_new_box(src_w, src_h, bbox)

        img = org_img[left_top_y: right_bottom_y+1,
                        left_top_x: right_bottom_x+1]
        dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img
def normalized_to_pixel_coordinates(
    normalized_x, normalized_y,normalized_w, normalized_h, image_width,image_height) :
  """Converts normalized value pair to pixel coordinates."""
  print()
  x_px = min(np.floor(normalized_x * image_width), image_width - 1)
  y_px = min(np.floor(normalized_y * image_height), image_height - 1)
  w_px = min(np.floor(normalized_w * image_width), image_width - 1)
  h_px = min(np.floor(normalized_h * image_height), image_height - 1)
  return x_px, y_px, w_px, h_px



def face_crop(results,image):  
    d = results.detections[0].location_data.relative_bounding_box
    src_h, src_w, _ = np.shape(image)
    image_bbox = normalized_to_pixel_coordinates(d.xmin,d.ymin,d.width,d.height,src_w,src_h)
            
    param = {
            "org_img": image,
            "bbox": image_bbox,
            "out_w": 512,
            "out_h": 512,
        }
    return crop(**param)
        


def cutOnSeg(image,mask,max_y=None):
    if max_y != None :
        where = np.where(mask[:max_y,:] > 0.1)
    else :
        where = np.where(mask > 0.1)
    top = where[0].min()
    bottom = where[0].max()
    left = where[1].min()
    right = where[1].max()

    return image[top:bottom,left:right,:]

# cut_on_l = ["face","half","close","full"]

#mediapipe opject
# for c in cut_on_l :
    # cut_on = c
def image_croping(imlist,out_path,cut_on):
    
    i = 0
    image_count = 0
    if cut_on is None :
        for img64 in imlist:
            # read image.
            i =i+1
            image = readb64(img64)
            if image.shape[0] !=0  :
                cv2.imwrite(os.path.join(out_path,str(i)+'.jpg'),image)
                image_count +=1
            else :
                continue
        return image_count

    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2,enable_segmentation=True) 
    image_count = 0
    
    for img64 in imlist:
        # read image.
        i =i+1
        image = readb64(img64)
        if image.shape[0] !=0  :
            #produce lacndmark and segmentaion
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if len(results.detections) ==1:
                if cut_on=='face':
                    im_cu = face_crop(results,image)
                    cv2.imwrite(os.path.join(out_path,str(i)+'.jpg'),im_cu)
                    image_count +=1
                else : 
                    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


                    image_hight, image_width, _ = image.shape
                    
                    # cut_image = image.copy()

                    points = {idx:( int(r.x * image_width),int(r.y * image_hight))
                                for idx, r in enumerate(results.pose_landmarks.landmark)
                                if int(r.x*image_width) <image_width and int(r.y*image_hight) <image_hight  }  
                    # print(points)
                    if cut_on =="full" :
                        cut_image = cutOnSeg(image,results.segmentation_mask)
                        
                    elif cut_on == "half":
                        if 24 in points and 23 in points:
                            max_y = max(points[23][1],points[24][1])

                            cut_image = cutOnSeg(image,results.segmentation_mask,max_y=max_y)
                            # print("found half body")  
                        else:
                            cut_image = cutOnSeg(image,results.segmentation_mask)

                        

                    elif cut_on == "close" :
                        if 11 in points and 12 in points:
                            max_y = max(points[11][1],points[12][1])

                            cut_image = cutOnSeg(image,results.segmentation_mask,max_y=max_y)

                            # print("found close ") 
                            
                        else :
                            cut_image = cutOnSeg(image,results.segmentation_mask)

                    image_count +=1
                    cv2.imwrite(os.path.join(out_path,str(i)+'.jpg'),cut_image)
            else:
                print("no single face detection")
        else :
            continue
    return image_count
