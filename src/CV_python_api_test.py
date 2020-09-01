#!/usr/bin/python
import cv2
import sys
import time

root_dir = '/home/jojo/face-detection'
data_dir = root_dir + '/data'
cascade_file = root_dir + '/opencv_model/lbpcascade_frontalface_improved.xml'

def getPIC(window_name,cam_id,pic_num,path,cascadeXML):
    # load opencv cascadeClassifier xml model
    classifier = cv2.CascadeClassifier(cascadeXML)
    # set the display windown name
    # set the camera
    cap = cv2.VideoCapture()
    cap.open(0)
    # set the const value
    color = (0,255,0)
    num = 0
    # get face pic loop
    if ~cap.isOpened():
        print("cap open failed!")
        return
    while cap.isOpened():
        if num < pic_num:
            states,frame = cap.read()
            if states == False:
                break
            # convert pic to the grey 
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # face detection, this function reuturns the center coordinate of face x,y, and the weight and height of the face
            face_coord = classifier.detectMultiScale(grey,scaleFactor = 1.2,minNeighbors = 3,minSize = (32,32))
            if len(face_coord) > 0:
                for oneface in face_coord:
                    x,y,w,h = oneface
                    image_name = '%s/%d.jpg' % (path,num)
                    image = frame[y:y+h,x:x+w]
                    cv2.imwrite(image_name, image)
                    num = num + 1
                    #draw a rectangle around face
                    cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'num:%d' % (num), (x+w,y+h), font, 1, (255,0,255),3)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1)
        else: 
            cap.release()
            cv2.destroyAllWindows()
            break

def main():
    print('Start face detction:')
    getPIC("Get Face",0, 100, data_dir,cascade_file)


if __name__ == '__main__':
    main()
