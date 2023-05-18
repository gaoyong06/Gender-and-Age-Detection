'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-05-18 16:10:01
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-05-18 16:23:26
FilePath: \Gender-and-Age-Detection\detect_image_gender.py
Description: 对detect.py做了一些裁剪,去掉了摄像头的处理,去掉了关于绘制边框和文字的部分，只保留图片内人脸识别及性别检测,如果识别为男性则打印Male,如果识别为女性则打印Female,如果为检测到人脸则打印No face detected
'''
import cv2
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

frame = cv2.imread(args.image)
resultImg,faceBoxes=highlightFace(faceNet,frame)
if not faceBoxes:
    print("No face detected")

for faceBox in faceBoxes:
    face=frame[max(0,faceBox[1]-20):
               min(faceBox[3]+20,frame.shape[0]-1),max(0,faceBox[0]-20)
               :min(faceBox[2]+20, frame.shape[1]-1)]

    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')