'''
Author: gaoyong gaoyong06@qq.com
Date: 2023-05-18 16:10:01
LastEditors: gaoyong gaoyong06@qq.com
LastEditTime: 2023-05-18 16:39:29
FilePath: \Gender-and-Age-Detection\detect_image_gender.py
Description: 对detect.py做了一些裁剪,去掉了摄像头的处理,去掉了关于绘制边框和文字的部分，只保留图片内人脸识别及性别检测,如果识别为男性则打印Male,如果识别为女性则打印Female,如果为检测到人脸则打印No face detected
'''
import sys
import cv2
import argparse

# 设置中文编码
sys.stdout.reconfigure(encoding='utf-8')

# 定义人脸检测函数highlightFace，用于人脸检测、绘制人脸框
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # 通过cv2.dnn.blobFromImage方法将图像转换为blob格式
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    # 将blob格式输入神经网络进行前向计算
    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        # 根据阈值thresh筛选出置信度大于阈值的人脸框
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
    return frameOpencvDnn,faceBoxes

# 解析命令行参数
parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()

# 定义人脸检测及性别分类所需的模型
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

genderList=['Male','Female']

# 加载模型
faceNet=cv2.dnn.readNet(faceModel,faceProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# 读取指定的图片
frame = cv2.imread(args.image)

# 调用highlightFace函数进行人脸检测，返回绘制人脸框后的图像及人脸框位置信息
resultImg,faceBoxes=highlightFace(faceNet,frame)
# 如果未检测到人脸，输出提示信息
if not faceBoxes:
    print("No face detected")

# 如果检测到人脸，分别对每张人脸进行性别分类，并输出性别信息
for faceBox in faceBoxes:
    face=frame[max(0,faceBox[1]-20):
               min(faceBox[3]+20,frame.shape[0]-1),max(0,faceBox[0]-20)
               :min(faceBox[2]+20, frame.shape[1]-1)]

    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')