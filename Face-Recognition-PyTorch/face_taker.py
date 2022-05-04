import cv2
import os

# 初始化摄像头参数
cam = cv2.VideoCapture(0)
cam.set(3,640)
face_count = 0

# 读取OpenCV级联分类器参数配置文件
faceCascade = cv2.CascadeClassifier('/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/haarcascade_frontalface_default.xml')

# 为每个人采集面部id
face_id = input('\n 输入您的名字，并按下回车键确认 -->  ')
print("\n [INFO] 正在采集面部数据，请直视摄像头 ...")
print("\n [INFO] 若要结束采集，请按下esc并等待 ...")

if not os.path.exists(f'/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/images_pytorch/{face_id}'):
    os.makedirs(f'/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/images_pytorch/{face_id}')

while(True):
    # 读取摄像头图像并检测脸部范围，三个参数分别为：要检测的目标图像，每次图像尺寸减小的比例，每一个目标至少要被检测到几次才算是真的目标
    ret, img = cam.read()
    faces = faceCascade.detectMultiScale(img, 1.3, 5)

    # 对于抓取到的图像进行框选处理
    # 绘制矩形rectangle函数参数依次为：图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        face_count += 1
        # 保存抓取到的图像
        cv2.imwrite(f"/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/images_pytorch/{face_id}/Users.{face_id}.{face_count}.jpg",img[y:y+h,x:x+w])
        cv2.imshow('image', img)
    # 按下esc强制结束程序
    k = cv2.waitKey(100) & 0xff
    if k < 100:
        break
    # 抓取够100张图片即可自动结束程序
    elif face_count >= 100:
         break

print("\n [INFO] 正在结束程序 ...")
cam.release()
cv2.destroyAllWindows()