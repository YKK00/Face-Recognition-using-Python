import warnings
warnings.filterwarnings("ignore")
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2
import time
import os

def recognition():
    # 初始化预训练的pytorch人脸检测模型MTCNN和预训练的pytorch人脸识别模型InceptionResnet
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # 加载模型文件
    load_data = torch.load('/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/model.pt')
    embedding_list = load_data[0]
    name_list = load_data[1]
    # 初始化摄像头
    cam = cv2.VideoCapture(0)
    while True:
        # 读取图像
        ret, frame = cam.read()
        if not ret:
            print("获取图像失败，请重试")
            break
        img = Image.fromarray(frame)
        # img = img.copy()
        # 用MTCNN检测是否为人脸
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        # 检测图像是否能对得上已知人名
        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)
            for i, prob in enumerate(prob_list):
                if prob > 0.95:
                    # 用InceptionResnet生成嵌入矩阵
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()
                    # 距离列表
                    dist_list = []
                    for idx, emb_db in enumerate(embedding_list):
                        # 算出现有图像的嵌入矩阵和模型中的嵌入矩阵之间的距离（使用二阶范数算出其岭回归/权值衰减曲线，并遍历取出值）
                        # 二阶范数算法为张量各元素的平方和然后求平方根
                        # 使用二阶范数的目的是为了防止过拟合
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)
                    # 获取最小距离
                    min_dist = min(dist_list)
                    # 获取最小距离在列表中的索引
                    min_dist_idx = dist_list.index(min_dist)
                    # 获取与最小距离的索引对应的人名
                    global name
                    name = name_list[min_dist_idx]
                    box = boxes[i]
                    # 存储处理之前的每帧图像
                    original_frame = frame.copy()
                    # 根据二阶范数值来确定识别结果
                    if min_dist > 0.85:
                        frame = cv2.putText(frame, 'Unknown', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                    elif min_dist <= 0.85 and min_dist >= 0.75:
                        frame = cv2.putText(frame, 'Recognizing..', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 255), 2, cv2.LINE_AA)
                    elif min_dist < 0.75:
                        frame = cv2.putText(frame, name, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                            cv2.LINE_AA)
                        print(name)
                    # 输出处理过的图像
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
        cv2.imshow("IMG", frame)
        k = cv2.waitKey(1)
        # 按下esc结束识别
        if k % 256 == 27:
            print('正在结束...')
            break
        # 按下空格键保存识别的图像
        elif k % 256 == 32:
            print('输入你的名字 :')
            name = input()
            # 若文件件不存在则创建文件夹
            if not os.path.exists('/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/result/'):
                os.mkdir('/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/result/')
            img_name = "/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/images_pytorch/result/{}-{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print("saved: {}".format(img_name))
    cam.release()
    cv2.destroyAllWindows()
recognition()