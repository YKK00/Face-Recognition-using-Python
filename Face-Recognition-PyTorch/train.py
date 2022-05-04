from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import datetime

# 初始化预训练的pytorch人脸检测模型MTCNN和预训练的pytorch人脸识别模型InceptionResnet
mtcnn = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 从照片集中读取数据
dataset = datasets.ImageFolder('/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch-main/images_pytorch')
# 关联名字和文件
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}
print('开始时间 :',datetime.datetime.now())
print('Training..')

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

# 关联人名和照片的列表
name_list = []
# 嵌入矩阵列表
embedding_list = []

# 用MTCNN检测是否为人脸并且用InceptionResnet生成嵌入矩阵
for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# 保存模型数据
data = [embedding_list, name_list]
torch.save(data, '/Users/zhengrongkai/PycharmProjects/Face-Recognition-PyTorch/model.pt')
print('训练完成')
print('完成时间 :',datetime.datetime.now())