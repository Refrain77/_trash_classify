#coding=utf-8
# coding=gbk
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
from torchvision import transforms, datasets, utils
from thop import profile, clever_format
import torch.nn.functional as F

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" #指定使用第二块GPU

# 加载数据集
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),#将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
    #随机裁剪后的大小: (224, 224)
                                 transforms.RandomHorizontalFlip(),#随机水平（左右）翻转，默认值为0.5
                                 transforms.ToTensor(),#图片转换成形状为(C, H, W)的Tensor格式
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                 #使用transforms.Normalize(mean, std)对图像按通道进行标准化，即减去均值，再除以方差。
                                 #这样做可以加快模型的收敛速度。其中参数mean和std分别表示图像每个通道的均值和方差序列。
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

image_path = "/data/lhl/split-garbage-dataset/"
#1、训练集（train set）：用于训练模型以及确定参数。相当于老师教学生知识的过程。

#2、验证集（validation set）：用于确定网络结构以及调整模型的超参数。相当于月考等小测验，用于学生对学习的查漏补缺。

#3、测试集（test set）：用于检验模型的泛化能力。相当于大考，上战场一样，真正的去检验学生的学习效果。

assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])

test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])


# 设置超参数
batch_size = 16
learning_rate = 0.0002
num_epochs = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义神经网络模型

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

 
 
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
expansion = 2
#用在resnet18中的结构，也就是两个3x3卷积
class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    #inplanes：输入通道数
    #planes：输出通道数
    #base_width，dilation，norm_layer不在本文讨论范围
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        #中间部分省略
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride  
        self.SE = SE_Block(planes)      
        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), 
            )


    def forward(self, x):
        #为后续相加保存输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.SE(out)
        
        woha = self.shortcut(x)
        
        out += woha

        out = self.relu(out)

        return out

class res18(nn.Module):
    def __init__(self):
        super(res18, self).__init__()
        self.Conv1 = nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3, groups=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.biock1 = BasicBlock(64,64, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)#inp, oups, kernel_size, stride
        #inplanes, planes, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None
        self.biock2 = BasicBlock(64,64, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.biock3 = BasicBlock(64,128, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.biock4 = BasicBlock(128,128, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        ##215x144
        self.biock5 = BasicBlock(128,256, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.biock6 = BasicBlock(256,256, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.biock7 = BasicBlock(256,512, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.biock8 = BasicBlock(512,512, stride=1, downsample=None, groups=1,base_width=64, dilation=1, norm_layer=None)
        self.pool2 = nn.AvgPool2d(4) 
      
        self.linear = nn.Linear(100352, 6) #一共6类

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.Conv1(x)
        x = self.pool1(x)
        x = self.biock1(x)
        x = self.biock2(x)  
        x = self.biock3(x)
        x = self.biock4(x) 
        x = self.biock5(x)
        x = self.biock6(x)
        x = self.biock7(x)
        x = self.biock8(x)
  
        x = self.pool2(x)  
        
        x = x.view(x.size(0),-1)  #输出拉伸为一行
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
        
model = res18().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 测试模型
def validate(model, device, test_loader):
    model.eval()
    loss = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    loss /= len(test_loader.dataset)
    accuracy = 100 * sum([int(predicted_labels[i] == true_labels[i]) for i in range(len(predicted_labels))]) / len(predicted_labels)
    recall = recall_score(true_labels, predicted_labels, average='macro')
    precision = precision_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, kappa, loss

# 训练模型
best_model_epoch = 0
best_acc = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

        train_loss += loss.item() * labels.size(0)

    train_loss /= len(train_dataset)

    # 每个epoch结束后，在测试集上评估一下模型的性能
    val_acc, val_precision, val_recall, val_f1, val_kappa, val_loss = validate(model, device, test_loader)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, val_loss))
    print('Val Accuracy: {:.2f}%'.format(val_acc))
    print('Val Recall: {:.4f}'.format(val_recall))
    print('Val Precision: {:.4f}'.format(val_precision))
    print('Val F1-score: {:.4f}'.format(val_f1))
    print('Val Kappa: {:.4f}'.format(val_kappa))
    print('-' * 25)
    
    
dummy_input = torch.rand(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(dummy_input, ))
macs, params = clever_format([macs, params], "%.4f")
print(f"Params: {params}")  # 输出模型参数量
print(f"FLOPs: {macs}")  # 输出模型计算量

