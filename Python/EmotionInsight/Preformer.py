# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.models as models

from torchtoolbox.transform import Cutout
from torchtoolbox.tools import mixup_data, mixup_criterion

from torch.autograd import Variable

# %%
lr = 1e-4
batchSize = 1024
workers = 16
epochs = 100
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
transfrom = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    Cutout(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transfromTest = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(),
    transforms.RandomAutocontrast(),
    transforms.RandomHorizontalFlip(),
    Cutout(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# %%
Labels = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}

# %%
import os
from PIL import Image
from torch.utils import data
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


# %%
class SeedlingData(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        self.transforms = transforms
        if self.test:
            imgs = [os.path.join(root, img) for img in os.listdir(root)]
            self.imgs = imgs
        else:
            img_labels = [os.path.join(root, img) for img in os.listdir(root)]
            imgs = []
            for imglabel in img_labels:
                for imgname in os.listdir(imglabel):
                    imgpath = os.path.join(imglabel, imgname)
                    imgs.append(imgpath)
            trainval_file, val_file = train_test_split(imgs,
                                                       test_size=0.3,
                                                       random_state=42)
            if train:
                self.imgs = trainval_file
            else:
                self.imgs = val_file

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_path = img_path.replace("\\", "/")
        if self.test:
            label = -1
        else:
            labelname = img_path.split('/')[-2]
            label = Labels[labelname]
        data = Image.open(img_path).convert('RGB')
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# %%
dataset_train = SeedlingData('./datasets/train',
                             transforms=transfrom,
                             train=True)
dataset_test = SeedlingData('./datasets/train',
                            transforms=transfrom,
                            train=False)

# %%
train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=batchSize,
    #num_workers=workers,
    pin_memory=True,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=batchSize,
    #num_workers=workers,
    pin_memory=True,
    shuffle=False)

# %%
criterion = nn.CrossEntropyLoss()
#criterion = SoftTargetCrossEntropy()
# model.fc = nn.Sequential(nn.Linear(2048,1024), nn.ReLU(), nn.Dropout(0.2),
#                          nn.Linear(512, 7), nn.LogSoftmax(dim=1))
# model.fc = nn.Sequential(nn.LogSoftmax(dim=1))
model = models.AlexNet(num_classes=7)
model.to(device)
# 选择简单暴力的Adam优化器，学习率调低
#optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
#optimizer = optim.SGD(model_ft.parameters(),lr=modellr)
optimizer = optim.RAdam(model.parameters(), lr=lr)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=20,
                                                       eta_min=1e-9)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(dataset_train), epochs=epochs)

# %%
EPOCHS_COUNT = 0
ACC_LIST = []
LOSS_LIST = []
ACC = 0

# %%
alpha = 0.2


# 验证过程
def val(model, device, test_loader):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(
                device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(avgloss, correct, len(test_loader.dataset), 100 * acc))
        ACC_LIST.append(acc)
        if acc > ACC:
            torch.save(
                model,
                'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            ACC = acc


def train(model, device, train_loader, optimizer, epoch, test_loader):
    for i in range(epoch):
        model.train()
        sum_loss = 0
        lr_now = lr
        total_num = len(train_loader.dataset)
        print(total_num, len(train_loader))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(
                device, non_blocking=True)
            data, labels_a, labels_b, lam = mixup_data(data, target, alpha)
            optimizer.zero_grad()
            # output = model(data)
            with torch.cuda.amp.autocast():
                loss = mixup_criterion(criterion, model(data), labels_a,
                                       labels_b, lam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scheduler.step()
            lr_now = scheduler.get_last_lr()
            scaler.update()
            # loss.backward()
            # optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr={}'.
                      format(epoch, (batch_idx + 1) * len(data),
                             len(train_loader.dataset),
                             100. * (batch_idx + 1) / len(train_loader),
                             loss.item(), lr_now))
        ave_loss = sum_loss / len(train_loader)
        LOSS_LIST.append(ave_loss)
        print('Epoch:{},loss:{},lr:{}'.format(epoch, ave_loss, lr_now))
        cosine_schedule.step()
        val(model, device, test_loader)


# %%
if __name__ == '__main__':
    num_processes = mp.cpu_count()
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train,
                       args=(model, device, train_loader, optimizer, epochs,
                             test_loader))
        p.start
        processes.append(p)
        for p in processes:
            p.join()

# %%
sns.set(palette='twilight')
sns.relplot(kind='line', data=ACC_LIST)
plt.xlabel("Epoch Time")
plt.ylabel("Accuracy")
#plt.ylim(top=1,bottom=0)
sns.relplot(kind='line', data=LOSS_LIST)
plt.xlabel("Epoch Time")
plt.ylabel("Loss")
