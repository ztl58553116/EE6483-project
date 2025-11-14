import os

# 设置代理环境变量
os.environ['http_proxy'] = 'http://turbo2.gpushare.com:30000'
os.environ['https_proxy'] = 'http://turbo2.gpushare.com:30000'

import torch
import cv2
import glob
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

device = 'cuda'

def seed_setting(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_setting(36)

def worker_init_fn(idx):
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    
transform_train_img = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=15),
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation = 0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val_img = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 128 
train_dataset = datasets.ImageFolder(root='./cifar10_expanded/train', transform=transform_train_img)
val_dataset = datasets.ImageFolder(root='./cifar10_expanded/val', transform=transform_val_img)
test_dataset = datasets.ImageFolder(root='./cifar10_expanded/test', transform=transform_val_img)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)

input_features = model.classifier[1].in_features

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(input_features, 10)
)

model = model.to(device)

for name, param in model.features.named_parameters():
    print(name)
    if '18' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
        
epoch_nums = 20

loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 2e-3) ###
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

training_loss_stat = []
training_accuracy_stat = []

val_loss_stat = []
val_accuracy_stat = []

patience_times = 5
minimum_loss = 1e9
best_loss = 0
best_accuracy = 0
training_epochs = 0
count = 0

for epoch in range(epoch_nums):
    model.train()
    train_loss = 0
    train_correct = train_total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted_labels = torch.max(outputs, 1)
        check_labels = (predicted_labels == labels)
        train_correct += check_labels.sum().item()
        train_total += labels.size(0) ###
        
    training_accuracy = train_correct / train_total * 100
    train_average_loss = train_loss / len(train_loader)

    training_loss_stat.append(train_average_loss)
    training_accuracy_stat.append(training_accuracy)
    
    print(f"epoch: {epoch + 1} / {epoch_nums}, Training loss: {train_average_loss: .4f}, Training accuracy: {training_accuracy: .2f}%", end = ' ')
    

    model.eval()
    val_loss = 0
    val_correct = val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            
            _, predicted_labels = torch.max(outputs, 1)
            check_labels = (predicted_labels == labels)
            val_correct += check_labels.sum().item()
            val_total += labels.size(0)
    
    
    val_accuracy = val_correct / val_total * 100
    val_average_loss = val_loss / len(val_loader)

    if val_average_loss < minimum_loss:
        minimum_loss = val_average_loss
        torch.save(model.state_dict(), './best_model.pth')
        count = 0
    else:
        count += 1

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

    val_loss_stat.append(val_average_loss)
    val_accuracy_stat.append(val_accuracy)
    
    print(f"Validation loss: {val_average_loss: .4f}, Validation accuracy: {val_accuracy: .2f}%\n")

    scheduler.step()

    training_epochs = epoch + 1
    if count >= patience_times:
        break

    
    
print("Training done!")


print(f"minimum validation loss: {minimum_loss}, best_accuracy: {best_accuracy}")

fig, ax = plt.subplots()

ax.plot(np.arange(1, training_epochs + 1), np.array(training_loss_stat), label='train', color='blue')
ax.plot(np.arange(1, training_epochs + 1), np.array(val_loss_stat), label='val', color='red')

ax.set_title('Training and validation loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.legend()

plt.show()

fig, ax = plt.subplots()

ax.plot(np.arange(1, training_epochs + 1), np.array(training_accuracy_stat), label='train', color='blue')
ax.plot(np.arange(1, training_epochs + 1), np.array(val_accuracy_stat), label='val', color='red')

ax.set_title('Training and validation accuracy')
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')
ax.legend()

plt.show()

model.load_state_dict(torch.load('./best_model.pth', map_location=device))
model = model.to(device)
model.eval()

pred_labels = []
real_labels = []
test_correct = test_total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        check_labels = (preds == labels)
        test_correct += check_labels.sum().item()
        test_total += labels.size(0)
        
        pred_labels.extend(list(preds.cpu()))
        real_labels.extend(list(labels.cpu()))
print(f"Accuracy: {test_correct / test_total * 100: .2f}%")

cm = confusion_matrix(real_labels, pred_labels)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()



