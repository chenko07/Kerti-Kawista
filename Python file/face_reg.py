# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder
# from torchvision.models import resnet18

# import pandas as pd
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T
# import tqdm
# import torch.nn as nn
# from torchvision.utils import make_grid


# # Contoh class ResNet15 dari implementasi sebelumnya

# # Persiapan Dataset
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])

# train_dataset = ImageFolder('Face Recognition', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# def F_score(output, label, threshold=0.5, beta=1): #Calculate the accuracy of the model
#     prob = output > threshold
#     label = label > threshold

#     TP = (prob & label).sum(1).float()
#     TN = ((~prob) & (~label)).sum(1).float()
#     FP = (prob & (~label)).sum(1).float()
#     FN = ((~prob) & label).sum(1).float()

#     precision = torch.mean(TP / (TP + FP + 1e-12))
#     recall = torch.mean(TP / (TP + FN + 1e-12))
#     F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
#     return F2.mean(0)

# # Rancang Model
# # Fungsi blok konvolusi
# def conv_block(in_channels, out_channels, pool=False):
#     layers = [
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True),
#     ]
#     if pool:
#         layers.append(nn.MaxPool2d(2))
#     return nn.Sequential(*layers)


# class MultilabelImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, targets = batch 
#         out = self(images)                            # Generate predictions
#         loss = F.binary_cross_entropy(out, targets)   # Calculate loss
#         return loss    

#     def validation_step(self, batch):
#         images, targets = batch 
#         out = self(images)                           # Generate predictions
#         loss = F.binary_cross_entropy(out, targets)  # Calculate loss
#         score = F_score(out, targets)                # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_score': score.detach() }      


#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()       # Combine losses and get the mean value
#         batch_scores = [x['val_score'] for x in outputs]    
#         epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies and get the mean value
#         return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}    

#     def epoch_end(self, epoch, result):                     # display the losses
#         print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}"
#               .format(epoch, 
#                       result['lrs'][-1], 
#                       result['train_loss'], 
#                       result['val_loss'], 
#                       result['val_score']))

# # Kelas ResNet15
# class ResNet15(MultilabelImageClassificationBase):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()     
#         # Input: 3 x 128 x 128
#         self.conv1 = conv_block(in_channels, 64)  # Output: 64 x 128 x 128
#         self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))  # Output: 64 x 128 x 128

#         self.conv2 = conv_block(64, 128, pool=True)  # Output: 128 x 64 x 64
#         self.res2 = nn.Sequential(conv_block(128, 128), conv_block(128, 128), conv_block(128, 128))  # Output: 128 x 64 x 64  

#         self.conv3 = conv_block(128, 512, pool=True)  # Output: 512 x 32 x 32
#         self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))  # Output: 512 x 32 x 32

#         self.conv4 = conv_block(512, 1024, pool=True)  # Output: 1024 x 16 x 16
#         self.res4 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))  # Output: 1024 x 16 x 16

#         self.classifier = nn.Sequential(nn.MaxPool2d(2),  # Output: 1024 x 8 x 8
#                                         nn.Flatten(), 
#                                         nn.Dropout(0.2),
#                                         nn.Linear(1024 * 8 * 8, 512),  # Output: 512
#                                         nn.ReLU(),
#                                         nn.Linear(512, num_classes))  # Output: num_classes (jumlah kelas)

#     def forward(self, xb):
#         out = self.conv1(xb)
#         out = self.res1(out) + out
#         out = self.conv2(out)
#         out = self.res2(out) + out
#         out = self.conv3(out)
#         out = self.res3(out) + out
#         out = self.conv4(out)
#         out = self.res4(out) + out
#         out = self.classifier(out)
#         out = torch.sigmoid(out)
#         return out

# # Tentukan jumlah kelas (chen, nicholas, rendie, alvian)
# num_classes = 4

# # Inisialisasi model dan fungsi kerugian
# model = ResNet15(in_channels=3, num_classes=num_classes)
# criterion = nn.CrossEntropyLoss()

# # Tentukan optimizer dan learning rate
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Map kelas ke integer labels (labels_map)
# labels_map = train_dataset.class_to_idx
# # Reverse mapping untuk prediksi label kembali ke nama kelas
# idx_to_class = {v: k for k, v in labels_map.items()}

# # Proses Pelatihan
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Simpan model setelah pelatihan
# torch.save(model.state_dict(), 'resnet_face_recognition_model.pth')
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in train_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy on Test Set: {100 * correct / total:.2f}%')

# # Deteksi Wajah pada Gambar Baru
# import cv2

# # Fungsi deteksi wajah menggunakan model yang telah dilatih
# def detect_face(image_path):
#     model.eval()
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (128, 128))
#     image = transform(image).unsqueeze(0)
    
#     with torch.no_grad():
#         output = model(image)
#         _, predicted_label = torch.max(output, 1)
#         predicted_class = idx_to_class[predicted_label.item()]
#         return predicted_class

# # Pengenalan Wajah pada Gambar Baru
# image_path = 'path/to/your/new_face.jpg'
# predicted_class = detect_face(image_path)
# print('Detected Face Class:', predicted_class)
# # Evaluasi Model

