import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # TkAgg yerine Agg kullan
import matplotlib.pyplot as plt


dataFrame=pd.read_excel("AllData.xlsx")


from pandas.plotting import scatter_matrix

Draw=["Number of Transformers","Power","Number Of Turn","Cooling Duct","Winding Height","Number Of Layers","Time"];    
scatter_matrix(dataFrame[Draw],figsize=(12,8))
plt.show()
corrMatrix=dataFrame.corr()
print(abs(corrMatrix["Time"]).sort_values(ascending=False))


dataTensor=torch.tensor(dataFrame.values)

# One-hot encoding öncesi kaç farklı Wire Type olduğunu görelim
print("Unique Wire Types:", len(dataFrame['Wire Type'].unique()))

# One-hot encoding kısmını düzenleyelim
Cat_colam=[9]  # Wire Type sütunu
from torch.nn.functional import one_hot
print("Before one-hot:", dataTensor.size())
for column_index in Cat_colam:
    # Sütundaki değerleri kontrol et
    column_values = dataTensor[:, column_index]
    print("Column values:", column_values)
    print("Max value:", torch.max(column_values))
    
    # Değerleri 0'dan başlayacak şekilde dönüştür
    unique_vals = torch.unique(column_values)
    value_map = {val.item(): idx for idx, val in enumerate(unique_vals)}
    
    # Değerleri yeniden numaralandır
    new_values = torch.tensor([value_map[val.item()] for val in column_values])
    dataTensor[:, column_index] = new_values
    
    # Get unique values count for proper one-hot size
    unique_values = len(torch.unique(dataTensor[:, column_index]))
    print("Unique values after mapping:", unique_values)
    print("Max value after mapping:", torch.max(dataTensor[:, column_index]))
    
    # Replace the column with one-hot encoding
    TempOneHot = one_hot(dataTensor[:, column_index].type(torch.LongTensor), num_classes=unique_values)
    # Remove the column to be replaced
    tensor_without_column = torch.cat((dataTensor[:, :column_index], dataTensor[:, column_index+1:]), dim=1)
    # Add one-hot encoded columns
    dataTensor = torch.cat((tensor_without_column[:, :column_index], TempOneHot, tensor_without_column[:, column_index:]), dim=1)

print("After one-hot:", dataTensor.size())


from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
# Define the split sizes
total_size = len(dataTensor)
train_size = int(0.8 * total_size)  # %80'i training için
test_size = total_size - train_size  # Kalan %20'si test için

# Use random_split to split the data
train_dataTensor, test_dataTensor = random_split(dataTensor.float(), [train_size, test_size])
# Convert `train_dataTensor` to a tensor
train_dataTensor = torch.stack([sample for sample in train_dataTensor])

# Extract input (all columns except last) and output (last column)
input = train_dataTensor[:, :-1]  # All rows, all columns except the last
output = train_dataTensor[:, -1]  # All rows, only the last column
standardized_column=range(0,9)
for i in standardized_column:
    column = input[:, i]
    # Calculate the mean and standard deviation
    mean = column.mean()
    std = column.std()
    # Standardize the column (mean 0, std 1)
    stand_column = (column - mean) / std
    # Replace the column in the tensor
    input[:, i] = stand_column

# Create dataset and dataloader
dataset = TensorDataset(input, output)
train_loader = DataLoader(dataset, batch_size=15, shuffle=True)


import torch.nn as nn
class AnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnnModel,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization ekle
        self.rl = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)  # Dropout oranını düşürdüm

    def forward(self,x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.rl(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Veri boyutunu kontrol et
print("Input shape:", input.shape)  # Giriş verilerinin boyutunu görmek için

# Model boyutunu otomatik ayarlayalım
input_features = dataTensor.size(1) - 1  # Son sütun (Time) hariç
model = AnnModel(input_size=input_features, hidden_size=25, output_size=1)

# Veri boyutlarını kontrol et
print("Input features shape:", input.shape)
print("One-hot encoded features:", input.shape[1])

import torch.optim as optim

criterion = nn.L1Loss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
    


# Training loop
num_epochs = 10000
train_losses = []  # Eğitim kayıplarını takip etmek için
val_losses = []    # Doğrulama kayıplarını takip etmek için

for epoch in range(num_epochs):
    model.train(True)
    epoch_losses = []
    
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 100 == 0:  # Her 100 epoch'ta bir yazdır
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Eğitim sonrası kayıp grafiğini çiz
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test verisi için değerlendirme
model.eval()
test_dataTensor = torch.stack([sample for sample in test_dataTensor])
test_input = test_dataTensor[:, :-1]
test_output = test_dataTensor[:, -1]

# Test verisini de standardize et
for i in standardized_column:
    column = test_input[:, i]
    mean = column.mean()
    std = column.std()
    test_input[:, i] = (column - mean) / std

with torch.no_grad():
    test_predictions = model(test_input)
    # Boyutları uyumlu hale getir
    test_predictions = test_predictions.squeeze()  # [53, 1] -> [53] boyutuna çevir
    test_loss = criterion(test_predictions, test_output)
    
    # Tahmin performansını göster
    for i in range(len(test_output)):
        print(f"Gerçek Değer: {test_output[i]:.2f}, Tahmin: {test_predictions[i].item():.2f}")
    
    print(f"\nTest Loss: {test_loss:.4f}")

def predict_transformer_time(model, features):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
        
        for i in standardized_column:
            column = features_tensor[:, i]
            mean = input[:, i].mean()
            std = input[:, i].std()
            features_tensor[:, i] = (column - mean) / std
        
        prediction = model(features_tensor)
        return prediction.squeeze().item()  # Boyutu düzelt

# Önce model boyutunu görelim
print("Model input features:", model.fc1.weight.shape[1])

# Örnek trafoyu model boyutuna uygun şekilde hazırlayalım
ornek_trafo = [
    10,      # Number of Transformers
    30,      # Overall Coil Number
    450,    # Power
    79,     # Number Of Turn
    5,      # Cooling Duct
    400,    # Winding Height
    6,      # Number Of Layers
    22.40,  # Wire Width
    3.,   # Wire Thickness
]

# Wire Type için one-hot değerlerini ekle
wire_type = 1  # Hangi tip olduğunu belirtin (0'dan başlayarak)
one_hot_size = model.fc1.weight.shape[1] - len(ornek_trafo)  # Gereken one-hot boyutu
wire_type_one_hot = [0] * one_hot_size  # Önce tüm değerleri 0 yap
if wire_type < one_hot_size:
    wire_type_one_hot[wire_type] = 1  # Sadece seçili tipi 1 yap
ornek_trafo.extend(wire_type_one_hot)

# Kontrol edelim
print("Example input size:", len(ornek_trafo))
print("Model input size:", model.fc1.weight.shape[1])

if len(ornek_trafo) == model.fc1.weight.shape[1]:
    tahmin_edilen_sure = predict_transformer_time(model, ornek_trafo)
    tahmin_edilen_sure=int(tahmin_edilen_sure)/60
    print(f"\nTahmini Üretim Süresi: {tahmin_edilen_sure} dakika")
else:
    print("Hata: Örnek trafo boyutu model giriş boyutuyla uyuşmuyor!")
    print(f"Örnek trafo boyutu: {len(ornek_trafo)}")
    print(f"Model giriş boyutu: {model.fc1.weight.shape[1]}")
