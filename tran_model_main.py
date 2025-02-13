import torch
import pandas as pd
import matplotlib.pyplot as plt

dataFrame=pd.read_excel("AllData.xlsx")
dataFrame.info()


from pandas.plotting import scatter_matrix

Draw=["Number of Transformers","Power","Number Of Turn","Cooling Duct","Winding Height","Number Of Layers","Time"];    
scatter_matrix(dataFrame[Draw],figsize=(12,8))
plt.show()
corrMatrix=dataFrame.corr()
print(abs(corrMatrix["Time"]).sort_values(ascending=False))


dataTensor=torch.tensor(dataFrame.values)
Cat_colam=[9,10]


from torch.nn.functional import one_hot
print(dataTensor.size())
for column_index in Cat_colam:
    # Replace the column with one-hot encoding
    # Remove the column to be replaced
    TempOneHot=one_hot(dataTensor[:,column_index].type(torch.LongTensor))
    tensor_without_column = torch.cat((dataTensor[:, :column_index], dataTensor[:, column_index+1:]), dim=1)
    dataTensor= torch.cat((tensor_without_column[:, :column_index], TempOneHot, tensor_without_column[:, column_index:]), dim=1)

print(dataTensor.size())

from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset

# Önce toplam veri seti boyutunu alalım
total_size = len(dataTensor)
# Train size'ı toplam verinin %80'i olarak ayarlayalım
train_size = int(0.8 * total_size)
# Test size'ı kalan veri olarak ayarlayalım
test_size = total_size - train_size

print(f"Toplam veri boyutu: {total_size}")
print(f"Eğitim seti boyutu: {train_size}")
print(f"Test seti boyutu: {test_size}")

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

# Output normalization
output_mean = output.mean()
output_std = output.std()
normalized_output = (output - output_mean) / output_std

# Create dataset with normalized output
dataset = TensorDataset(input, normalized_output)
train_loader = DataLoader(dataset, batch_size=15, shuffle=True)


import torch.nn as nn
class AnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnnModel,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)  
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc4 = nn.Linear(hidden_size//4, output_size)
        self.dropout = nn.Dropout(0.2)  # Dropout oranını düşürdük
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//4)
        self.rl = nn.ReLU()

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
        x = self.bn3(x)
        x = self.rl(x)
        
        x = self.fc4(x)
        return x

# Veri boyutunu kontrol etmek için print ekleyelim
print("Input shape:", input.shape)  # Giriş verisi boyutunu göster

# Model parametrelerini güncelleyelim
model = AnnModel(input_size=input.shape[1], hidden_size=64, output_size=1)


import torch.optim as optim

criterion = nn.L1Loss()

# Optimizer ve learning rate ayarları
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Learning rate scheduler ekleyelim
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)

# Training loop
num_epochs = 10000
best_loss = float('inf')
patience_counter = 0
patience_limit = 1000  # Early stopping için sabır limiti

for epoch in range(num_epochs):
    epoch_losses = []
    for batch_features, batch_labels in train_loader:
        model.train(True)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    # Epoch'un ortalama loss değeri
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    
    # Learning rate scheduler'ı güncelle
    scheduler.step(avg_loss)
    
    # Early stopping kontrolü
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience_limit:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 100 == 0:  # Her 100 epoch'ta bir loss değerini yazdır
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Training loop kısmından sonra, modeli test moduna alıp tahmin yapalım
model.eval()  # Modeli değerlendirme moduna al

# Test verisini tensöre çevir
test_dataTensor = torch.stack([sample for sample in test_dataTensor])

# Test verisi için input ve output ayır
test_input = test_dataTensor[:, :-1]  # Son sütun hariç tüm sütunlar
test_output = test_dataTensor[:, -1]  # Sadece son sütun

# Test verisini de aynı şekilde standardize et
for i in standardized_column:
    column = test_input[:, i]
    mean = column.mean()
    std = column.std()
    test_input[:, i] = (column - mean) / std

# Tahmin yap
with torch.no_grad():  # Gradyan hesaplama
    predictions = model(test_input.float())

# Tahminleri denormalize et
denormalized_predictions = predictions.squeeze() * output_std + output_mean
print("Tahmin Değerleri:", denormalized_predictions.numpy())

# Ortalama mutlak hata hesapla
mae = criterion(denormalized_predictions, test_output)
print(f"\nTest Verisi Üzerinde Ortalama Mutlak Hata: {mae.item():.4f}")

# Early stopping sonrası modeli kaydet
model_info = {
    'model_state_dict': model.state_dict(),
    'input_size': input.shape[1],
    'hidden_size': 64,
    'output_size': 1,
    'output_mean': output_mean,
    'output_std': output_std,
    'standardized_column': standardized_column,
    'input_means': {i: input[:, i].mean().item() for i in standardized_column},
    'input_stds': {i: input[:, i].std().item() for i in standardized_column}
}

torch.save(model_info, 'zeki_transformer_model.pth')
print("Model başarıyla kaydedildi: zeki_transformer_model.pth")
