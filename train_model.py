import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.optim as optim
from torch.nn.functional import one_hot

# Model sınıfını güncelle
class AnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnnModel,self).__init__()
        # Daha derin bir model
        self.fc1 = nn.Linear(input_size, hidden_size*2)  # İlk katmanı genişlet
        self.bn1 = nn.BatchNorm1d(hidden_size*2)
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.2)  # Dropout oranını azalt
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

# Veri yükleme
dataFrame = pd.read_excel("AllData.xlsx")

# Korelasyon analizi
from pandas.plotting import scatter_matrix
Draw = ["Number of Transformers","Power","Number Of Turn","Cooling Duct","Winding Height","Number Of Layers","Time"]
scatter_matrix(dataFrame[Draw], figsize=(12,8))
plt.savefig('correlation_matrix.png')
plt.close()

corrMatrix = dataFrame.corr()
print(abs(corrMatrix["Time"]).sort_values(ascending=False))

# Veriyi tensor'a çevir
dataTensor = torch.tensor(dataFrame.values)

# One-hot encoding
Cat_colam = [9]  # Wire Type sütunu
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
    
    # One-hot encoding uygula
    unique_values = len(torch.unique(dataTensor[:, column_index]))
    TempOneHot = one_hot(dataTensor[:, column_index].type(torch.LongTensor), num_classes=unique_values)
    tensor_without_column = torch.cat((dataTensor[:, :column_index], dataTensor[:, column_index+1:]), dim=1)
    dataTensor = torch.cat((tensor_without_column[:, :column_index], TempOneHot, tensor_without_column[:, column_index:]), dim=1)

print("After one-hot:", dataTensor.size())

# Veriyi eğitim ve test olarak böl
total_size = len(dataTensor)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataTensor, test_dataTensor = random_split(dataTensor.float(), [train_size, test_size])
train_dataTensor = torch.stack([sample for sample in train_dataTensor])

# Giriş ve çıkış verilerini ayır
input = train_dataTensor[:, :-1]
output = train_dataTensor[:, -1]

# Standardizasyon
standardized_column = range(0,9)
for i in standardized_column:
    column = input[:, i]
    mean = column.mean()
    std = column.std()
    input[:, i] = (column - mean) / std

# Dataset ve DataLoader oluştur
dataset = TensorDataset(input, output)
train_loader = DataLoader(dataset, batch_size=15, shuffle=True)

# Özellik ağırlıklandırma - Overall Coil Number için özel ağırlık
def weighted_l1_loss(pred, target, feature_weights):
    return torch.mean(feature_weights * torch.abs(pred - target))

# Model boyutunu ayarla
input_features = dataTensor.size(1) - 1
model = AnnModel(input_size=input_features, hidden_size=64, output_size=1)  # Hidden size artırıldı

# Loss function ve optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Learning rate düşürüldü

# Eğitim döngüsü
num_epochs = 15000  # Epoch sayısı artırıldı
train_losses = []
best_loss = float('inf')
patience = 500  # Early stopping için sabır
patience_counter = 0

for epoch in range(num_epochs):
    model.train(True)
    epoch_losses = []
    
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping ekle
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(avg_loss)
    
    # Early stopping kontrolü
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # En iyi modeli kaydet
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_features': input_features,
            'standardization_params': {
                'means': [input[:, i].mean().item() for i in standardized_column],
                'stds': [input[:, i].std().item() for i in standardized_column]
            }
        }, 'best_transformer_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Eğitim grafiğini kaydet
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title('Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.close()

# Test verisi değerlendirmesi
model.eval()
test_dataTensor = torch.stack([sample for sample in test_dataTensor])
test_input = test_dataTensor[:, :-1]
test_output = test_dataTensor[:, -1]

# Test verisini standardize et
for i in standardized_column:
    column = test_input[:, i]
    mean = column.mean()
    std = column.std()
    test_input[:, i] = (column - mean) / std

with torch.no_grad():
    test_predictions = model(test_input)
    test_predictions = test_predictions.squeeze()
    test_loss = criterion(test_predictions, test_output)
    
    print("\nTest Sonuçları:")
    for i in range(len(test_output)):
        print(f"Gerçek Değer: {test_output[i]:.2f}, Tahmin: {test_predictions[i].item():.2f}")
    
    print(f"\nTest Loss: {test_loss:.4f}")

print("\nModel başarıyla kaydedildi!") 