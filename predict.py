import torch
import torch.nn as nn
from torch.nn.functional import one_hot

# Aynı model sınıfını tanımlayın
class AnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AnnModel,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)  
        self.fc3 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc4 = nn.Linear(hidden_size//4, output_size)
        self.dropout = nn.Dropout(0.2)
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

def predict_time(num_transformers, overall_coil_num, power, num_turns, cooling_duct, 
                 winding_height, num_layers, wire_width, wire_thickness):
    # Model ve parametreleri yükle
    checkpoint = torch.load('zeki_transformer_model.pth')
    
    # Sabit değerlerle modeli oluştur
    input_size = checkpoint.get('input_size', 16)  # Varsayılan değer
    hidden_size = checkpoint.get('hidden_size', 64)  # Varsayılan değer
    output_size = 1  # Süre tahmini için tek çıktı
    
    # Modeli oluştur ve ağırlıkları yükle
    model = AnnModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Giriş verilerini bir liste haline getir
    input_data = [
        num_transformers, overall_coil_num, power, num_turns,
        cooling_duct, winding_height, num_layers, wire_width, wire_thickness
    ]
    
    # Giriş verilerini tensöre çevir
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1)
    
    # Standardizasyon uygula
    for i in checkpoint['standardized_column']:
        mean = checkpoint['input_means'][i]
        std = checkpoint['input_stds'][i]
        input_tensor[:, i] = (input_tensor[:, i] - mean) / std
    
    # Tahmin yap
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Tahmini denormalize et
    denormalized_prediction = prediction.item() * checkpoint['output_std'] + checkpoint['output_mean']
    
    return denormalized_prediction

# Örnek kullanım
if __name__ == "__main__":
    # Örnek kullanım
    predicted_time = predict_time(
        num_transformers=2,
        overall_coil_num=4,
        power=100,
        num_turns=50,
        cooling_duct=3,
        winding_height=500,
        num_layers=4,
        wire_width=2.5,
        wire_thickness=1.5
    )
    
    print(f"Tahmini süre: {predicted_time:.2f} dakika") 