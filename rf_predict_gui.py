import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                           QDoubleSpinBox, QMessageBox, QFrame, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pandas as pd
import joblib

class RFPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Transformatör Tahmin')
        self.setGeometry(100, 100, 600, 500)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-size: 12px; }
            QPushButton { 
                background-color: #0d47a1; 
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #3d3d3d;
                border-radius: 5px;
                padding: 5px;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: white;
                selection-background-color: #0d47a1;
            }
            QSpinBox, QDoubleSpinBox { 
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                padding: 3px;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Model seçim container'ı
        model_selection_container = QFrame()
        model_selection_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 5px;
            }
        """)
        model_selection_layout = QHBoxLayout(model_selection_container)
        
        # Model seçim başlığı
        model_selection_title = QLabel("Model Seçimi:")
        model_selection_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        model_selection_layout.addWidget(model_selection_title)

        # Dropdown menu için ComboBox
        self.model_combo = QComboBox()
        self.models = [
            ("AG Yassı Tel", "ag_yassi_tel.joblib"),
            ("Yuvarlak Tel", "YG_Emaye.joblib"),
            ("Folyo", "folyo.joblib")
        ]
        
        # ComboBox'a modelleri ekle
        for name, _ in self.models:
            self.model_combo.addItem(name)
        
        # Model değişikliğinde çağrılacak fonksiyon
        self.model_combo.currentIndexChanged.connect(self.load_model)
        model_selection_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_selection_container)

        # Input container
        input_container = QFrame()
        input_container.setStyleSheet("QFrame { background-color: #2b2b2b; padding: 15px; }")
        input_layout = QVBoxLayout(input_container)

        # Input alanları
        self.inputs = {}
        
        # Temel parametreler
        parameters = [
            ("Number of Transformers", QSpinBox, (0, 5000)),
            ("Overall Coil Number", QSpinBox, (0, 5000)),
            ("Power", QSpinBox, (0, 5000)),
            ("Number Of Turn", QSpinBox, (0, 10000)),
            ("Cooling Duct", QSpinBox, (0, 5000)),
            ("Winding Height", QSpinBox, (0, 5000)),
            ("Number Of Layers", QSpinBox, (0, 5000)),
            ("Wire Width", QDoubleSpinBox, (0.0, 5000.0)),
            ("Wire Thickness", QDoubleSpinBox, (0.0, 5000.0)),
            ("Wire Type", QSpinBox, (0, 100)),
            ("AG", QSpinBox, (0, 100))
        ]

        for name, spinbox_type, (min_val, max_val) in parameters:
            h_layout = QHBoxLayout()
            label = QLabel(f"{name}:")
            label.setMinimumWidth(150)
            
            spinbox = spinbox_type()
            spinbox.setRange(min_val, max_val)
            if spinbox_type == QDoubleSpinBox:
                spinbox.setDecimals(2)
            
            self.inputs[name] = spinbox
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            input_layout.addLayout(h_layout)

        layout.addWidget(input_container)

        # Tahmin butonu
        self.predict_button = QPushButton('Tahmin Et')
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_button)

        # Sonuç etiketi
        self.result_label = QLabel('')
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 15px;
                font-size: 16px;
                color: #4caf50;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        # İlk modeli yükle
        self.load_model()

    def load_model(self):
        try:
            # Seçili modelin dosya adını al
            _, selected_model = self.models[self.model_combo.currentIndex()]
            
            # Modeli yükle
            self.model_data = joblib.load(selected_model)
            self.model = self.model_data['model']
            self.feature_order = self.model_data['feature_order']
            print(f"Model yüklendi: {selected_model}")
            
            # Sonuç etiketini temizle
            self.result_label.setText('')
            
        except Exception as e:
            self.model = None
            self.feature_order = None
            self.model_data = None
            print(f"Model yükleme hatası: {str(e)}")
            QMessageBox.warning(self, 'Uyarı', f'Model dosyası yüklenemedi: {selected_model}')

    def make_prediction(self):
        try:
            if self.model is None:
                raise Exception("Model yüklenemedi!")

            input_data = {name: [spinbox.value()] 
                         for name, spinbox in self.inputs.items()}
            
            new_transformer = pd.DataFrame(input_data)
            new_transformer = new_transformer[self.feature_order]
            
            predicted_time = self.model.predict(new_transformer)[0]
            
            # Tüm süre değerlerini al
            mean_time = self.model_data.get('mean_time', 0)
            optimal_time = self.model_data.get('optimal_time', 0)
            typical_time = self.model_data.get('typical_time', 0)
            max_efficient_time = self.model_data.get('max_efficient_time', 0)
            
            # Sonuçları göster
            result_text = (
                f'Tahmini Üretim Süresi: {predicted_time:.1f} saat ({predicted_time/8:.1f} gün)\n'
                f'En İyi Süre: {optimal_time:.1f} saat ({optimal_time/8:.1f} gün)\n'
                f'Tipik Süre: {typical_time:.1f} saat ({typical_time/8:.1f} gün)\n'
                f'Ortalama Süre: {mean_time:.1f} saat ({mean_time/8:.1f} gün)\n'
                f'Verimli Maksimum Süre: {max_efficient_time:.1f} saat ({max_efficient_time/8:.1f} gün)'
            )
            self.result_label.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(self, 'Hata', f'Bir hata oluştu: {str(e)}')

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 9))
    window = RFPredictorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 