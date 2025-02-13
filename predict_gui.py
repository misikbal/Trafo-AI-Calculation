import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                           QDoubleSpinBox, QComboBox, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import torch
from predict import AnnModel, predict_transformer_time

class ModernSpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QSpinBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 3px;
                min-width: 100px;
                min-height: 25px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3d3d3d;
                border-radius: 3px;
            }
        """)

class ModernDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 3px;
                min-width: 100px;
                min-height: 25px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #3d3d3d;
                border-radius: 3px;
            }
        """)

class TransformerPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Transformatör Üretim Süresi Tahmini')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QComboBox {
                background-color: #2b2b2b;
                color: white;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                padding: 3px;
                min-width: 100px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid none;
                border-right: 5px solid none;
                border-top: 5px solid white;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
        """)

        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Başlık
        title = QLabel('Transformatör Üretim Süresi Tahmini')
        title.setStyleSheet("""
            font-size: 20px;
            color: #ffffff;
            margin-bottom: 10px;
            padding: 8px;
            background-color: #2b2b2b;
            border-radius: 8px;
        """)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Input container
        input_container = QFrame()
        input_container.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        input_layout.setSpacing(8)

        # Input alanları
        self.inputs = {}
        
        # Sayısal girişler
        numerical_inputs = [
            ("Number of Transformers", 0, 1000),
            ("Overall Coil Number", 0, 1000),
            ("Power", 0, 1000),
            ("Number Of Turn", 0, 1000),
            ("Cooling Duct", 0, 1000),
            ("Winding Height", 0, 1000),
            ("Number Of Layers", 0, 1000),
        ]

        for name, min_val, max_val in numerical_inputs:
            h_layout = QHBoxLayout()
            label = QLabel(name + ":")
            label.setMinimumWidth(150)
            spinbox = ModernSpinBox()
            spinbox.setRange(min_val, max_val)
            self.inputs[name] = spinbox
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            input_layout.addLayout(h_layout)

        # Ondalıklı sayılar
        double_inputs = [
            ("Wire Width", 0.0, 100.0),
            ("Wire Thickness", 0.0, 100.0),
        ]

        for name, min_val, max_val in double_inputs:
            h_layout = QHBoxLayout()
            label = QLabel(name + ":")
            label.setMinimumWidth(150)
            spinbox = ModernDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setDecimals(2)
            self.inputs[name] = spinbox
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            input_layout.addLayout(h_layout)

        # Wire Type
        h_layout = QHBoxLayout()
        label = QLabel("Wire Type:")
        label.setMinimumWidth(150)
        self.wire_type_combo = QComboBox()
        self.wire_type_combo.addItems(['Tip 1', 'Tip 2', 'Tip 3', 'Tip 4', 'Tip 5'])
        h_layout.addWidget(label)
        h_layout.addWidget(self.wire_type_combo)
        input_layout.addLayout(h_layout)

        main_layout.addWidget(input_container)

        # Tahmin butonu
        self.predict_button = QPushButton('Tahmin Et')
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.clicked.connect(self.make_prediction)
        main_layout.addWidget(self.predict_button)

        # Sonuç etiketi
        self.result_label = QLabel('')
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 15px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                color: #4caf50;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

    def make_prediction(self):
        try:
            # Değerleri topla
            features = [
                self.inputs["Number of Transformers"].value(),
                self.inputs["Overall Coil Number"].value(),
                self.inputs["Power"].value(),
                self.inputs["Number Of Turn"].value(),
                self.inputs["Cooling Duct"].value(),
                self.inputs["Winding Height"].value(),
                self.inputs["Number Of Layers"].value(),
                self.inputs["Wire Width"].value(),
                self.inputs["Wire Thickness"].value(),
            ]

            # Wire Type için one-hot encoding
            wire_type = self.wire_type_combo.currentIndex()
            checkpoint = torch.load('zeki_transformer_model.pth')
            one_hot_size = checkpoint['input_features'] - len(features)
            wire_type_one_hot = [0] * one_hot_size
            if wire_type < one_hot_size:
                wire_type_one_hot[wire_type] = 1
            features.extend(wire_type_one_hot)

            # Tahmin yap
            tahmin_edilen_sure = predict_transformer_time(features)
            tahmin_edilen_sure = int(tahmin_edilen_sure)/60
            
            # Sonucu göster
            self.result_label.setText(f'Tahmini Üretim Süresi: {tahmin_edilen_sure+6}')
            
        except Exception as e:
            QMessageBox.critical(self, 'Hata', f'Bir hata oluştu: {str(e)}')

def main():
    app = QApplication(sys.argv)
    
    # Uygulama genelinde font ayarı
    app.setFont(QFont('Segoe UI', 9))
    
    window = TransformerPredictorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 