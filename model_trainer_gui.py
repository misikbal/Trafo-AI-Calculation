import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog,
                           QMessageBox, QFrame, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

class ModelTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Eğitim Arayüzü')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { 
                color: #ffffff; 
                font-size: 12px;
            }
            QPushButton { 
                background-color: #0d47a1; 
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QProgressBar {
                border: 2px solid #2b2b2b;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4caf50;
            }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Başlık
        title = QLabel('Model Eğitim Arayüzü')
        title.setStyleSheet("""
            font-size: 24px;
            color: #ffffff;
            margin-bottom: 20px;
            padding: 10px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Dosya seçim container'ı
        file_container = QFrame()
        file_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                padding: 20px;
                border-radius: 5px;
            }
        """)
        file_layout = QVBoxLayout(file_container)

        # Dosya seçim butonu
        self.file_button = QPushButton('Excel Dosyası Seç')
        self.file_button.setCursor(Qt.PointingHandCursor)
        self.file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_button, alignment=Qt.AlignCenter)

        # Seçili dosya etiketi
        self.file_label = QLabel('Dosya seçilmedi')
        self.file_label.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.file_label)

        layout.addWidget(file_container)

        # Model eğitim container'ı
        train_container = QFrame()
        train_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                padding: 20px;
                border-radius: 5px;
            }
        """)
        train_layout = QVBoxLayout(train_container)

        # Model tipi seçimi
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel('Model Tipi:')
        self.model_type_buttons = {}
        
        for model_type in ['AG Yassı Tel', 'YG Emaye', 'Folyo']:
            btn = QPushButton(model_type)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=model_type: self.select_model_type(m))
            model_type_layout.addWidget(btn)
            self.model_type_buttons[model_type] = btn

        train_layout.addLayout(model_type_layout)

        # Eğitim butonu
        self.train_button = QPushButton('Modeli Eğit')
        self.train_button.setCursor(Qt.PointingHandCursor)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        train_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        layout.addWidget(train_container)

        # Sonuç container'ı
        result_container = QFrame()
        result_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                padding: 20px;
                border-radius: 5px;
            }
        """)
        result_layout = QVBoxLayout(result_container)

        # Sonuç etiketi
        self.result_label = QLabel('')
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)

        layout.addWidget(result_container)

        # Değişkenler
        self.selected_file = None
        self.selected_model_type = None

        # Dark mode message box style
        self.message_box_style = """
            QMessageBox {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
                min-width: 80px;
                font-size: 13px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1565c0;
            }
        """

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Excel Dosyası Seç",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        if file_name:
            self.selected_file = file_name
            self.file_label.setText(f'Seçili dosya: {file_name.split("/")[-1]}')
            self.check_ready()

    def select_model_type(self, model_type):
        # Diğer butonları temizle
        for btn in self.model_type_buttons.values():
            btn.setChecked(False)
        # Seçili butonu işaretle
        self.model_type_buttons[model_type].setChecked(True)
        self.selected_model_type = model_type
        self.check_ready()

    def check_ready(self):
        # Dosya ve model tipi seçili ise eğitim butonunu aktifleştir
        self.train_button.setEnabled(
            self.selected_file is not None and 
            self.selected_model_type is not None
        )

    def show_styled_message_box(self, icon, title, text):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStyleSheet(self.message_box_style)
        return msg.exec_()

    def train_model(self):
        try:
            # Excel dosyasını oku
            data = pd.read_excel(self.selected_file)
            
            # Sütun isimlerini düzelt
            if 'Wire Widhth' in data.columns:
                data.rename(columns={'Wire Widhth': 'Wire Width'}, inplace=True)

            # NaN değerleri temizle
            data = data.dropna(subset=['Time'])
            
            # Özellikler ve hedef değişken
            X = data.drop('Time', axis=1)
            y = data['Time']

            # NaN değerleri 0 ile doldur
            X = X.fillna(0)

            # Eğitim ve test setleri
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Model eğitimi
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Test seti üzerinde tahmin
            y_pred = model.predict(X_test)

            # Performans metrikleri
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = (1 - mae/y_test.mean()) * 100  # Doğruluk yüzdesi

            # Süre hesaplamaları
            mean_time = y.mean()
            optimal_time = y.quantile(0.25)
            typical_time = y.median()
            max_efficient_time = y.quantile(0.75)

            # Model verilerini kaydet
            model_data = {
                'model': model,
                'feature_order': list(X.columns),
                'mean_time': float(mean_time),
                'optimal_time': float(optimal_time),
                'typical_time': float(typical_time),
                'max_efficient_time': float(max_efficient_time),
                'accuracy': float(accuracy)  # Doğruluk yüzdesini kaydet
            }

            # Model tipine göre dosya adı belirle
            model_files = {
                'AG Yassı Tel': 'ag_yassi_tel.joblib',
                'YG Emaye': 'YG_Emaye.joblib',
                'Folyo': 'folyo.joblib'
            }
            
            # Modeli kaydet
            model_file = model_files[self.selected_model_type]
            joblib.dump(model_data, model_file)

            # Sonuçları göster
            self.result_label.setText(
                f'Model başarıyla eğitildi ve kaydedildi!\n'
                f'Model Doğruluk Oranı: %{accuracy:.1f}\n'
                f'MAE: {mae:.2f}\n'
                f'R²: {r2:.2f}\n'
                f'Ortalama Süre: {mean_time:.1f} saat\n'
                f'En İyi Süre: {optimal_time:.1f} saat\n'
                f'Tipik Süre: {typical_time:.1f} saat\n'
                f'Verimli Maksimum Süre: {max_efficient_time:.1f} saat'
            )

            # Kaydetme başarılı mesajı göster
            self.show_styled_message_box(
                QMessageBox.Information,
                'Başarılı',
                f'Model başarıyla kaydedildi!\n'
                f'Dosya: {model_file}\n'
                f'Doğruluk Oranı: %{accuracy:.1f}'
            )

            # Eğer doğruluk oranı düşükse uyarı ver
            if accuracy < 80:
                self.show_styled_message_box(
                    QMessageBox.Warning,
                    'Uyarı',
                    f'Model doğruluk oranı düşük: %{accuracy:.1f}\n'
                    'Daha iyi sonuçlar için veri setinizi kontrol edin.'
                )

        except Exception as e:
            self.show_styled_message_box(
                QMessageBox.Critical,
                'Hata',
                f'Model eğitimi sırasında hata oluştu: {str(e)}'
            )

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 9))
    window = ModelTrainerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 