import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                           QDoubleSpinBox, QMessageBox, QFrame, QComboBox,
                           QTabWidget, QFileDialog, QDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime

class AboutTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Logo container
        logo_container = QFrame()
        logo_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 0px;
            }
        """)
        logo_layout = QVBoxLayout(logo_container)

        # Logo
        try:
            logo_label = QLabel()
            pixmap = QPixmap('logo.jpeg')
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
                logo_label.setAlignment(Qt.AlignCenter)
            else:
                logo_label.setText("Logo could not be loaded")
            logo_layout.addWidget(logo_label)
        except Exception as e:
            print(f"Logo loading error: {str(e)}")

        layout.addWidget(logo_container)

        # Bilgi container
        info_container = QFrame()
        info_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b;
                border-radius: 5px;
                padding: 20px;
            }
            QLabel { 
                color: #ffffff;
                font-size: 14px;
            }
        """)
        info_layout = QVBoxLayout(info_container)

        # Program adı ve versiyon
        self.title = QLabel('Transformer Prediction System')
        self.title.setStyleSheet('font-size: 24px; font-weight: bold; margin-bottom: 10px;')
        self.title.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.title)

        self.version = QLabel('Version 1.0')
        self.version.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.version)

        # Geliştirici bilgileri
        self.developers = QLabel("""
Developers:
- Muhammed İkbal SIRDAŞ
- Zeki Can YAVUZ

© 2025 All rights reserved.

This software has been developed for
prediction and optimization of transformer
production processes.
        """)
        self.developers.setStyleSheet('padding: 20px;')
        self.developers.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.developers)

        layout.addWidget(info_container)

    def update_texts(self, is_english):
        self.title.setText('Transformer Prediction System' if is_english else 'Transformatör Tahmin Sistemi')
        self.version.setText('Version 1.0' if is_english else 'Versiyon 1.0')
        
        developers_text = """
Developers:
- Muhammed İkbal SIRDAŞ
- Zeki Can YAVUZ

© 2025 All rights reserved.

This software has been developed for
prediction and optimization of transformer
production processes.
        """ if is_english else """
Geliştiriciler:
- Muhammed İkbal SIRDAŞ
- Zeki Can YAVUZ

© 2025 Tüm hakları saklıdır.

Bu yazılım, transformatör üretim süreçlerinin
tahminlenmesi ve optimizasyonu için
geliştirilmiştir.
        """
        self.developers.setText(developers_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Transformer Prediction System')
        self.setGeometry(100, 100, 800, 600)
        
        # Dil seçimi için değişken
        self.is_english = True
        
        # Program ikonu ayarla
        try:
            self.setWindowIcon(QIcon('datsanstlogo.ico'))
        except Exception as e:
            print(f"Icon loading error: {str(e)}")
        
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Üst bar - Dil seçimi için
        top_bar = QHBoxLayout()
        
        # Dil değiştirme butonu
        self.language_button = QPushButton('🌐 TR / EN')
        self.language_button.setStyleSheet("""
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
                max-width: 80px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        self.language_button.clicked.connect(self.toggle_language)
        top_bar.addWidget(self.language_button, alignment=Qt.AlignRight)
        layout.addLayout(top_bar)
        
        # Tab widget oluştur
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2b2b2b;
                color: white;
                padding: 8px 20px;
                border: 1px solid #3d3d3d;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0d47a1;
            }
            QTabBar::tab:hover {
                background: #1565c0;
            }
        """)
        
        # Sekmeleri oluştur - parent parametresini ekleyerek
        self.prediction_tab = PredictionTab(parent=self)
        self.training_tab = TrainingTab(parent=self)
        self.about_tab = AboutTab()
        
        # Sekmeleri ekle
        self.tab_widget.addTab(self.prediction_tab, "Prediction" if self.is_english else "Tahmin")
        self.tab_widget.addTab(self.training_tab, "Model Training" if self.is_english else "Model Eğitimi")
        self.tab_widget.addTab(self.about_tab, "About" if self.is_english else "Hakkında")
        
        layout.addWidget(self.tab_widget)
        
        # Dark mode stil
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; }
        """)

    def toggle_language(self):
        self.is_english = not self.is_english
        self.update_texts()
    
    def update_texts(self):
        # Ana pencere başlığı
        self.setWindowTitle('Transformer Prediction System' if self.is_english else 'Transformatör Tahmin Sistemi')
        
        # Tab başlıkları
        self.tab_widget.setTabText(0, "Prediction" if self.is_english else "Tahmin")
        self.tab_widget.setTabText(1, "Model Training" if self.is_english else "Model Eğitimi")
        self.tab_widget.setTabText(2, "About" if self.is_english else "Hakkında")
        
        # Alt sekmelerin metinlerini güncelle
        self.prediction_tab.update_texts(self.is_english)
        self.training_tab.update_texts(self.is_english)
        self.about_tab.update_texts(self.is_english)

class PredictionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
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
        self.model_selection_title = QLabel("Model Selection:")
        self.model_selection_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        model_selection_layout.addWidget(self.model_selection_title)

        # Dropdown menu için ComboBox
        self.model_combo = QComboBox()
        self.models = [
            ("AG Flat Wire", "ag_yassi_tel.joblib"),
            ("YG Emaye", "YG_Emaye.joblib"),
            ("Folyo", "folyo.joblib")
        ]
        
        # ComboBox'a modelleri ekle
        for name, _ in self.models:
            self.model_combo.addItem(name)
        
        self.model_combo.setStyleSheet("""
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
        """)
        
        # Model değişikliğinde çağrılacak fonksiyon
        self.model_combo.currentIndexChanged.connect(self.load_model)
        model_selection_layout.addWidget(self.model_combo)
        
        layout.addWidget(model_selection_container)

        # Input container
        input_container = QFrame()
        input_container.setStyleSheet("""
            QFrame { 
                background-color: #2b2b2b; 
                padding: 15px;
                border-radius: 5px;
            }
        """)
        input_layout = QVBoxLayout(input_container)

        # Input alanları
        self.inputs = {}
        self.parameter_labels = {}  # Etiketleri saklamak için yeni sözlük
        
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
            
            spinbox.setStyleSheet("""
                QSpinBox, QDoubleSpinBox { 
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #3d3d3d;
                    padding: 3px;
                }
            """)
            
            self.inputs[name] = spinbox
            self.parameter_labels[name] = label  # Etiketi sakla
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            input_layout.addLayout(h_layout)

        layout.addWidget(input_container)

        self.predict_button = QPushButton('Estimate Duration')
        self.predict_button.setCursor(Qt.PointingHandCursor)
        self.predict_button.clicked.connect(self.make_prediction)
        self.predict_button.setStyleSheet("""
            QPushButton { 
                background-color: #0d47a1; 
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        layout.addWidget(self.predict_button)

        # Sonuç etiketi
        self.result_label = QLabel('')
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                padding: 15px;
                font-size: 16px;
                color: #4caf50;
                border-radius: 5px;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)


        self.load_model()

    def load_model(self):
        try:
            _, selected_model = self.models[self.model_combo.currentIndex()]
            self.model_data = joblib.load(selected_model)
            self.model = self.model_data['model']
            self.feature_order = self.model_data['feature_order']
            self.result_label.setText('')
        except Exception as e:
            self.model = None
            self.feature_order = None
            self.model_data = None
            print(f"Model loading error: {str(e)}")
            QMessageBox.warning(self, 'Warning', f'Model file could not be loaded: {selected_model}')

    def make_prediction(self):
        try:
            if self.model is None:
                raise Exception("Model not loaded!" if self.parent.is_english else "Model yüklenemedi!")

            input_data = {name: [spinbox.value()] 
                         for name, spinbox in self.inputs.items()}
            
            new_transformer = pd.DataFrame(input_data)
            new_transformer = new_transformer[self.feature_order]
            
            predicted_time = self.model.predict(new_transformer)[0]
            
            mean_time = self.model_data.get('mean_time', 0)
            optimal_time = self.model_data.get('optimal_time', 0)
            typical_time = self.model_data.get('typical_time', 0)
            max_efficient_time = self.model_data.get('max_efficient_time', 0)
            accuracy = self.model_data.get('accuracy', 0)

            if self.parent.is_english:
                result_text = (
                    f'Estimated Production Time: {predicted_time:.1f} Hours ({predicted_time/8:.1f} Days)\n'
                    f'Model Accuracy: %{accuracy:.1f}\n'
                )
            else:
                result_text = (
                    f'Tahmini Üretim Süresi: {predicted_time:.1f} Saat ({predicted_time/8:.1f} Gün)\n'
                    f'Model Doğruluk Oranı: %{accuracy:.1f}\n'
                )
            
            self.result_label.setText(result_text)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                'Error' if self.parent.is_english else 'Hata',
                f'{"An error occurred" if self.parent.is_english else "Bir hata oluştu"}: {str(e)}'
            )

    def update_texts(self, is_english):
        # Model seçim başlığı
        self.model_selection_title.setText("Model Selection:" if is_english else "Model Seçimi:")
        
        # Model isimleri
        current_index = self.model_combo.currentIndex()
        self.model_combo.clear()
        models = [
            ("AG Flat Wire", "AG Yassı Tel"),
            ("HV Enamel", "YG Emaye"),
            ("Foil", "Folyo")
        ]
        for eng, tr in models:
            self.model_combo.addItem(eng if is_english else tr)
        self.model_combo.setCurrentIndex(current_index)
        
        # Tahmin butonu
        self.predict_button.setText('Estimate Duration' if is_english else 'Süre Tahmin Et')
        
        # Parametre isimleri
        parameter_translations = {
            "Number of Transformers": "Transformatör Sayısı",
            "Overall Coil Number": "Toplam Bobin Sayısı",
            "Power": "Güç",
            "Number Of Turn": "Sarım Sayısı",
            "Cooling Duct": "Soğutma Kanalı",
            "Winding Height": "Sargı Yüksekliği",
            "Number Of Layers": "Kat Sayısı",
            "Wire Width": "Tel Genişliği",
            "Wire Thickness": "Tel Kalınlığı",
            "Wire Type": "Tel Tipi",
            "AG": "AG"
        }
        
        # Etiketleri güncelle
        for name, label in self.parameter_labels.items():
            label.setText(f"{parameter_translations[name] if not is_english else name}:")

class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Başlık
        self.title = QLabel('Model Training Interface')
        self.title.setStyleSheet("""
            font-size: 24px;
            color: #ffffff;
            margin-bottom: 20px;
            padding: 10px;
        """)
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

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
        self.file_button = QPushButton('Excel File Select')
        self.file_button.setCursor(Qt.PointingHandCursor)
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setStyleSheet("""
            QPushButton { 
                background-color: #0d47a1; 
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        file_layout.addWidget(self.file_button, alignment=Qt.AlignCenter)

        # Seçili dosya etiketi
        self.file_label = QLabel('No file selected')
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
        self.model_type_label = QLabel('Model Type:')
        self.model_type_buttons = {}
        
        for model_type in ['AG Flat Wire', 'YG Emaye', 'Folyo']:
            btn = QPushButton(model_type)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=model_type: self.select_model_type(m))
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #0d47a1; 
                    color: white;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QPushButton:checked {
                    background-color: #1565c0;
                }
            """)
            train_layout.addWidget(btn)
            self.model_type_buttons[model_type] = btn

        train_layout.addWidget(self.model_type_label)

        # Eğitim butonu
        self.train_button = QPushButton('Model Train')
        self.train_button.setCursor(Qt.PointingHandCursor)
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        self.train_button.setStyleSheet("""
            QPushButton { 
                background-color: #0d47a1; 
                color: white;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
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
        self.result_label.setStyleSheet("""
            QLabel {
                color: #4caf50;
                font-size: 14px;
            }
        """)
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
            "Excel File Select",
            "",
            "Excel Files (*.xlsx *.xls)"
        )
        if file_name:
            self.selected_file = file_name
            self.file_label.setText(f'Selected file: {file_name.split("/")[-1]}')
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
            accuracy = (1 - mae/y_test.mean()) * 100

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
                'accuracy': float(accuracy)
            }

            # Model tipine göre dosya adı belirle
            model_files = {
                'AG Flat Wire': 'ag_yassi_tel.joblib',
                'YG Emaye': 'YG_Emaye.joblib',
                'Folyo': 'folyo.joblib'
            }
            
            # Modeli kaydet
            model_file = model_files[self.selected_model_type]
            joblib.dump(model_data, model_file)

            # Sonuçları göster
            if self.parent.is_english:
                self.result_label.setText(
                    f'Model successfully trained and saved!\n'
                    f'Model Accuracy: %{accuracy:.1f}\n'
                    f'MAE: {mae:.2f}\n'
                    f'R²: {r2:.2f}\n'
                )

                # Kaydetme başarılı mesajı göster
                self.show_styled_message_box(
                    QMessageBox.Information,
                    'Success',
                    f'Model successfully saved!\n'
                    f'File: {model_file}\n'
                    f'Accuracy: %{accuracy:.1f}'
                )

                # Eğer doğruluk oranı düşükse uyarı ver
                if accuracy < 80:
                    self.show_styled_message_box(
                        QMessageBox.Warning,
                        'Warning',
                        f'Model accuracy is low: %{accuracy:.1f}\n'
                        'Please check your dataset.'
                    )
            else:
                self.result_label.setText(
                    f'Model başarıyla eğitildi ve kaydedildi!\n'
                    f'Model Doğruluk Oranı: %{accuracy:.1f}\n'
                    f'MAE: {mae:.2f}\n'
                    f'R²: {r2:.2f}\n'
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
                        'Veri setinizi kontrol edin.'
                    )

        except Exception as e:
            self.show_styled_message_box(
                QMessageBox.Critical,
                'Error' if self.parent.is_english else 'Hata',
                f'{"An error occurred during model training" if self.parent.is_english else "Model eğitimi sırasında hata oluştu"}: {str(e)}'
            )

    def update_texts(self, is_english):
        self.title.setText('Model Training Interface' if is_english else 'Model Eğitim Arayüzü')
        self.file_button.setText('Select Excel File' if is_english else 'Excel Dosyası Seç')
        self.file_label.setText('No file selected' if is_english else 'Dosya seçilmedi')
        self.model_type_label.setText('Model Type:' if is_english else 'Model Tipi:')
        self.train_button.setText('Train Model' if is_english else 'Modeli Eğit')

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 