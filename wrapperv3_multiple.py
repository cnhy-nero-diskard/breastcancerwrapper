import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QGraphicsDropShadowEffect


model = tf.keras.models.load_model('version1.h5')


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Images Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa;
                font-size: 18px;
                color: #aaa;
            }
        ''')
        self.setAcceptDrops(True)  # Enable drag-and-drop for this widget
        self.parent_window = parent  # Store a reference to the parent window

    def dragEnterEvent(self, event):
        # Check if the dropped data contains URLs (file paths)
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        # Allow dragging over the widget
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        # Handle the dropped files
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            valid_files = [f for f in files if os.path.splitext(f)[-1].lower() in ['.png', '.jpg', '.jpeg']]
            if valid_files:
                self.parent_window.dropped_files.extend(valid_files)  # Access dropped_files through parent_window
                event.acceptProposedAction()
                self.parent_window.update_drop_label()  # Update the drop label
            else:
                QMessageBox.warning(self, "Invalid Files", "Only .png, .jpg, and .jpeg files are supported.")
        else:
            event.ignore()
class PredictionWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet('background-color: #f0e9e9')

        self.dropped_files = []  # List to store dropped file paths

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.drop_label = ImageLabel(self)  # Pass the PredictionWindow instance as the parent
        self.layout.addWidget(self.drop_label)

        self.button_layout = QHBoxLayout()
        
        # Predict Images Button
        self.predict_button = QPushButton("Predict Images")
        self.predict_button.clicked.connect(self.predict_images)
        self.predict_button.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;  /* Green color */
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;  /* Darker green on hover */
            }
            QPushButton:pressed {
                background-color: #3e8e41;  /* Even darker green on press */
            }
        ''')
        self.button_layout.addWidget(self.predict_button)

        # Clear All Images Button
        self.clear_button = QPushButton("Clear All Images")
        self.clear_button.clicked.connect(self.clear_all_images)
        self.clear_button.setStyleSheet('''
            QPushButton {
                background-color: #f44336;  /* Red color */
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #e53935;  /* Darker red on hover */
            }
            QPushButton:pressed {
                background-color: #d32f2f;  /* Even darker red on press */
            }
        ''')
        self.button_layout.addWidget(self.clear_button)

        self.layout.addLayout(self.button_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)

        self.scroll_layout = QVBoxLayout(self.scroll_widget)

    def update_drop_label(self):
        # Update the drop label to show the number of dropped files
        if self.dropped_files:
            self.drop_label.setText(f"\n\n {len(self.dropped_files)} images ready for prediction \n\n")
        else:
            self.drop_label.setText("\n\n Drop Images Here \n\n")

    def predict_images(self):
        if not self.dropped_files:
            QMessageBox.warning(self, "No Images", "Please drop some images first.")
            return

        # Clear the existing results before displaying new ones
        self.clear_scroll_layout()

        for file_path in self.dropped_files:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, "Invalid Image", f"Could not read image: {file_path}")
                continue

            # Preprocess the image
            input_image = self.preprocess_image(image)
            # Make a prediction
            prediction = model.predict(input_image)
            # Get the predicted class (0 or 1)
            predicted_class = "CANCEROUS" if np.argmax(prediction, axis=1)[0] == 1 else "NONCANCEROUS"
            # Get the probabilities for both classes
            class_0_prob = prediction[0][0]
            class_1_prob = prediction[0][1]

            # Display the image with prediction
            pixmap = self.convert_cv_to_qpixmap(image)
            label = QLabel()
            label.setPixmap(pixmap)
            # label.setStyleSheet("border: 2px solid green;")  # Add green border to the image

            # Get the filename from the file path
            filename = os.path.basename(file_path)
            prediction_text = f"Filename: {filename}\nPredicted Class: {predicted_class}\nProbability for Class 0: {class_0_prob:.6f}\nProbability for Class 1: {class_1_prob:.6f}"
            prediction_label = QLabel(prediction_text)

            image_layout = QHBoxLayout()
            image_layout.addWidget(label)
            image_layout.addWidget(prediction_label)

            image_widget = QWidget()
            image_widget.setLayout(image_layout)
            image_widget.setStyleSheet("background-color: #fff3f2; padding: 10px; border-radius: 5px;")  # Set background color and padding

            # Add drop shadow effect to the result row
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setColor(QColor(0, 0, 0, 100))
            shadow.setOffset(2, 2)
            image_widget.setGraphicsEffect(shadow)

            self.scroll_layout.addWidget(image_widget)

        self.scroll_layout.addStretch()

    def clear_all_images(self):
        # Clear the dropped files list and reset the UI
        self.dropped_files.clear()
        self.update_drop_label()
        self.clear_scroll_layout()

    def clear_scroll_layout(self):
        # Remove all widgets from the scroll layout
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def preprocess_image(self, image):
        # Resize the image to 50x50 pixels
        resized_image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_LINEAR)
        # Convert the image to a numpy array and expand dimensions to match the model input shape
        input_image = np.expand_dims(resized_image, axis=0)
        return input_image

    def convert_cv_to_qpixmap(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionWindow()
    window.show()
    sys.exit(app.exec_())