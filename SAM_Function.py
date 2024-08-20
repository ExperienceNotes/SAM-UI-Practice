import os

import cv2
import numpy as np
import hashlib
from typing import Tuple
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt


def load_folder():
    folder_path = QtWidgets.QFileDialog.getExistingDirectory()
    if folder_path:
        file_list = os.listdir(folder_path)
        image_list = [os.path.join(folder_path, f) for f in file_list
                      if f.lower().endswith(('png', 'jpg', 'jpeg', '.bmp'))]
    return image_list


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, label_key: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the mask to the image with the specified color.

    :param image: The original image.
    :param mask: The mask to be applied.
    :param label_key: The label key to determine the mask color.
    :return: The image with the mask applied.
    """
    color = get_color_from_label_key(label_key)
    masked_image: np.ndarray = image.copy()
    label_image: np.ndarray = np.zeros_like(image)
    for c in range(3):
        masked_image[:, :, c] = np.where(mask, masked_image[:, :, c] * 0.3 + color[c] * 0.7, masked_image[:, :, c])
        label_image[:, :, c] = np.where(mask, color[c], label_image[:, :, c])
    return masked_image, label_image


def get_color_from_label_key(label_key: int) -> list[int]:
    """
    Generate a color based on the label key.

    :param label_key: The label key.
    :return: A list of RGB values.
    """
    hash_object = hashlib.md5(str(label_key).encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return [r, g, b]


def resize_image(image: np.ndarray, window_w: int, window_h: int) -> np.ndarray:
    image_height, image_width, _ = image.shape
    image_aspect_ratio: float = image_width / image_height
    label_aspect_ratio: float = window_w / window_h

    if image_aspect_ratio > label_aspect_ratio:
        new_width: int = window_w
        new_height: int = int(new_width / image_aspect_ratio)
    else:
        new_height: int = window_h
        new_width: int = int(new_height * image_aspect_ratio)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class MyGraphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(MyGraphicsScene, self).__init__(parent)
        self.clicked_position: list[tuple[int, int, int]] = []

    def mousePressEvent(self, event):
        # 獲取點擊的位置
        # pos = event.scenePos()
        # 輸出點擊的座標
        # print(f"Clicked at: {pos.x()}, {pos.y()}")
        # super(MyGraphicsScene, self).mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            print(f"Clicked at: {pos.x()}, {pos.y()}")
            self.clicked_position.append((int(pos.x()), int(pos.y()), 1))
            self.parent().on_image_click(int(pos.x()), int(pos.y()), 1)
            self.update()
        elif event.button() == Qt.RightButton:
            pos = event.scenePos()
            print(f"Clicked at: {pos.x()}, {pos.y()}")
            self.clicked_position.append((int(pos.x()), int(pos.y()), 1))
            self.parent().on_image_click(int(pos.x()), int(pos.y()), 1)
            self.update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        for x, y, label in self.clicked_position:
            if label == 1:
                pen: QtGui.QPen = QtGui.QPen(Qt.red)
                brush: QtGui.QBrush = QtGui.QBrush(Qt.red)
            elif label == 0:
                pen: QtGui.QPen = QtGui.QPen(Qt.green)
                brush: QtGui.QBrush = QtGui.QBrush(Qt.green)

            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.drawText(x, y, f'Label: {label}, (x, y): ({x}, {y})')
        painter.end()

    def clear_points(self):
        self.clicked_position = []
        self.update()


class MaskSelectionDialog(QtWidgets.QDialog):
    def __init__(self, image: np.ndarray, masks: np.ndarray, scores: list[float], label_key: int):
        super(MaskSelectionDialog, self).__init__()
        self.setWindowTitle("Select Mask")

        self.image: np.ndarray = image
        self.masks: np.ndarray = masks
        self.scores: list[float] = scores
        self.label_key: int = label_key
        self.current_index: int = 0

        self.initUI()
        self.show_mask(self.current_index)

    def initUI(self):
        """Initialize the user interface components."""
        self.layout = QtWidgets.QVBoxLayout()

        self.image_label: QtWidgets.QLabel = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.score_label: QtWidgets.QLabel = QtWidgets.QLabel()
        self.score_label.setAlignment(Qt.AlignCenter)
        font: QtGui.QFont = QtGui.QFont()
        font.setPointSize(16)
        self.score_label.setFont(font)
        self.layout.addWidget(self.score_label)

        self.num_masks_label: QtWidgets.QLabel = QtWidgets.QLabel()
        self.num_masks_label.setAlignment(Qt.AlignCenter)
        font: QtGui.QFont = QtGui.QFont()
        font.setPointSize(16)
        self.num_masks_label.setFont(font)
        self.layout.addWidget(self.num_masks_label)

        self.update_info()

        self.button_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout()

        self.prev_button: QtWidgets.QPushButton = QtWidgets.QPushButton('Previous')
        self.prev_button.clicked.connect(self.show_previous_mask)
        self.button_layout.addWidget(self.prev_button)

        self.next_button: QtWidgets.QPushButton = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(self.show_next_mask)
        self.button_layout.addWidget(self.next_button)

        self.select_button: QtWidgets.QPushButton = QtWidgets.QPushButton('Select')
        self.select_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.select_button)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def update_info(self):
        """Update the score and mask count information labels."""
        self.score_label.setText(f"Score: {self.scores[self.current_index]:.3f}")
        self.num_masks_label.setText(f"Masks: {self.current_index + 1} / {len(self.masks)}")

    def show_mask(self, index: int):
        """
        Display the mask at the specified index.

        :param index: Index of the mask to be displayed.
        """
        mask_image, _ = apply_mask_to_image(self.image, self.masks[index], self.label_key)
        # self.image_label.width(), self.image_label.height()
        mask_image = resize_image(mask_image, self.image.shape[0], self.image.shape[1])
        height, width, _ = mask_image.shape
        bytes_per_line: int = 3 * width
        q_image: QtGui.QImage = QtGui.QImage(mask_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_image))
        self.image_label.adjustSize()

        self.update_info()

    def show_previous_mask(self):
        """Show the previous mask in the list."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_mask(self.current_index)

    def show_next_mask(self):
        """Show the next mask in the list."""
        if self.current_index < len(self.masks) - 1:
            self.current_index += 1
            self.show_mask(self.current_index)

    def get_selected_mask_index(self) -> int:
        """
        Get the index of the currently selected mask.

        :return: The index of the selected mask.
        """
        return self.current_index


if __name__ == "__main__":
    print("")
