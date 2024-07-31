import os.path
import sys
import time

from PyQt5 import QtGui, QtWidgets
from segment_anything import SamPredictor
from typing import Optional
from SAM_Function import load_folder, resize_image, MaskSelectionDialog, apply_mask_to_image
import SAM_UI
import cv2
import numpy as np


class Test_demo(QtWidgets.QWidget, SAM_UI.Ui_Dialog):
    def __init__(self):
        super(Test_demo, self).__init__()
        self.setupUi(self)
        self.image: Optional[np.ndarray] = None

        self.image_list = []
        self.image_path: str = ""
        self.input_points: list[list[int]] = []
        self.input_labels: list[int] = []

        self.pushButton.clicked.connect(self.handle_load_folder)
        self.listWidget.clicked.connect(self.select_Image)
        self.pushButton_2.clicked.connect(self.predict_mark)
        self.pushButton_3.clicked.connect(self.save_mask)
        self.pushButton_4.clicked.connect(self.clear_point)

    def handle_load_folder(self):
        self.image_list = load_folder()
        self.listWidget.addItems(self.image_list)

    def select_Image(self):
        self.image_path = self.listWidget.currentItem().text()
        if self.image_path:
            self.graphicScene.clear()
            print(self.image_path)
            self.logger.info("Loading Graph and Encoder Image...")
            time_s = time.perf_counter()
            '''init point and graph'''
            self.graphicScene.clear()
            self.graphicScene.clear_points()
            self.input_points = []
            self.input_labels = []

            self.image = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), -1)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            # print(f"self.graphicsView_2.width(): {self.graphicsView_2.width()}")
            # print(f"self.graphicsView_2.height(): {self.graphicsView_2.height()}")
            resized_image: np.ndarray = resize_image(self.image, self.graphicsView_2.width(),
                                                     self.graphicsView_2.height())
            height, width, _ = resized_image.shape
            frame = QtGui.QImage(resized_image, width, height, width * 3, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(frame)
            item = QtWidgets.QGraphicsPixmapItem(pix)
            self.graphicScene.addItem(item)
            self.graphicsView_2.setScene(self.graphicScene)
            self.graphicsView_2.show()
            self.predictor: SamPredictor = SamPredictor(self.seg_model)
            self.predictor.set_image(self.image)
            time_e = time.perf_counter()
            self.logger.info(f"Loading Graph and Encoder Image successfully, Loading time: {(time_e - time_s):.4f} sec")

    def predict_mark(self):
        if self.image is not None and self.input_points:
            input_points_np: np.ndarray = np.array(self.input_points)
            input_labels_np: np.ndarray = np.array(self.input_labels)
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points_np,
                point_labels=input_labels_np,
                multimask_output=True,
            )
            if self.comboBox.currentText():
                print("")
                label_key: int = self.comboBox.currentIndex()
            else:
                reply = QtWidgets.QMessageBox.question(self, "Enter label name", "You did not enter a label name",
                                                       QtWidgets.QMessageBox.Yes)
                if reply == QtWidgets.QMessageBox.Yes:
                    return

            self.show_mask_selection_dialog(masks, scores, label_key)
            # self.logger.info(f"Predicted mask for image: {self.image_path}")

    def on_image_click(self, x: int, y: int, label: int):
        if self.image is not None:
            window_w: int = self.graphicsView_2.width()
            window_h: int = self.graphicsView_2.height()

            image_height, image_width, _ = self.image.shape
            scale_x: float = image_width / window_w
            scale_y: float = image_height / window_h

            point: list[int] = [int(x * scale_x), int(y * scale_y)]
            self.input_points.append(point)
            self.input_labels.append(label)

    def show_mask_selection_dialog(self, mask: np.ndarray, scores: np.ndarray, label_key: int):
        dialog = MaskSelectionDialog(self.image, mask, scores, label_key)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_mask_index: int = dialog.get_selected_mask_index()
            self.show_select_mask(mask[selected_mask_index], label_key)

    def show_select_mask(self, mask: np.ndarray, label_key: int):
        self.graphicScene_2.clear()
        self.graphicScene_3.clear()
        self.mask_image, self.label_image = apply_mask_to_image(self.image, mask, label_key)
        mask_resize_image: np.ndarray = resize_image(self.mask_image, self.graphicsView.width(),
                                                     self.graphicsView.height())
        label_resize_image: np.ndarray = resize_image(self.label_image, self.graphicsView_3.width(),
                                                      self.graphicsView_3.height())
        # mask image setting and show
        height, width, _ = mask_resize_image.shape
        frame = QtGui.QImage(mask_resize_image.data, width, height, width*3, QtGui.QImage.Format_RGB888)
        mask_image_item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(frame))
        self.graphicScene_2.addItem(mask_image_item)
        self.graphicsView.setScene(self.graphicScene_2)
        # label image setting and show
        height, width, _ = label_resize_image.shape
        frame = QtGui.QImage(label_resize_image.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
        label_image_item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(frame))
        self.graphicScene_3.addItem(label_image_item)
        self.graphicsView_3.setScene(self.graphicScene_3)

        self.graphicsView.show()
        self.graphicsView_3.show()

    def save_mask(self):
        mask_folder_path: str = os.path.join(os.getcwd(), "masks")
        if not os.path.exists(mask_folder_path):
            os.makedirs(mask_folder_path)
        org_name = os.path.splitext(os.path.basename(self.image_path))
        mask_save_path = os.path.join(mask_folder_path, org_name[0]) + "_mask.jpg"
        label_save_path = os.path.join(mask_folder_path, org_name[0]) + "_label.jpg"
        status_mask: bool = cv2.imwrite(mask_save_path, cv2.cvtColor(self.mask_image, cv2.COLOR_RGB2BGR))
        status_label: bool = cv2.imwrite(label_save_path, cv2.cvtColor(self.label_image, cv2.COLOR_RGB2BGR))

        if status_label or status_mask:
            self.logger.info(f"Label saved: {os.path.join(mask_folder_path, mask_save_path)}")
            self.logger.info(f"Mask saved: {os.path.join(mask_folder_path, mask_save_path)}")
        else:
            # self.logger.info(f"Failed to save mask: {os.path.join(mask_folder_path, mask_save_path)}")
            self.logger.info(f"Failed status: status_mask: {status_mask}, status_label: {status_label}")

    def clear_point(self):
        """
        Clear the selected points.
        """
        self.input_points = []
        self.input_labels = []
        self.graphicScene.clicked_position = []
        self.graphicScene.clear_points()

    def closeEvent(self, event) -> None:
        """
        Handle the close event by logging a message.
        """
        self.logger.info("Close app " + "=" * 40)
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = Test_demo()
    win.show()
    sys.exit(app.exec_())
