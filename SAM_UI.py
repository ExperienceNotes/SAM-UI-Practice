import os
import sys
import json
import time
import requests
import logging

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from types import TracebackType
from typing import Optional
from SAM_Function import MyGraphicsScene
from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2


class Ui_Dialog(object):

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        file_handler: logging.FileHandler = logging.FileHandler('app.log')
        file_handler.setLevel(logging.INFO)
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        sys.excepthook = self.handle_exception
        self.setting_path = "Setting.json"
        self.Setting_Para = self.load_setting()
        self.device: str = "cuda"
        if self.Setting_Para["version"] == 1:
            self.model_type: str = "vit_h"
            self.model_url = self.Setting_Para["Model_Url"]
            self.check_and_download_model()
            self.seg_model = self.initialize_predictor_v1()
        elif self.Setting_Para["version"] == 2:
            self.model_cfg = "./sam2_hiera_b+.yaml"
            self.model_url = self.Setting_Para["Model_Url"]
            self.check_and_download_model()
            self.seg_model = self.initialize_predictor_v2()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1311, 811)
        self.horizontalWidget = QtWidgets.QWidget(Dialog)
        self.horizontalWidget.setGeometry(QtCore.QRect(40, 30, 1241, 401))
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.horizontalWidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.graphicsView = QtWidgets.QGraphicsView(self.horizontalWidget)
        self.graphicsView.setObjectName("graphicsView")
        self.horizontalLayout.addWidget(self.graphicsView)
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.horizontalWidget)
        self.graphicsView_3.setObjectName("graphicsView_3")

        self.graphicScene = MyGraphicsScene(Dialog)
        self.graphicsView_2.setScene(self.graphicScene)

        self.graphicScene_2 = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.graphicScene_2)

        self.graphicScene_3 = QtWidgets.QGraphicsScene()
        self.graphicsView_3.setScene(self.graphicScene_3)

        self.horizontalLayout.addWidget(self.graphicsView_3)
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(48, 490, 401, 201))
        self.listWidget.setObjectName("listWidget")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(50, 460, 61, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 10, 61, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(460, 10, 61, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(870, 10, 61, 16))
        self.label_4.setObjectName("label_4")
        self.horizontalGroupBox = QtWidgets.QGroupBox(Dialog)
        self.horizontalGroupBox.setGeometry(QtCore.QRect(490, 470, 361, 231))
        self.horizontalGroupBox.setObjectName("horizontalGroupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalGroupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalGroupBox = QtWidgets.QGroupBox(self.horizontalGroupBox)
        self.verticalGroupBox.setObjectName("verticalGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalGroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.verticalGroupBox)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalGroupBox)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalGroupBox)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalGroupBox)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.horizontalLayout_2.addWidget(self.verticalGroupBox)
        self.verticalGroupBox1 = QtWidgets.QGroupBox(self.horizontalGroupBox)
        self.verticalGroupBox1.setObjectName("verticalGroupBox1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalGroupBox1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.comboBox = QtWidgets.QComboBox(self.verticalGroupBox1)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(self.Setting_Para["classes"])

        self.verticalLayout_2.addWidget(self.comboBox)
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalGroupBox1)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_2.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalGroupBox1)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalGroupBox1)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        self.horizontalLayout_2.addWidget(self.verticalGroupBox1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Image path"))
        self.label_2.setText(_translate("Dialog", "Org Image"))
        self.label_3.setText(_translate("Dialog", "Mask Image"))
        self.label_4.setText(_translate("Dialog", "Seg Image"))
        self.pushButton.setText(_translate("Dialog", "Load Folder"))
        self.pushButton_2.setText(_translate("Dialog", "Predict Mask"))
        self.pushButton_3.setText(_translate("Dialog", "Save Mask"))
        self.pushButton_4.setText(_translate("Dialog", "Clear Points"))
        self.pushButton_5.setText(_translate("Dialog", "Yolo Format"))
        self.pushButton_6.setText(_translate("Dialog", "Xml Format"))
        self.pushButton_7.setText(_translate("Dialog", "Other Format"))

    def handle_exception(self, exc_type: type[BaseException], exc_value: BaseException,
                         exc_traceback: Optional[TracebackType]):
        """
        Handle uncaught exceptions by logging them.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def check_and_download_model(self):
        checkpoint_dir = os.path.dirname(self.Setting_Para["sam_checkpoint"])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if not os.path.exists(self.Setting_Para["sam_checkpoint"]):
            reply = QtWidgets.QMessageBox.question(self, 'Model Download',
                                                   'Model file not found. Do you want to download it?',
                                                   QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                   QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.logger.info("Model file not found. Downloading...")
                try:
                    response = requests.get(self.model_url, stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    progress_dialog = QtWidgets.QProgressDialog("Downloading model...", "Cancel",
                                                                0, total_size // 1024, self)
                    progress_dialog.setWindowModality(Qt.WindowModal)
                    progress_dialog.setMinimumDuration(0)
                    progress_dialog.show()

                    downloaded_size = 0
                    with open(self.Setting_Para["sam_checkpoint"], 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if progress_dialog.wasCanceled():
                                self.logger.info("Model download canceled by user.")
                                os.remove(self.Setting_Para["sam_checkpoint"])
                                return
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            progress_dialog.setValue(downloaded_size // 1024)

                    self.logger.info("Model downloaded successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to download the model: {e}")
                    QtWidgets.QMessageBox.critical(self, 'Download Failed', f'Failed to download the model: {e}')
                    raise
            else:
                self.logger.info("Model download canceled by user.")
                sys.exit()

    def initialize_predictor_v1(self):
        self.logger.info("Loading Sam1 Model...")
        t_s = time.perf_counter()
        try:
            seg_model = sam_model_registry[self.model_type](checkpoint=self.Setting_Para["sam_checkpoint"])
            seg_model.to(device=self.device)
            t_e = time.perf_counter()
            self.logger.info(f"Model initialized successfully, Loading time: {(t_e - t_s):.4f} sec")
            return seg_model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")

    def initialize_predictor_v2(self):
        self.logger.info("Loading Sam2 Model...")
        t_s = time.perf_counter()
        try:
            check_point_path = os.path.join(os.getcwd(), self.Setting_Para["sam_checkpoint"])
            seg_model = build_sam2(self.model_cfg, check_point_path, self.device)
            seg_model.to(device=self.device)
            t_e = time.perf_counter()
            self.logger.info(f"Model initialized successfully, Loading time: {(t_e - t_s):.4f} sec")
            return seg_model
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")

    def load_setting(self):
        with open(self.setting_path, encoding='utf-8') as f:
            data = json.load(f)
        return data


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

