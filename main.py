import os
import subprocess
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from mainwindow_ui import Ui_MainWindow


class Communicate(QObject):
    output_updated = pyqtSignal(str)


class TrainingThread(QThread):
    def __init__(
        self, model_name, epochs, batch_size, output_directory, yaml_file, communicate
    ):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_directory = output_directory
        self.yaml_file = yaml_file
        self.communicate = communicate

    def run(self):
        current_directory = os.path.abspath(os.getcwd())

        yolov5_path = os.path.join(current_directory, "yolov5")

        if not os.path.exists(yolov5_path):
            print("Error: yolov5 directory does not exist.")
            return

        os.chdir(yolov5_path)
        os.makedirs(self.output_directory, exist_ok=True)
        command = f"python train.py --img 416 --batch {self.batch_size} --epochs {self.epochs} --data {self.yaml_file} --cfg models/yolov5l.yaml --name {self.model_name} --cache --project {self.output_directory}"
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        while process.poll() is None:
            output = process.stderr.readline()
            self.communicate.output_updated.emit(output)

        # Capture remaining output
        remaining_output = process.stderr.read()
        self.communicate.output_updated.emit(remaining_output)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.yaml_file_button.clicked.connect(self.select_yaml_file)
        self.output_directory_button.clicked.connect(self.select_output_directory)
        self.start_training_button.clicked.connect(self.start_training)

    def select_yaml_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("YAML files (*.yaml)")
        file_dialog.selectFile("")
        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            self.yaml_file_entry.setText(file_name)

    def select_output_directory(self):
        directory_dialog = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if directory_dialog:
            self.output_directory_entry.setText(directory_dialog)

    def start_training(self):
        model_name = self.model_name_entry.text()
        epochs = self.epochs_entry.text()
        batch_size = self.batch_size_entry.text()
        yaml_file = self.yaml_file_entry.text()
        output_directory = self.output_directory_entry.text()

        print("**************")
        print("**  Training Parameters  **\n\n")
        print("Model Name:", model_name)
        print("Epochs:", epochs)
        print("Batch Size:", batch_size)
        print("YAML File:", yaml_file)
        print("Output Directory:", output_directory)
        print("**************")

        if not all([model_name, epochs, batch_size, yaml_file, output_directory]):
            QMessageBox.critical(self, "Error", "Please set all required parameters.")
            return

        self.training_thread = TrainingThread(
            model_name, epochs, batch_size, output_directory, yaml_file, Communicate()
        )
        self.training_thread.communicate.output_updated.connect(
            self.output_text_edit.append
        )
        self.training_thread.start()


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()