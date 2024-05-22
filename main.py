import os
import subprocess
import sys
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
    error_message = pyqtSignal(str)


class TrainingThread(QThread):
    def __init__(
        self,
        model_name,
        epochs,
        batch_size,
        output_directory,
        yaml_file,
        img_size,
        model_size,
        communicate,
    ):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_directory = output_directory
        self.yaml_file = yaml_file
        self.img_size = img_size
        self.model_size = model_size
        self.communicate = communicate

    def run(self):
        current_directory = os.path.abspath(os.getcwd())

        yolov5_path = os.path.join(current_directory, "yolov5")

        if not os.path.exists(yolov5_path):
            print("Error: yolov5 directory does not exist.")
            return

        os.chdir(yolov5_path)
        os.makedirs(self.output_directory, exist_ok=True)
        command = (
            f"python3 train.py --img {self.img_size} --batch {self.batch_size} --epochs {self.epochs} "
            f"--data {self.yaml_file} --cfg models/{self.model_size}.yaml --name {self.model_name} "
            f"--cache --project {self.output_directory}"
        )
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=os.environ,
        )

        while process.poll() is None:
            output = process.stderr.readline()
            self.communicate.output_updated.emit(output)

        # Capture remaining output
        remaining_output = process.stderr.read()
        self.communicate.output_updated.emit(remaining_output)


class UpdateYoloV5Thread(QThread):
    def __init__(self, communicate):
        super().__init__()
        self.communicate = communicate

    def run(self):
        if not self.check_git_installed():
            if not self.install_git():
                self.communicate.error_message.emit("Failed to install git.")
                return

        current_directory = os.path.abspath(os.getcwd())
        yolov5_path = os.path.join(current_directory, "yolov5")

        # Check if the yolov5 directory exists
        if not os.path.exists(yolov5_path):
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
                    check=True,
                )
            except Exception as e:
                self.communicate.error_message.emit(f"Error updating yolov5: {e}")
                return
        else:
            try:
                os.chdir(yolov5_path)
                subprocess.run(["git", "pull"], check=True)
            except Exception as e:
                self.communicate.error_message.emit(
                    f"Error updating Yolov5 repository: {e}"
                )
                return

    def check_git_installed(self):
        try:
            subprocess.run(
                ["git", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except Exception:
            return False

    def install_git(self):
        try:
            if sys.platform.startswith("linux"):
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "git"], check=True)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["brew", "install", "git"], check=True)
            elif sys.platform.startswith("win32"):
                subprocess.run(["winget", "install", "--id", "Git.Git"], check=True)
            else:
                self.communicate.error_message.emit("Unsupported operating system.")
                return False
            return True
        except Exception as e:
            self.communicate.error_message.emit(f"Error installing git: {e}")
            return False


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.yaml_file_button.clicked.connect(self.select_yaml_file)
        self.output_directory_button.clicked.connect(self.select_output_directory)
        self.start_training_button.clicked.connect(self.start_training)

        self.communicate = Communicate()
        self.communicate.error_message.connect(self.show_error_message)

        self.update_yolov5_directory()

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
        img_size = self.img_size_combobox.currentText()
        model_size = self.comboBox.currentText()

        print("**************")
        print("**  Training Parameters  **\n\n")
        print("Model Name:", model_name)
        print("Epochs:", epochs)
        print("Batch Size:", batch_size)
        print("YAML File:", yaml_file)
        print("Output Directory:", output_directory)
        print("Image Size:", img_size)
        print("Model Size:", model_size)
        print("**************")

        if not all(
            [
                model_name,
                epochs,
                batch_size,
                yaml_file,
                output_directory,
                img_size,
                model_size,
            ]
        ):
            QMessageBox.critical(self, "Error", "Please set all required parameters.")
            return

        self.training_thread = TrainingThread(
            model_name,
            epochs,
            batch_size,
            output_directory,
            yaml_file,
            img_size,
            model_size,
            self.communicate,
        )
        self.training_thread.communicate.output_updated.connect(
            self.output_text_edit.append
        )
        self.training_thread.start()

    def update_yolov5_directory(self):
        self.update_thread = UpdateYoloV5Thread(self.communicate)
        self.update_thread.start()

    def show_error_message(self, message):
        self.output_text_edit.append(f"ERROR: {message}")


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
