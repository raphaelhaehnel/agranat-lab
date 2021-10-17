################################################################
# FILE : gui.py
# WRITER : Raphael Haehnel (Created by: PyQt5 UI code generator 5.15.2)
# DESCRIPTION: Generate the gui of the application for the
#              striation analyzer
################################################################

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QApplication, QDialog
from PyQt5.QtCore import QThreadPool, QRunnable, pyqtSlot, pyqtSignal, QObject
import analyzer
import builder
import pickle
import time
import angle_finder
import traceback, sys

''' Number of digits to display for the period '''
PRECISION = 6

''' The microscope scale from pixel to um '''
SCALE = 1.0 / 0.0678

''' The correction angle for the image '''
ANGLE = 0
ANGLE_1085C = 359.5

DEFAULT_MIN = -5
DEFAULT_MAX = 5
DEFAULT_RESOLUTION = 0.2


class ThirdWindow(QDialog):
    """ Third window of the gui
    """
    def __init__(self, rotate_parameters):
        super().__init__()
        self.rotate_parameters = rotate_parameters
        builder.build_third_window(self)
        self.ok = False

    def slot_confirm_parameters(self):
        self.rotate_parameters[0] = float(self.lineEdit_1.text())
        self.rotate_parameters[1] = float(self.lineEdit_2.text())
        self.rotate_parameters[2] = float(self.lineEdit_3.text())
        self.ok = True
        self.close()

    def set_text(self):
        self.setWindowTitle("Automatic rotation")
        self.label_1.setText("min")
        self.label_2.setText("max")
        self.label_3.setText("resolution")
        self.push_button.setText("OK")
        self.lineEdit_1.setText(str(self.rotate_parameters[0]))
        self.lineEdit_2.setText(str(self.rotate_parameters[1]))
        self.lineEdit_3.setText(str(self.rotate_parameters[2]))


class ParamWindow(builder.QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, parent=None):
        super(ParamWindow, self).__init__(parent)
        builder.build_param_window(self)
        self.threadpool = QThreadPool()

        self.angle = ANGLE
        self.scale = SCALE
        self.rotate_parameters = [DEFAULT_MIN, DEFAULT_MAX, DEFAULT_RESOLUTION]

    def slot_set_default(self):
        self.scale = float(SCALE)
        self.angle = float(ANGLE)
        self.lineEdit.setText(str(SCALE))
        self.lineEdit_2.setText(str(ANGLE))

    def slot_save(self):
        path = QFileDialog.getSaveFileName(self, "Save", filter="Pickle file (*.pkl)")[0]
        if path:
            filename, file_extension = analyzer.os.path.splitext(path)
            if file_extension == ".pkl":
                with open(path, 'wb') as f:
                    pickle.dump([self.lineEdit.text(), self.lineEdit_2.text()], f)
                    msgBox = QMessageBox()
                    msgBox.setText("The parameters have been saved")
                    msgBox.setWindowTitle("Save")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec_()

    def slot_load(self):
        path = QFileDialog.getOpenFileName(self, "Load")[0]
        if path:
            filename, file_extension = analyzer.os.path.splitext(path)
            if file_extension == ".pkl":
                with open(path, 'rb') as f:
                    obj1, obj2 = pickle.load(f)
                    self.lineEdit.setText(obj1)
                    self.lineEdit_2.setText(obj2)
            else:
                msgBox = QMessageBox()
                msgBox.setText("Bad format")
                msgBox.setWindowTitle("Error")
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()

    def slot_ok(self):
        self.scale = float(self.lineEdit.text())
        self.angle = float(self.lineEdit_2.text())
        self.hide()

    def slot_cancel(self):
        self.lineEdit.setText(str(self.scale))
        self.lineEdit_2.setText(str(self.angle))
        self.hide()

    def slot_angle_finder(self):
        path = QFileDialog.getOpenFileName(self, "Open", filter="Image files (*.tif *.png *.jpeg)")[0]

        if path:
            small_window = ThirdWindow(self.rotate_parameters)
            small_window.exec_()

            if small_window.ok:

                self.rotate_parameters = small_window.rotate_parameters

                min_angle = small_window.rotate_parameters[0]
                max_angle = small_window.rotate_parameters[1]
                resolution = small_window.rotate_parameters[2]

                if resolution >= max_angle - min_angle:
                    show_bad_dialog("<h4>The resolution is smaller than the covered range</h4>")
                else:
                    finder = angle_finder.AutoRotator(path, small_window.rotate_parameters)

                    worker = Worker(finder.calc_optimal_transformation)
                    worker.signals.result.connect(self.get_result)
                    worker.signals.progress.connect(self.update_progression)
                    self.threadpool.start(worker)

    def get_result(self, result):
        self.lineEdit_2.setText(str((result[0])))
        self.angle = float(result[0])

    def update_progression(self, obj):
        if obj:
            (progression, max_value, angle) = obj
            self.progressBar.setValue(progression)

            if angle is not None:
                self.label_3.setText(f'Current best is {max_value}°. Checking {angle}°...')
            else:
                self.label_3.setText(f'Best is {max_value}°.')

    def set_text(self):
        self.setWindowTitle("Parameters")
        self.label.setText("Scale")
        self.lineEdit.setText(str(SCALE))
        self.label_2.setText("Rotation angle")
        self.lineEdit_2.setText(str(ANGLE))
        self.pushButton.setText("Set default")
        self.pushButton_2.setText("Ok")
        self.pushButton_3.setText("Cancel")
        self.pushButton_4.setText("Save to file")
        self.pushButton_5.setText("Load parameters")
        self.pushButton_6.setText("Angle finder")


class GuiMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.number_of_files = 0
        self.data = []
        self.options = [False, False, False, False, False, False]
        self.save = False
        self.threadpool = QThreadPool()
        self.param = ParamWindow()
        self.path = "../output"
        builder.build_main_window(self)

    def update_state(self):
        self.options[0] = self.checkBoxArray[0].isChecked()
        self.options[1] = self.checkBoxArray[1].isChecked()
        self.options[2] = self.checkBoxArray[2].isChecked()
        self.options[3] = self.checkBoxArray[3].isChecked()
        self.options[4] = self.checkBoxArray[4].isChecked()
        self.options[5] = self.checkBoxArray[5].isChecked()
        self.save = self.radioButton_2.isChecked()
        self.path = self.lineEdit_3.text()

    def slot_select_all(self):
        for check_box in self.checkBoxArray:
            if self.checkBox_7.isChecked():
                check_box.setChecked(True)
            else:
                check_box.setChecked(False)

    def slot_load_image(self):
        """
        Slot connected to the button "Load image".
        :return: none
        """
        paths = QFileDialog.getOpenFileNames(self, "Open", filter="Image files (*.tif *.png *.jpeg)")[0]
        if paths:
            self.file_path = []
            self.number_of_files = 0
            for path in paths:
                filename, file_extension = analyzer.os.path.splitext(path)
                if file_extension == ".tif" or file_extension == ".png" or file_extension == ".jpeg":
                    self.file_path.append(path)
                    self.number_of_files += 1
            if self.number_of_files == 0:
                show_bad_dialog(msg="Bad image format")

            self.lineEdit_1.setText(str(paths[0]))

        if self.number_of_files == 1:
            self.statusBar().showMessage("The image has been loaded !", 3000)
        else:
            self.statusBar().showMessage(str(self.number_of_files) + " images have been loaded !", 3000)

    def slot_run(self):
        """
        Slot connected to the button "Run".
        :return: none
        """
        self.update_state()

        # Run the funtion slot_run_helper in another thread
        worker = Worker(self.slot_run_helper)

        if self.file_path:
            self.threadpool.start(worker)
        else:
            show_bad_dialog(msg="No image loaded")
            self.lineEdit_1.setText("")
            self.lineEdit_2.setText("0.0")

    def slot_run_helper(self, progress_callback):
        if self.save:
            self.progressBar.setTextVisible(True)

        num_file = len(self.file_path)
        i = 100.0 / num_file
        length = 0

        for path in self.file_path:
            if path:
                self.data = analyzer.main(path, self.param.angle, self.param.scale)
                self.lineEdit_1.setText(str(path))
                self.lineEdit_2.setText(str(self.data["period"])[:PRECISION] + " um")
                analyzer.display_graphs(self.data, self.options, path, self.path, self.save, self.param.scale)

                if self.save:
                    length = length + i
                    self.progressBar.setProperty("value", length)

        if self.save:
            time.sleep(1)
            self.progressBar.setTextVisible(False)
            self.progressBar.setProperty("value", 0)

    def slot_browse(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            path = dlg.selectedFiles()[0]
            self.lineEdit_3.setText(path)

    def slot_parameters(self):
        '''
        To keep a reference to the w variable, we use the self object.
        :return:
        '''
        self.param.show()

    def slot_about(self):
        """
        Display the 'about' window
        :return: none
        """
        text = "<center>" \
               "<h1>Striations analyzer</h1></center>" \
               "<p>New features:<br/>" \
               "- Automatic image rotation</p>" \
               "<center>&#8291;" \
               "<img src=res/just_logo.png>" \
               "</center>" \
               "<p>Version 2.0<br/>" \
               "For internal use, Agranat Lab.<br/>"\
               "Developped by R. Haehnel.<br/>"\
               "&copy; 2021 Hebrew University</p>"
        QMessageBox.about(self, "About Striations Analyzer", text)
    
    def setText(self):
        self.setWindowTitle("Striations analyzer")
        self.pushButton.setText("Load image")
        self.groupBox.setTitle("Items to display/save")
        self.checkBoxArray[0].setText("Original image")
        self.checkBoxArray[1].setText("Rotated image")
        self.checkBoxArray[2].setText("Filtered image")
        self.checkBoxArray[3].setText("Graph line")
        self.checkBoxArray[4].setText("Image line")
        self.checkBoxArray[5].setText("Fourier transform")
        self.checkBox_7.setText("Select all")
        self.groupBox_2.setTitle("Render")
        self.radioButton.setText("Display")
        self.radioButton_2.setText("Save")
        self.pushButton_2.setText("Run")
        self.groupBox_3.setTitle("Output")
        self.label_1.setText("File path")
        self.label_2.setText("Period")
        self.lineEdit_2.setText("0.0")
        self.label_3.setText("Save as")
        self.pushButton_3.setText("Browse")
        self.lineEdit_3.setText(self.path)
        self.menuFile.setTitle("File")
        self.menuHelp.setTitle("Help")
        self.actionDefine_parameters.setText("Define parameters")
        self.actionAbout.setText("About")


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:

    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    progress
        int indicating % progress
    '''

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(list)


class Worker(QRunnable):
    '''
    Worker thread
    Code from https://www.learnpyqt.com/tutorials/multithreading-pyqt-applications-qthreadpool/
    '''

    def __init__(self, foo, *args, **kwargs):
        super(Worker, self).__init__()
        self.foo = foo
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.foo(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


def show_bad_dialog(msg):
    msgBox = QMessageBox()
    msgBox.setText(msg)
    msgBox.setWindowTitle("Error")
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec_()


if __name__ == "__main__":
    app = QApplication(analyzer.sys.argv)
    ui = GuiMainWindow()
    ui.show()
    sys.exit(app.exec_())
