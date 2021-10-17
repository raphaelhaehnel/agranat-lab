from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,\
    QLabel, QPushButton, QGroupBox, QCheckBox, QRadioButton, QSizePolicy, QMenuBar, QMenu, QStatusBar, \
    QAction, QSpacerItem, QProgressBar, QFrame
from PyQt5.QtCore import QRect, QMetaObject

PATH_ICON = "./res/logo-transparent.ico"


def build_main_window(obj):
    """
    Builder for the main window
    :param obj: The main widget
    :param MainWindow: an object QMainWindow
    :return: None
    """
    
    """ Widgets """
    
    obj.setObjectName("MainWindow")
    obj.setFixedSize(551, 398)
    set_icon(obj, PATH_ICON)
    obj.centralwidget = QWidget(obj)
    obj.centralwidget.setObjectName("centralwidget")
    obj.horizontalMain = QHBoxLayout(obj.centralwidget)
    obj.pushButton = QPushButton(obj.centralwidget)
    obj.pushButton.setGeometry(QRect(60, 30, 93, 28))
    obj.pushButton.setObjectName("pushButton")
    obj.groupBox = QGroupBox(obj.centralwidget)
    obj.groupBox.setGeometry(QRect(30, 80, 181, 261))
    obj.groupBox.setObjectName("groupBox")
    obj.verticalLayoutWidget = QWidget(obj.groupBox)
    obj.verticalLayoutWidget.setGeometry(QRect(10, 20, 160, 231))
    obj.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
    obj.verticalLayout = QVBoxLayout(obj.verticalLayoutWidget)
    obj.verticalLayout.setContentsMargins(0, 0, 0, 0)
    obj.verticalLayout.setObjectName("verticalLayout")
    obj.checkBox_7 = QCheckBox(obj.verticalLayoutWidget)
    obj.checkBox_7.setObjectName("checkBox_7")
    obj.verticalLayout.addWidget(obj.checkBox_7)
    obj.line = QFrame(obj.verticalLayoutWidget)
    obj.line.setFrameShape(QFrame.HLine)
    obj.line.setFrameShadow(QFrame.Sunken)
    obj.line.setObjectName("line")
    obj.verticalLayout.addWidget(obj.line)
    checkBox_1 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_2 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_3 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_4 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_5 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_6 = QCheckBox(obj.verticalLayoutWidget)
    checkBox_1.setObjectName("checkBox_1")
    checkBox_2.setObjectName("checkBox_2")
    checkBox_3.setObjectName("checkBox_3")
    checkBox_4.setObjectName("checkBox_4")
    checkBox_5.setObjectName("checkBox_5")
    checkBox_6.setObjectName("checkBox_6")
    obj.checkBoxArray = [checkBox_1, checkBox_2, checkBox_3, checkBox_4, checkBox_5, checkBox_6]
    for check_box in obj.checkBoxArray:
        obj.verticalLayout.addWidget(check_box)
    obj.groupBox_2 = QGroupBox(obj.centralwidget)
    obj.groupBox_2.setGeometry(QRect(220, 80, 131, 81))
    obj.groupBox_2.setObjectName("groupBox_2")
    obj.verticalLayoutWidget_2 = QWidget(obj.groupBox_2)
    obj.verticalLayoutWidget_2.setGeometry(QRect(10, 20, 111, 51))
    obj.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
    obj.verticalLayout_2 = QVBoxLayout(obj.verticalLayoutWidget_2)
    obj.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
    obj.verticalLayout_2.setObjectName("verticalLayout_2")
    obj.radioButton = QRadioButton(obj.verticalLayoutWidget_2)
    obj.radioButton.setChecked(True)
    obj.radioButton.setObjectName("radioButton")
    obj.verticalLayout_2.addWidget(obj.radioButton)
    obj.radioButton_2 = QRadioButton(obj.verticalLayoutWidget_2)
    obj.radioButton_2.setChecked(False)
    obj.radioButton_2.setObjectName("radioButton_2")
    obj.verticalLayout_2.addWidget(obj.radioButton_2)
    obj.pushButton_2 = QPushButton(obj.centralwidget)
    obj.pushButton_2.setGeometry(QRect(230, 30, 93, 28))
    obj.pushButton_2.setObjectName("pushButton_2")
    obj.progressBar = QProgressBar(obj.centralwidget)
    obj.progressBar.setGeometry(QRect(360, 30, 171, 23))
    obj.progressBar.setProperty("value", 0)
    obj.progressBar.setTextVisible(False)
    obj.progressBar.setObjectName("progressBar")
    obj.groupBox_3 = QGroupBox(obj.centralwidget)
    obj.groupBox_3.setGeometry(QRect(370, 80, 161, 261))
    obj.groupBox_3.setObjectName("groupBox_3")
    obj.verticalLayoutWidget_3 = QWidget(obj.groupBox_3)
    obj.verticalLayoutWidget_3.setGeometry(QRect(10, 20, 141, 240))
    obj.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
    obj.verticalLayout_3 = QVBoxLayout(obj.verticalLayoutWidget_3)
    obj.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
    obj.verticalLayout_3.setObjectName("verticalLayout_3")
    obj.label_1 = QLabel(obj.verticalLayoutWidget_3)
    obj.label_1.setObjectName("label_1")
    obj.verticalLayout_3.addWidget(obj.label_1)
    obj.lineEdit_1 = QLineEdit(obj.verticalLayoutWidget_3)
    obj.lineEdit_1.setEnabled(True)
    obj.lineEdit_1.setText("")
    obj.lineEdit_1.setReadOnly(True)
    obj.lineEdit_1.setClearButtonEnabled(False)
    obj.lineEdit_1.setObjectName("lineEdit_1")
    obj.verticalLayout_3.addWidget(obj.lineEdit_1)
    spacerItem = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
    obj.verticalLayout_3.addItem(spacerItem)
    obj.label_2 = QLabel(obj.verticalLayoutWidget_3)
    obj.label_2.setObjectName("label_2")
    obj.verticalLayout_3.addWidget(obj.label_2)
    obj.lineEdit_2 = QLineEdit(obj.verticalLayoutWidget_3)
    obj.lineEdit_2.setEnabled(True)
    obj.lineEdit_2.setAutoFillBackground(False)
    obj.lineEdit_2.setReadOnly(True)
    obj.lineEdit_2.setObjectName("lineEdit_2")
    obj.verticalLayout_3.addWidget(obj.lineEdit_2)
    spacerItem1 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
    obj.verticalLayout_3.addItem(spacerItem1)
    obj.label_3 = QLabel(obj.verticalLayoutWidget_3)
    obj.label_3.setObjectName("label_3")
    obj.verticalLayout_3.addWidget(obj.label_3)
    obj.pushButton_3 = QPushButton(obj.verticalLayoutWidget_3)
    obj.pushButton_3.setObjectName("pushButton_3")
    obj.verticalLayout_3.addWidget(obj.pushButton_3)
    obj.lineEdit_3 = QLineEdit(obj.verticalLayoutWidget_3)
    obj.lineEdit_3.setEnabled(True)
    obj.lineEdit_3.setText("")
    obj.lineEdit_3.setReadOnly(True)
    obj.lineEdit_3.setClearButtonEnabled(False)
    obj.lineEdit_3.setObjectName("lineEdit_3")
    obj.verticalLayout_3.addWidget(obj.lineEdit_3)
    obj.setCentralWidget(obj.centralwidget)
    obj.menubar = QMenuBar(obj)
    obj.menubar.setGeometry(QRect(0, 0, 546, 26))
    obj.menubar.setObjectName("menubar")
    obj.menuFile = QMenu(obj.menubar)
    obj.menuFile.setObjectName("menuFile")
    obj.menuHelp = QMenu(obj.menubar)
    obj.menuHelp.setObjectName("menuHelp")
    obj.setMenuBar(obj.menubar)
    obj.statusbar = QStatusBar(obj)
    obj.statusbar.setObjectName("statusbar")
    obj.setStatusBar(obj.statusbar)
    obj.actionDefine_parameters = QAction(obj)
    obj.actionDefine_parameters.setObjectName("actionDefine_parameters")
    obj.actionAbout = QAction(obj)
    obj.actionAbout.setObjectName("actionAbout")
    obj.menuFile.addAction(obj.actionDefine_parameters)
    obj.menuHelp.addAction(obj.actionAbout)
    obj.menubar.addAction(obj.menuFile.menuAction())
    obj.menubar.addAction(obj.menuHelp.menuAction())
    obj.setText()

    """ Events """

    obj.pushButton.clicked.connect(obj.slot_load_image)
    obj.pushButton_2.clicked.connect(obj.slot_run)
    obj.pushButton_3.clicked.connect(obj.slot_browse)
    obj.actionDefine_parameters.triggered.connect(obj.slot_parameters)
    obj.actionAbout.triggered.connect(obj.slot_about)
    obj.checkBox_7.clicked.connect(obj.slot_select_all)


def build_param_window(obj):
    """
    Builder for the parameter window
    :param obj: The main widget
    :return: None
    """

    """ Widgets """

    set_icon(obj, PATH_ICON)
    obj.resize(435, 250)
    obj.verticalLayoutWidget_2 = QWidget(obj)
    obj.verticalLayoutWidget_2.setGeometry(QRect(10, 10, 415, 230))
    obj.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
    obj.verticalLayout_2 = QVBoxLayout(obj.verticalLayoutWidget_2)
    obj.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
    obj.verticalLayout_2.setObjectName("verticalLayout_2")
    obj.horizontalLayout_2 = QHBoxLayout()
    obj.horizontalLayout_2.setObjectName("horizontalLayout_2")
    obj.label = QLabel(obj.verticalLayoutWidget_2)
    obj.label.setObjectName("label")
    obj.horizontalLayout_2.addWidget(obj.label)
    obj.lineEdit = QLineEdit(obj.verticalLayoutWidget_2)
    obj.lineEdit.setMaxLength(15)
    obj.lineEdit.setObjectName("lineEdit")
    obj.horizontalLayout_2.addWidget(obj.lineEdit)
    obj.verticalLayout_2.addLayout(obj.horizontalLayout_2)
    obj.horizontalLayout_3 = QHBoxLayout()
    obj.horizontalLayout_3.setObjectName("horizontalLayout_3")
    obj.label_2 = QLabel(obj.verticalLayoutWidget_2)
    obj.label_2.setObjectName("label_2")
    obj.horizontalLayout_3.addWidget(obj.label_2)
    obj.lineEdit_2 = QLineEdit(obj.verticalLayoutWidget_2)
    obj.lineEdit_2.setMaxLength(10)
    obj.lineEdit_2.setObjectName("lineEdit_2")
    obj.horizontalLayout_3.addWidget(obj.lineEdit_2)
    obj.pushButton_6 = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton_6.setObjectName("pushButton_6")
    obj.horizontalLayout_3.addWidget(obj.pushButton_6)
    obj.verticalLayout_2.addLayout(obj.horizontalLayout_3)
    obj.verticalLayout_3 = QVBoxLayout()
    obj.verticalLayout_3.setObjectName("verticalLayout_3")
    obj.pushButton_4 = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton_4.setObjectName("pushButton_4")
    obj.verticalLayout_3.addWidget(obj.pushButton_4)
    obj.pushButton_5 = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton_5.setObjectName("pushButton_5")
    obj.verticalLayout_3.addWidget(obj.pushButton_5)
    obj.verticalLayout_2.addLayout(obj.verticalLayout_3)
    obj.horizontalLayout_4 = QHBoxLayout()
    obj.horizontalLayout_4.setObjectName("horizontalLayout_4")
    obj.pushButton = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton.setObjectName("pushButton")
    obj.horizontalLayout_4.addWidget(obj.pushButton)
    obj.pushButton_2 = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton_2.setObjectName("pushButton_2")
    obj.horizontalLayout_4.addWidget(obj.pushButton_2)
    obj.pushButton_3 = QPushButton(obj.verticalLayoutWidget_2)
    obj.pushButton_3.setObjectName("pushButton_3")
    obj.horizontalLayout_4.addWidget(obj.pushButton_3)
    obj.verticalLayout_2.addLayout(obj.horizontalLayout_4)
    obj.horizontalLayout = QHBoxLayout()
    obj.horizontalLayout.setObjectName("horizontalLayout")
    obj.progressBar = QProgressBar(obj.verticalLayoutWidget_2)
    obj.progressBar.setProperty("value", 0)
    obj.progressBar.setTextVisible(True)
    obj.progressBar.setObjectName("progressBar")
    obj.horizontalLayout.addWidget(obj.progressBar)
    obj.label_3 = QLabel(obj.verticalLayoutWidget_2)
    obj.label_3.setObjectName("label_3")
    obj.horizontalLayout.addWidget(obj.label_3)
    obj.verticalLayout_2.addLayout(obj.horizontalLayout)
    obj.set_text()
    
    """ Events """

    obj.pushButton.clicked.connect(obj.slot_set_default)
    obj.pushButton_2.clicked.connect(obj.slot_ok)
    obj.pushButton_3.clicked.connect(obj.slot_cancel)
    obj.pushButton_4.clicked.connect(obj.slot_save)
    obj.pushButton_5.clicked.connect(obj.slot_load)
    obj.pushButton_6.clicked.connect(obj.slot_angle_finder)

def build_third_window(obj):

    """ Widgets """

    set_icon(obj, PATH_ICON)
    obj.vertical_layout = QVBoxLayout(obj)
    obj.horizontal_layout = QHBoxLayout()
    obj.label_1 = QLabel(obj)
    obj.label_2 = QLabel(obj)
    obj.label_3 = QLabel(obj)
    obj.lineEdit_1 = QLineEdit(obj)
    obj.lineEdit_2 = QLineEdit(obj)
    obj.lineEdit_3 = QLineEdit(obj)
    obj.push_button = QPushButton(obj)
    obj.horizontal_layout.addWidget(obj.label_1)
    obj.horizontal_layout.addWidget(obj.lineEdit_1)
    obj.horizontal_layout.addWidget(obj.label_2)
    obj.horizontal_layout.addWidget(obj.lineEdit_2)
    obj.horizontal_layout.addWidget(obj.label_3)
    obj.horizontal_layout.addWidget(obj.lineEdit_3)
    obj.vertical_layout.addLayout(obj.horizontal_layout)
    obj.vertical_layout.addWidget(obj.push_button)
    obj.set_text()

    """ Events """

    obj.push_button.clicked.connect(obj.slot_confirm_parameters)


def set_icon(obj, path_icon):
    icon = QIcon()
    icon.addPixmap(QPixmap(path_icon), QIcon.Normal, QIcon.Off)
    icon.addPixmap(QPixmap(path_icon), QIcon.Normal, QIcon.On)
    obj.setWindowIcon(icon)
