# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'load.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout,
    QHeaderView, QPushButton, QSizePolicy, QSpacerItem,
    QSplitter, QTableView, QTreeView, QVBoxLayout,
    QWidget)

class Ui_LoadTab(object):
    def setupUi(self, LoadTab):
        if not LoadTab.objectName():
            LoadTab.setObjectName(u"LoadTab")
        LoadTab.resize(1600, 900)
        LoadTab.setBaseSize(QSize(1600, 900))
        self.verticalLayout = QVBoxLayout(LoadTab)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(LoadTab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Orientation.Horizontal)
        self.treeView = QTreeView(self.splitter)
        self.treeView.setObjectName(u"treeView")
        self.treeView.setMinimumSize(QSize(200, 0))
        self.splitter.addWidget(self.treeView)
        self.file_viewer = QTableView(self.splitter)
        self.file_viewer.setObjectName(u"file_viewer")
        self.file_viewer.setSortingEnabled(True)
        self.splitter.addWidget(self.file_viewer)

        self.verticalLayout.addWidget(self.splitter)

        self.bottomMenu = QFrame(LoadTab)
        self.bottomMenu.setObjectName(u"bottomMenu")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottomMenu.sizePolicy().hasHeightForWidth())
        self.bottomMenu.setSizePolicy(sizePolicy)
        self.bottomMenu.setFrameShape(QFrame.Shape.NoFrame)
        self.bottomMenu.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.bottomMenu)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.navigationFrame = QFrame(self.bottomMenu)
        self.navigationFrame.setObjectName(u"navigationFrame")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.navigationFrame.sizePolicy().hasHeightForWidth())
        self.navigationFrame.setSizePolicy(sizePolicy1)
        self.navigationFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.navigationFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.navigationLayout = QHBoxLayout(self.navigationFrame)
        self.navigationLayout.setSpacing(5)
        self.navigationLayout.setObjectName(u"navigationLayout")
        self.navigationLayout.setContentsMargins(0, 0, 0, 0)
        self.back_button = QPushButton(self.navigationFrame)
        self.back_button.setObjectName(u"back_button")
        self.back_button.setEnabled(False)
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.back_button.sizePolicy().hasHeightForWidth())
        self.back_button.setSizePolicy(sizePolicy2)
        self.back_button.setMinimumSize(QSize(0, 0))

        self.navigationLayout.addWidget(self.back_button)

        self.up_button = QPushButton(self.navigationFrame)
        self.up_button.setObjectName(u"up_button")
        self.up_button.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.up_button.sizePolicy().hasHeightForWidth())
        self.up_button.setSizePolicy(sizePolicy2)
        self.up_button.setMinimumSize(QSize(0, 0))

        self.navigationLayout.addWidget(self.up_button)

        self.forward_button = QPushButton(self.navigationFrame)
        self.forward_button.setObjectName(u"forward_button")
        self.forward_button.setEnabled(False)
        sizePolicy2.setHeightForWidth(self.forward_button.sizePolicy().hasHeightForWidth())
        self.forward_button.setSizePolicy(sizePolicy2)
        self.forward_button.setMinimumSize(QSize(0, 0))

        self.navigationLayout.addWidget(self.forward_button)


        self.horizontalLayout.addWidget(self.navigationFrame)

        self.rememberDirCheckBox = QCheckBox(self.bottomMenu)
        self.rememberDirCheckBox.setObjectName(u"rememberDirCheckBox")
        self.rememberDirCheckBox.setChecked(True)

        self.horizontalLayout.addWidget(self.rememberDirCheckBox)

        self.horizontalSpacer = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.openDatasetButton = QPushButton(self.bottomMenu)
        self.openDatasetButton.setObjectName(u"openDatasetButton")
        sizePolicy2.setHeightForWidth(self.openDatasetButton.sizePolicy().hasHeightForWidth())
        self.openDatasetButton.setSizePolicy(sizePolicy2)
        self.openDatasetButton.setMinimumSize(QSize(0, 0))
        self.openDatasetButton.setVisible(True)

        self.horizontalLayout.addWidget(self.openDatasetButton)


        self.verticalLayout.addWidget(self.bottomMenu)


        self.retranslateUi(LoadTab)

        QMetaObject.connectSlotsByName(LoadTab)
    # setupUi

    def retranslateUi(self, LoadTab):
        self.back_button.setText(QCoreApplication.translate("LoadTab", u"Back", None))
        self.up_button.setText(QCoreApplication.translate("LoadTab", u"Up", None))
        self.forward_button.setText(QCoreApplication.translate("LoadTab", u"Forward", None))
        self.rememberDirCheckBox.setText(QCoreApplication.translate("LoadTab", u"Remember current directory", None))
        self.openDatasetButton.setText(QCoreApplication.translate("LoadTab", u"Open Dataset", None))
        pass
    # retranslateUi

