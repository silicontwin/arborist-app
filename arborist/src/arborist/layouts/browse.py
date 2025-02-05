# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'browse.ui'
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QHeaderView,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QTableView, QTreeView, QVBoxLayout, QWidget)

class Ui_BrowseTab(object):
    def setupUi(self, BrowseTab):
        if not BrowseTab.objectName():
            BrowseTab.setObjectName(u"BrowseTab")
        BrowseTab.resize(1600, 900)
        BrowseTab.setBaseSize(QSize(1600, 900))
        self.verticalLayout = QVBoxLayout(BrowseTab)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.splitter = QSplitter(BrowseTab)
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

        self.bottomMenu = QFrame(BrowseTab)
        self.bottomMenu.setObjectName(u"bottomMenu")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bottomMenu.sizePolicy().hasHeightForWidth())
        self.bottomMenu.setSizePolicy(sizePolicy)
        self.bottomMenu.setFrameShape(QFrame.Shape.NoFrame)
        self.bottomMenu.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.bottomMenu)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
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
        self.navigationLayout.setSpacing(10)
        self.navigationLayout.setObjectName(u"navigationLayout")
        self.navigationLayout.setContentsMargins(0, 0, 0, 0)
        self.back_button = QPushButton(self.navigationFrame)
        self.back_button.setObjectName(u"back_button")
        self.back_button.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.back_button.sizePolicy().hasHeightForWidth())
        self.back_button.setSizePolicy(sizePolicy1)
        self.back_button.setMinimumSize(QSize(80, 30))

        self.navigationLayout.addWidget(self.back_button)

        self.up_button = QPushButton(self.navigationFrame)
        self.up_button.setObjectName(u"up_button")
        self.up_button.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.up_button.sizePolicy().hasHeightForWidth())
        self.up_button.setSizePolicy(sizePolicy1)
        self.up_button.setMinimumSize(QSize(80, 30))

        self.navigationLayout.addWidget(self.up_button)

        self.forward_button = QPushButton(self.navigationFrame)
        self.forward_button.setObjectName(u"forward_button")
        self.forward_button.setEnabled(False)
        sizePolicy1.setHeightForWidth(self.forward_button.sizePolicy().hasHeightForWidth())
        self.forward_button.setSizePolicy(sizePolicy1)
        self.forward_button.setMinimumSize(QSize(80, 30))

        self.navigationLayout.addWidget(self.forward_button)


        self.horizontalLayout.addWidget(self.navigationFrame)

        self.horizontalSpacer = QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.openDatasetButton = QPushButton(self.bottomMenu)
        self.openDatasetButton.setObjectName(u"openDatasetButton")
        sizePolicy1.setHeightForWidth(self.openDatasetButton.sizePolicy().hasHeightForWidth())
        self.openDatasetButton.setSizePolicy(sizePolicy1)
        self.openDatasetButton.setMinimumSize(QSize(120, 30))
        self.openDatasetButton.setVisible(True)

        self.horizontalLayout.addWidget(self.openDatasetButton)


        self.verticalLayout.addWidget(self.bottomMenu)


        self.retranslateUi(BrowseTab)

        QMetaObject.connectSlotsByName(BrowseTab)
    # setupUi

    def retranslateUi(self, BrowseTab):
        self.back_button.setText(QCoreApplication.translate("BrowseTab", u"Back", None))
        self.up_button.setText(QCoreApplication.translate("BrowseTab", u"Up", None))
        self.forward_button.setText(QCoreApplication.translate("BrowseTab", u"Forward", None))
        self.openDatasetButton.setText(QCoreApplication.translate("BrowseTab", u"Open Dataset", None))
        pass
    # retranslateUi

