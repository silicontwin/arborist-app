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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QPushButton,
    QSizePolicy, QSplitter, QTableView, QTreeView,
    QVBoxLayout, QWidget)

class Ui_BrowseTab(object):
    def setupUi(self, BrowseTab):
        if not BrowseTab.objectName():
            BrowseTab.setObjectName(u"BrowseTab")
        BrowseTab.resize(1600, 900)
        BrowseTab.setBaseSize(QSize(1600, 900))
        self.verticalLayout = QVBoxLayout(BrowseTab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.navigationLayout = QHBoxLayout()
        self.navigationLayout.setObjectName(u"navigationLayout")
        self.back_button = QPushButton(BrowseTab)
        self.back_button.setObjectName(u"back_button")
        self.back_button.setEnabled(False)

        self.navigationLayout.addWidget(self.back_button)

        self.forward_button = QPushButton(BrowseTab)
        self.forward_button.setObjectName(u"forward_button")
        self.forward_button.setEnabled(False)

        self.navigationLayout.addWidget(self.forward_button)


        self.verticalLayout.addLayout(self.navigationLayout)

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

        self.openDatasetButton = QPushButton(BrowseTab)
        self.openDatasetButton.setObjectName(u"openDatasetButton")
        self.openDatasetButton.setVisible(False)

        self.verticalLayout.addWidget(self.openDatasetButton)


        self.retranslateUi(BrowseTab)

        QMetaObject.connectSlotsByName(BrowseTab)
    # setupUi

    def retranslateUi(self, BrowseTab):
        self.back_button.setText(QCoreApplication.translate("BrowseTab", u"Back", None))
        self.forward_button.setText(QCoreApplication.translate("BrowseTab", u"Forward", None))
        self.openDatasetButton.setText(QCoreApplication.translate("BrowseTab", u"Open Dataset", None))
        pass
    # retranslateUi

