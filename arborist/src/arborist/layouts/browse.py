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
from PySide6.QtWidgets import (QApplication, QHeaderView, QPushButton, QSizePolicy,
    QSplitter, QTableView, QTreeView, QVBoxLayout,
    QWidget)

class Ui_BrowseTab(object):
    def setupUi(self, BrowseTab):
        if not BrowseTab.objectName():
            BrowseTab.setObjectName(u"BrowseTab")
        self.verticalLayout = QVBoxLayout(BrowseTab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.splitter = QSplitter(BrowseTab)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.treeView = QTreeView(self.splitter)
        self.treeView.setObjectName(u"treeView")
        self.treeView.setMinimumSize(QSize(200, 0))
        self.splitter.addWidget(self.treeView)
        self.file_viewer = QTableView(self.splitter)
        self.file_viewer.setObjectName(u"file_viewer")
        self.file_viewer.setSortingEnabled(True)
        self.splitter.addWidget(self.file_viewer)

        self.verticalLayout.addWidget(self.splitter)

        self.analyze_button = QPushButton(BrowseTab)
        self.analyze_button.setObjectName(u"analyze_button")
        self.analyze_button.setVisible(False)

        self.verticalLayout.addWidget(self.analyze_button)


        self.retranslateUi(BrowseTab)

        QMetaObject.connectSlotsByName(BrowseTab)
    # setupUi

    def retranslateUi(self, BrowseTab):
        self.analyze_button.setText(QCoreApplication.translate("BrowseTab", u"Analyze Dataset", None))
        pass
    # retranslateUi

