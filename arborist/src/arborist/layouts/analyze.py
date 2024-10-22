# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'analyze.ui'
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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel,
    QSizePolicy, QTableView, QVBoxLayout, QWidget)

class Ui_AnalyzeTab(object):
    def setupUi(self, AnalyzeTab):
        if not AnalyzeTab.objectName():
            AnalyzeTab.setObjectName(u"AnalyzeTab")
        self.verticalLayout = QVBoxLayout(AnalyzeTab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.no_dataset_label = QLabel(AnalyzeTab)
        self.no_dataset_label.setObjectName(u"no_dataset_label")
        self.no_dataset_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.no_dataset_label)

        self.analytics_viewer = QTableView(AnalyzeTab)
        self.analytics_viewer.setObjectName(u"analytics_viewer")
        self.analytics_viewer.setSortingEnabled(True)

        self.verticalLayout.addWidget(self.analytics_viewer)

        self.toolbar = QWidget(AnalyzeTab)
        self.toolbar.setObjectName(u"toolbar")
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        self.toolbar_layout.setObjectName(u"toolbar_layout")

        self.verticalLayout.addWidget(self.toolbar)


        self.retranslateUi(AnalyzeTab)

        QMetaObject.connectSlotsByName(AnalyzeTab)
    # setupUi

    def retranslateUi(self, AnalyzeTab):
        self.no_dataset_label.setText(QCoreApplication.translate("AnalyzeTab", u"Please select a dataset from the \"Browse\" tab", None))
        pass
    # retranslateUi

