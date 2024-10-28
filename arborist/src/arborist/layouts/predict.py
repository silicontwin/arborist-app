# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'predict.ui'
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
    QPushButton, QSizePolicy, QTableView, QVBoxLayout,
    QWidget)

class Ui_PredictTab(object):
    def setupUi(self, PredictTab):
        if not PredictTab.objectName():
            PredictTab.setObjectName(u"PredictTab")
        PredictTab.resize(800, 600)
        self.verticalLayout = QVBoxLayout(PredictTab)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.selectFileButton = QPushButton(PredictTab)
        self.selectFileButton.setObjectName(u"selectFileButton")

        self.horizontalLayout.addWidget(self.selectFileButton)

        self.selectedFileLabel = QLabel(PredictTab)
        self.selectedFileLabel.setObjectName(u"selectedFileLabel")

        self.horizontalLayout.addWidget(self.selectedFileLabel)

        self.predictButton = QPushButton(PredictTab)
        self.predictButton.setObjectName(u"predictButton")

        self.horizontalLayout.addWidget(self.predictButton)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.tableView = QTableView(PredictTab)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setSortingEnabled(True)

        self.verticalLayout.addWidget(self.tableView)


        self.retranslateUi(PredictTab)

        QMetaObject.connectSlotsByName(PredictTab)
    # setupUi

    def retranslateUi(self, PredictTab):
        PredictTab.setWindowTitle(QCoreApplication.translate("PredictTab", u"Predict", None))
        self.selectFileButton.setText(QCoreApplication.translate("PredictTab", u"Load File", None))
        self.selectedFileLabel.setText(QCoreApplication.translate("PredictTab", u"Selected file will appear here", None))
        self.predictButton.setText(QCoreApplication.translate("PredictTab", u"Predict", None))
    # retranslateUi

