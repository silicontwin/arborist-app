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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QHeaderView,
    QPushButton, QSizePolicy, QSpacerItem, QTableView,
    QVBoxLayout, QWidget)

class Ui_PredictTab(object):
    def setupUi(self, PredictTab):
        if not PredictTab.objectName():
            PredictTab.setObjectName(u"PredictTab")
        PredictTab.resize(800, 600)
        self.verticalLayout = QVBoxLayout(PredictTab)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.tableView = QTableView(PredictTab)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setSortingEnabled(True)

        self.verticalLayout.addWidget(self.tableView)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.frame = QFrame(PredictTab)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.selectFileButton = QPushButton(self.frame)
        self.selectFileButton.setObjectName(u"selectFileButton")

        self.horizontalLayout_2.addWidget(self.selectFileButton)

        self.predictButton = QPushButton(self.frame)
        self.predictButton.setObjectName(u"predictButton")

        self.horizontalLayout_2.addWidget(self.predictButton)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.resetButton = QPushButton(self.frame)
        self.resetButton.setObjectName(u"resetButton")

        self.horizontalLayout_2.addWidget(self.resetButton)


        self.horizontalLayout.addWidget(self.frame)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(PredictTab)

        QMetaObject.connectSlotsByName(PredictTab)
    # setupUi

    def retranslateUi(self, PredictTab):
        PredictTab.setWindowTitle(QCoreApplication.translate("PredictTab", u"Predict", None))
        self.selectFileButton.setText(QCoreApplication.translate("PredictTab", u"Load File", None))
        self.predictButton.setText(QCoreApplication.translate("PredictTab", u"Predict", None))
        self.resetButton.setText(QCoreApplication.translate("PredictTab", u"Reset", None))
    # retranslateUi

