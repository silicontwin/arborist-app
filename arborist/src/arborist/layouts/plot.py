# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plot.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
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
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_PlotTab(object):
    def setupUi(self, PlotTab):
        if not PlotTab.objectName():
            PlotTab.setObjectName(u"PlotTab")
        PlotTab.resize(800, 600)
        self.verticalLayout = QVBoxLayout(PlotTab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayoutPlot1 = QVBoxLayout()
        self.verticalLayoutPlot1.setObjectName(u"verticalLayoutPlot1")
        self.plotFrame1 = QFrame(PlotTab)
        self.plotFrame1.setObjectName(u"plotFrame1")
        self.plotFrame1.setFrameShape(QFrame.Box)
        self.plotFrame1.setFrameShadow(QFrame.Raised)
        self.plotFrame1.setMinimumSize(QSize(200, 200))

        self.verticalLayoutPlot1.addWidget(self.plotFrame1)

        self.copyCodeButton1 = QPushButton(PlotTab)
        self.copyCodeButton1.setObjectName(u"copyCodeButton1")

        self.verticalLayoutPlot1.addWidget(self.copyCodeButton1)


        self.horizontalLayout.addLayout(self.verticalLayoutPlot1)

        self.verticalLayoutPlot2 = QVBoxLayout()
        self.verticalLayoutPlot2.setObjectName(u"verticalLayoutPlot2")
        self.plotFrame2 = QFrame(PlotTab)
        self.plotFrame2.setObjectName(u"plotFrame2")
        self.plotFrame2.setFrameShape(QFrame.Box)
        self.plotFrame2.setFrameShadow(QFrame.Raised)
        self.plotFrame2.setMinimumSize(QSize(200, 200))

        self.verticalLayoutPlot2.addWidget(self.plotFrame2)

        self.copyCodeButton2 = QPushButton(PlotTab)
        self.copyCodeButton2.setObjectName(u"copyCodeButton2")

        self.verticalLayoutPlot2.addWidget(self.copyCodeButton2)


        self.horizontalLayout.addLayout(self.verticalLayoutPlot2)

        self.verticalLayoutPlot3 = QVBoxLayout()
        self.verticalLayoutPlot3.setObjectName(u"verticalLayoutPlot3")
        self.plotFrame3 = QFrame(PlotTab)
        self.plotFrame3.setObjectName(u"plotFrame3")
        self.plotFrame3.setFrameShape(QFrame.Box)
        self.plotFrame3.setFrameShadow(QFrame.Raised)
        self.plotFrame3.setMinimumSize(QSize(200, 200))

        self.verticalLayoutPlot3.addWidget(self.plotFrame3)

        self.copyCodeButton3 = QPushButton(PlotTab)
        self.copyCodeButton3.setObjectName(u"copyCodeButton3")

        self.verticalLayoutPlot3.addWidget(self.copyCodeButton3)


        self.horizontalLayout.addLayout(self.verticalLayoutPlot3)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(PlotTab)

        QMetaObject.connectSlotsByName(PlotTab)
    # setupUi

    def retranslateUi(self, PlotTab):
        PlotTab.setWindowTitle(QCoreApplication.translate("PlotTab", u"Plot", None))
        self.copyCodeButton1.setText(QCoreApplication.translate("PlotTab", u"Copy Code", None))
        self.copyCodeButton2.setText(QCoreApplication.translate("PlotTab", u"Copy Code", None))
        self.copyCodeButton3.setText(QCoreApplication.translate("PlotTab", u"Copy Code", None))
    # retranslateUi

