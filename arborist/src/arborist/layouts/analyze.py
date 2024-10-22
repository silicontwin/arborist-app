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
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QSizePolicy,
    QTableView, QVBoxLayout, QWidget)

class Ui_AnalyzeTab(object):
    def setupUi(self, AnalyzeTab):
        if not AnalyzeTab.objectName():
            AnalyzeTab.setObjectName(u"AnalyzeTab")
        AnalyzeTab.resize(1600, 900)
        AnalyzeTab.setBaseSize(QSize(1600, 900))
        self.verticalLayout = QVBoxLayout(AnalyzeTab)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.frame_2 = QFrame(AnalyzeTab)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout.addWidget(self.frame_2)

        self.frame_3 = QFrame(AnalyzeTab)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy1)
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setSpacing(10)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self.frame_3)
        self.frame.setObjectName(u"frame")
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet(u"")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.outcomeComboBox = QComboBox(self.frame)
        self.outcomeComboBox.setObjectName(u"outcomeComboBox")

        self.horizontalLayout.addWidget(self.outcomeComboBox)

        self.treatmentLabel = QLabel(self.frame)
        self.treatmentLabel.setObjectName(u"treatmentLabel")

        self.horizontalLayout.addWidget(self.treatmentLabel)

        self.treatmentComboBox = QComboBox(self.frame)
        self.treatmentComboBox.setObjectName(u"treatmentComboBox")

        self.horizontalLayout.addWidget(self.treatmentComboBox)


        self.horizontalLayout_3.addWidget(self.frame)

        self.trainButton = QPushButton(self.frame_3)
        self.trainButton.setObjectName(u"trainButton")
        sizePolicy1.setHeightForWidth(self.trainButton.sizePolicy().hasHeightForWidth())
        self.trainButton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_3.addWidget(self.trainButton)

        self.frame_4 = QFrame(self.frame_3)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.trainingTimeLabel = QLabel(self.frame_4)
        self.trainingTimeLabel.setObjectName(u"trainingTimeLabel")
        sizePolicy.setHeightForWidth(self.trainingTimeLabel.sizePolicy().hasHeightForWidth())
        self.trainingTimeLabel.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.trainingTimeLabel)

        self.trainingTimeValue = QLabel(self.frame_4)
        self.trainingTimeValue.setObjectName(u"trainingTimeValue")
        sizePolicy.setHeightForWidth(self.trainingTimeValue.sizePolicy().hasHeightForWidth())
        self.trainingTimeValue.setSizePolicy(sizePolicy)

        self.horizontalLayout_4.addWidget(self.trainingTimeValue)


        self.horizontalLayout_3.addWidget(self.frame_4)


        self.verticalLayout.addWidget(self.frame_3)

        self.no_dataset_label = QLabel(AnalyzeTab)
        self.no_dataset_label.setObjectName(u"no_dataset_label")
        self.no_dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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
        self.label.setText(QCoreApplication.translate("AnalyzeTab", u"Outcome Variable (y):", None))
        self.treatmentLabel.setText(QCoreApplication.translate("AnalyzeTab", u"Treatment Variable (Z):", None))
        self.trainButton.setText(QCoreApplication.translate("AnalyzeTab", u"Train Model", None))
        self.trainingTimeLabel.setText(QCoreApplication.translate("AnalyzeTab", u"Training time:", None))
        self.trainingTimeValue.setText(QCoreApplication.translate("AnalyzeTab", u"0 seconds", None))
        self.no_dataset_label.setText(QCoreApplication.translate("AnalyzeTab", u"Please select a dataset from the \"Browse\" tab", None))
        pass
    # retranslateUi

