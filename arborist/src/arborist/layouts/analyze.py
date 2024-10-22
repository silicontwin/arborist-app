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
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_5 = QFrame(self.frame_3)
        self.frame_5.setObjectName(u"frame_5")
        self.frame_5.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setSpacing(10)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.modelLabel = QLabel(self.frame_5)
        self.modelLabel.setObjectName(u"modelLabel")

        self.horizontalLayout_5.addWidget(self.modelLabel)

        self.modelComboBox = QComboBox(self.frame_5)
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.setObjectName(u"modelComboBox")

        self.horizontalLayout_5.addWidget(self.modelComboBox)


        self.horizontalLayout_3.addWidget(self.frame_5)

        self.frame = QFrame(self.frame_3)
        self.frame.setObjectName(u"frame")
        sizePolicy1.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy1)
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

        self.treatmentFrame = QFrame(self.frame)
        self.treatmentFrame.setObjectName(u"treatmentFrame")
        sizePolicy1.setHeightForWidth(self.treatmentFrame.sizePolicy().hasHeightForWidth())
        self.treatmentFrame.setSizePolicy(sizePolicy1)
        self.treatmentFrame.setFrameShape(QFrame.Shape.StyledPanel)
        self.treatmentFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.treatmentFrame)
        self.horizontalLayout_9.setSpacing(10)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.treatmentLabel = QLabel(self.treatmentFrame)
        self.treatmentLabel.setObjectName(u"treatmentLabel")

        self.horizontalLayout_9.addWidget(self.treatmentLabel)

        self.treatmentComboBox = QComboBox(self.treatmentFrame)
        self.treatmentComboBox.setObjectName(u"treatmentComboBox")

        self.horizontalLayout_9.addWidget(self.treatmentComboBox)


        self.horizontalLayout.addWidget(self.treatmentFrame)


        self.horizontalLayout_3.addWidget(self.frame)

        self.frame_9 = QFrame(self.frame_3)
        self.frame_9.setObjectName(u"frame_9")
        sizePolicy1.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy1)
        self.frame_9.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_9.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_9)
        self.horizontalLayout_6.setSpacing(10)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.frame_6 = QFrame(self.frame_9)
        self.frame_6.setObjectName(u"frame_6")
        sizePolicy1.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy1)
        self.frame_6.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.trainButton = QPushButton(self.frame_6)
        self.trainButton.setObjectName(u"trainButton")
        sizePolicy1.setHeightForWidth(self.trainButton.sizePolicy().hasHeightForWidth())
        self.trainButton.setSizePolicy(sizePolicy1)

        self.horizontalLayout_7.addWidget(self.trainButton)

        self.frame_7 = QFrame(self.frame_6)
        self.frame_7.setObjectName(u"frame_7")
        sizePolicy1.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy1)
        self.frame_7.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_8.setSpacing(10)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.trainingTimeLabel = QLabel(self.frame_7)
        self.trainingTimeLabel.setObjectName(u"trainingTimeLabel")
        sizePolicy.setHeightForWidth(self.trainingTimeLabel.sizePolicy().hasHeightForWidth())
        self.trainingTimeLabel.setSizePolicy(sizePolicy)

        self.horizontalLayout_8.addWidget(self.trainingTimeLabel)

        self.trainingTimeValue = QLabel(self.frame_7)
        self.trainingTimeValue.setObjectName(u"trainingTimeValue")
        sizePolicy.setHeightForWidth(self.trainingTimeValue.sizePolicy().hasHeightForWidth())
        self.trainingTimeValue.setSizePolicy(sizePolicy)

        self.horizontalLayout_8.addWidget(self.trainingTimeValue)


        self.horizontalLayout_7.addWidget(self.frame_7)


        self.horizontalLayout_6.addWidget(self.frame_6)


        self.horizontalLayout_3.addWidget(self.frame_9)

        self.frame_4 = QFrame(self.frame_3)
        self.frame_4.setObjectName(u"frame_4")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy2)
        self.frame_4.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)

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
        self.modelLabel.setText(QCoreApplication.translate("AnalyzeTab", u"Model:", None))
        self.modelComboBox.setItemText(0, QCoreApplication.translate("AnalyzeTab", u"BART", None))
        self.modelComboBox.setItemText(1, QCoreApplication.translate("AnalyzeTab", u"XBART", None))
        self.modelComboBox.setItemText(2, QCoreApplication.translate("AnalyzeTab", u"BCF", None))
        self.modelComboBox.setItemText(3, QCoreApplication.translate("AnalyzeTab", u"XBCF", None))

        self.label.setText(QCoreApplication.translate("AnalyzeTab", u"Outcome Variable (y):", None))
        self.treatmentLabel.setText(QCoreApplication.translate("AnalyzeTab", u"Treatment Variable (Z):", None))
        self.trainButton.setText(QCoreApplication.translate("AnalyzeTab", u"Train Model", None))
        self.trainingTimeLabel.setText(QCoreApplication.translate("AnalyzeTab", u"Training time:", None))
        self.trainingTimeValue.setText(QCoreApplication.translate("AnalyzeTab", u"0 seconds", None))
        self.no_dataset_label.setText(QCoreApplication.translate("AnalyzeTab", u"Please select a dataset from the \"Browse\" tab", None))
        pass
    # retranslateUi

