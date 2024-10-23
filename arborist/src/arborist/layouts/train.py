# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train.ui'
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

class Ui_TrainTab(object):
    def setupUi(self, TrainTab):
        if not TrainTab.objectName():
            TrainTab.setObjectName(u"TrainTab")
        TrainTab.resize(1600, 900)
        TrainTab.setBaseSize(QSize(1600, 900))
        self.verticalLayout = QVBoxLayout(TrainTab)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.frame_3 = QFrame(TrainTab)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setBaseSize(QSize(0, 0))
        self.frame_3.setStyleSheet(u"background-color: none;")
        self.frame_3.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setSpacing(20)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.frame_5 = QFrame(self.frame_3)
        self.frame_5.setObjectName(u"frame_5")
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setStyleSheet(u"background-color: #DDDDDD;")
        self.frame_5.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_5.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.frame_5)
        self.horizontalLayout_5.setSpacing(10)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(5, 5, 5, 5)
        self.modelLabel = QLabel(self.frame_5)
        self.modelLabel.setObjectName(u"modelLabel")
        sizePolicy.setHeightForWidth(self.modelLabel.sizePolicy().hasHeightForWidth())
        self.modelLabel.setSizePolicy(sizePolicy)

        self.horizontalLayout_5.addWidget(self.modelLabel)

        self.modelComboBox = QComboBox(self.frame_5)
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.setObjectName(u"modelComboBox")
        sizePolicy.setHeightForWidth(self.modelComboBox.sizePolicy().hasHeightForWidth())
        self.modelComboBox.setSizePolicy(sizePolicy)
        self.modelComboBox.setFrame(True)

        self.horizontalLayout_5.addWidget(self.modelComboBox)


        self.horizontalLayout_3.addWidget(self.frame_5)

        self.frame_2 = QFrame(self.frame_3)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet(u"background-color: #DDDDDD;")
        self.frame_2.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.label = QLabel(self.frame_2)
        self.label.setObjectName(u"label")
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.label)

        self.outcomeComboBox = QComboBox(self.frame_2)
        self.outcomeComboBox.setObjectName(u"outcomeComboBox")
        sizePolicy.setHeightForWidth(self.outcomeComboBox.sizePolicy().hasHeightForWidth())
        self.outcomeComboBox.setSizePolicy(sizePolicy)

        self.horizontalLayout_2.addWidget(self.outcomeComboBox)


        self.horizontalLayout_3.addWidget(self.frame_2)

        self.treatmentFrame = QFrame(self.frame_3)
        self.treatmentFrame.setObjectName(u"treatmentFrame")
        sizePolicy.setHeightForWidth(self.treatmentFrame.sizePolicy().hasHeightForWidth())
        self.treatmentFrame.setSizePolicy(sizePolicy)
        self.treatmentFrame.setStyleSheet(u"background-color: #DDDDDD;")
        self.treatmentFrame.setFrameShape(QFrame.Shape.NoFrame)
        self.treatmentFrame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_9 = QHBoxLayout(self.treatmentFrame)
        self.horizontalLayout_9.setSpacing(10)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(5, 5, 5, 5)
        self.treatmentLabel = QLabel(self.treatmentFrame)
        self.treatmentLabel.setObjectName(u"treatmentLabel")
        sizePolicy.setHeightForWidth(self.treatmentLabel.sizePolicy().hasHeightForWidth())
        self.treatmentLabel.setSizePolicy(sizePolicy)

        self.horizontalLayout_9.addWidget(self.treatmentLabel)

        self.treatmentComboBox = QComboBox(self.treatmentFrame)
        self.treatmentComboBox.setObjectName(u"treatmentComboBox")
        sizePolicy.setHeightForWidth(self.treatmentComboBox.sizePolicy().hasHeightForWidth())
        self.treatmentComboBox.setSizePolicy(sizePolicy)

        self.horizontalLayout_9.addWidget(self.treatmentComboBox)


        self.horizontalLayout_3.addWidget(self.treatmentFrame)

        self.trainButton = QPushButton(self.frame_3)
        self.trainButton.setObjectName(u"trainButton")
        sizePolicy.setHeightForWidth(self.trainButton.sizePolicy().hasHeightForWidth())
        self.trainButton.setSizePolicy(sizePolicy)

        self.horizontalLayout_3.addWidget(self.trainButton)

        self.frame_7 = QFrame(self.frame_3)
        self.frame_7.setObjectName(u"frame_7")
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setStyleSheet(u"background-color: #DDDDDD;")
        self.frame_7.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_7.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_7)
        self.horizontalLayout_8.setSpacing(10)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(5, 5, 5, 5)
        self.trainingTimeLabel = QLabel(self.frame_7)
        self.trainingTimeLabel.setObjectName(u"trainingTimeLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.trainingTimeLabel.sizePolicy().hasHeightForWidth())
        self.trainingTimeLabel.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.trainingTimeLabel)

        self.trainingTimeValue = QLabel(self.frame_7)
        self.trainingTimeValue.setObjectName(u"trainingTimeValue")
        sizePolicy1.setHeightForWidth(self.trainingTimeValue.sizePolicy().hasHeightForWidth())
        self.trainingTimeValue.setSizePolicy(sizePolicy1)

        self.horizontalLayout_8.addWidget(self.trainingTimeValue)


        self.horizontalLayout_3.addWidget(self.frame_7)


        self.verticalLayout.addWidget(self.frame_3)

        self.no_dataset_label = QLabel(TrainTab)
        self.no_dataset_label.setObjectName(u"no_dataset_label")
        self.no_dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.no_dataset_label)

        self.analytics_viewer = QTableView(TrainTab)
        self.analytics_viewer.setObjectName(u"analytics_viewer")
        self.analytics_viewer.setSortingEnabled(True)

        self.verticalLayout.addWidget(self.analytics_viewer)


        self.retranslateUi(TrainTab)

        QMetaObject.connectSlotsByName(TrainTab)
    # setupUi

    def retranslateUi(self, TrainTab):
        self.modelLabel.setText(QCoreApplication.translate("TrainTab", u"Model:", None))
        self.modelComboBox.setItemText(0, QCoreApplication.translate("TrainTab", u"BART", None))
        self.modelComboBox.setItemText(1, QCoreApplication.translate("TrainTab", u"XBART", None))
        self.modelComboBox.setItemText(2, QCoreApplication.translate("TrainTab", u"BCF", None))
        self.modelComboBox.setItemText(3, QCoreApplication.translate("TrainTab", u"XBCF", None))

        self.label.setText(QCoreApplication.translate("TrainTab", u"Outcome Variable (y):", None))
        self.treatmentLabel.setText(QCoreApplication.translate("TrainTab", u"Treatment Variable (Z):", None))
        self.trainButton.setText(QCoreApplication.translate("TrainTab", u"Train Model", None))
        self.trainingTimeLabel.setText(QCoreApplication.translate("TrainTab", u"Training time:", None))
        self.trainingTimeValue.setText(QCoreApplication.translate("TrainTab", u"0 seconds", None))
        self.no_dataset_label.setText(QCoreApplication.translate("TrainTab", u"Please select a dataset from the \"Browse\" tab", None))
        pass
    # retranslateUi

