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
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QTableView,
    QVBoxLayout, QWidget)

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
        self.frame = QFrame(TrainTab)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.frame_3 = QFrame(self.frame)
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
        self.modelComboBox.setMinimumSize(QSize(130, 0))
        self.modelComboBox.setBaseSize(QSize(0, 0))
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
        self.trainButton.setMinimumSize(QSize(100, 30))
        self.trainButton.setStyleSheet(u"background-color: #FF0000;\n"
"color: #FFFFFF;\n"
"border-radius: 4px;")

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


        self.horizontalLayout.addWidget(self.frame_3)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.frame_4 = QFrame(self.frame)
        self.frame_4.setObjectName(u"frame_4")
        self.frame_4.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_4.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.frame_4)
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.codeGenPushButton = QPushButton(self.frame_4)
        self.codeGenPushButton.setObjectName(u"codeGenPushButton")
        self.codeGenPushButton.setMinimumSize(QSize(80, 30))
        self.codeGenPushButton.setStyleSheet(u"background-color: #DDDDDD;\n"
"border-radius: 4px;")

        self.horizontalLayout_4.addWidget(self.codeGenPushButton)

        self.parametersPushButton = QPushButton(self.frame_4)
        self.parametersPushButton.setObjectName(u"parametersPushButton")
        sizePolicy.setHeightForWidth(self.parametersPushButton.sizePolicy().hasHeightForWidth())
        self.parametersPushButton.setSizePolicy(sizePolicy)
        self.parametersPushButton.setMinimumSize(QSize(100, 30))
        self.parametersPushButton.setStyleSheet(u"background-color: #DDDDDD;\n"
"border-radius: 4px;")

        self.horizontalLayout_4.addWidget(self.parametersPushButton)


        self.horizontalLayout.addWidget(self.frame_4)


        self.verticalLayout.addWidget(self.frame)

        self.no_dataset_label = QLabel(TrainTab)
        self.no_dataset_label.setObjectName(u"no_dataset_label")
        self.no_dataset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.no_dataset_label)

        self.frame_6 = QFrame(TrainTab)
        self.frame_6.setObjectName(u"frame_6")
        self.frame_6.setFrameShape(QFrame.Shape.NoFrame)
        self.frame_6.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_6)
        self.horizontalLayout_6.setSpacing(10)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.analytics_viewer = QTableView(self.frame_6)
        self.analytics_viewer.setObjectName(u"analytics_viewer")
        self.analytics_viewer.setSortingEnabled(True)

        self.horizontalLayout_6.addWidget(self.analytics_viewer)

        self.parametersMenu = QFrame(self.frame_6)
        self.parametersMenu.setObjectName(u"parametersMenu")
        self.parametersMenu.setMinimumSize(QSize(300, 0))
        self.parametersMenu.setVisible(False)
        self.parametersMenu.setStyleSheet(u"background-color: #FFFFFF;")
        self.verticalLayout_Parameters = QVBoxLayout(self.parametersMenu)
        self.verticalLayout_Parameters.setSpacing(10)
        self.verticalLayout_Parameters.setObjectName(u"verticalLayout_Parameters")
        self.verticalLayout_Parameters.setContentsMargins(10, 10, 10, 10)
        self.parametersTitleLabel = QLabel(self.parametersMenu)
        self.parametersTitleLabel.setObjectName(u"parametersTitleLabel")
        self.parametersTitleLabel.setStyleSheet(u"font-weight: bold; border-bottom: 1px solid lightgray; padding-bottom: 5px;")
        self.parametersTitleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_Parameters.addWidget(self.parametersTitleLabel)

        self.horizontalLayout_Trees = QHBoxLayout()
        self.horizontalLayout_Trees.setObjectName(u"horizontalLayout_Trees")
        self.treesLabel = QLabel(self.parametersMenu)
        self.treesLabel.setObjectName(u"treesLabel")

        self.horizontalLayout_Trees.addWidget(self.treesLabel)

        self.treesSpinBox = QSpinBox(self.parametersMenu)
        self.treesSpinBox.setObjectName(u"treesSpinBox")
        self.treesSpinBox.setMinimum(1)
        self.treesSpinBox.setMaximum(1000)
        self.treesSpinBox.setValue(100)

        self.horizontalLayout_Trees.addWidget(self.treesSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Trees)

        self.horizontalLayout_BurnIn = QHBoxLayout()
        self.horizontalLayout_BurnIn.setObjectName(u"horizontalLayout_BurnIn")
        self.burnInLabel = QLabel(self.parametersMenu)
        self.burnInLabel.setObjectName(u"burnInLabel")

        self.horizontalLayout_BurnIn.addWidget(self.burnInLabel)

        self.burnInSpinBox = QSpinBox(self.parametersMenu)
        self.burnInSpinBox.setObjectName(u"burnInSpinBox")
        self.burnInSpinBox.setMinimum(1)
        self.burnInSpinBox.setMaximum(10000)
        self.burnInSpinBox.setValue(1000)

        self.horizontalLayout_BurnIn.addWidget(self.burnInSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_BurnIn)

        self.horizontalLayout_Draws = QHBoxLayout()
        self.horizontalLayout_Draws.setObjectName(u"horizontalLayout_Draws")
        self.drawsLabel = QLabel(self.parametersMenu)
        self.drawsLabel.setObjectName(u"drawsLabel")

        self.horizontalLayout_Draws.addWidget(self.drawsLabel)

        self.drawsSpinBox = QSpinBox(self.parametersMenu)
        self.drawsSpinBox.setObjectName(u"drawsSpinBox")
        self.drawsSpinBox.setMinimum(1)
        self.drawsSpinBox.setMaximum(100000)
        self.drawsSpinBox.setValue(5000)

        self.horizontalLayout_Draws.addWidget(self.drawsSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Draws)

        self.horizontalLayout_Thinning = QHBoxLayout()
        self.horizontalLayout_Thinning.setObjectName(u"horizontalLayout_Thinning")
        self.thinningLabel = QLabel(self.parametersMenu)
        self.thinningLabel.setObjectName(u"thinningLabel")

        self.horizontalLayout_Thinning.addWidget(self.thinningLabel)

        self.thinningSpinBox = QSpinBox(self.parametersMenu)
        self.thinningSpinBox.setObjectName(u"thinningSpinBox")
        self.thinningSpinBox.setMinimum(1)
        self.thinningSpinBox.setMaximum(100)
        self.thinningSpinBox.setValue(1)

        self.horizontalLayout_Thinning.addWidget(self.thinningSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Thinning)

        self.horizontalLayout_PriorMean = QHBoxLayout()
        self.horizontalLayout_PriorMean.setObjectName(u"horizontalLayout_PriorMean")
        self.priorMeanLabel = QLabel(self.parametersMenu)
        self.priorMeanLabel.setObjectName(u"priorMeanLabel")

        self.horizontalLayout_PriorMean.addWidget(self.priorMeanLabel)

        self.priorMeanSpinBox = QDoubleSpinBox(self.parametersMenu)
        self.priorMeanSpinBox.setObjectName(u"priorMeanSpinBox")
        self.priorMeanSpinBox.setMinimum(0.000000000000000)
        self.priorMeanSpinBox.setMaximum(0.000000000000000)
        self.priorMeanSpinBox.setValue(0.000000000000000)

        self.horizontalLayout_PriorMean.addWidget(self.priorMeanSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_PriorMean)

        self.horizontalLayout_PriorVariance = QHBoxLayout()
        self.horizontalLayout_PriorVariance.setObjectName(u"horizontalLayout_PriorVariance")
        self.priorVarianceLabel = QLabel(self.parametersMenu)
        self.priorVarianceLabel.setObjectName(u"priorVarianceLabel")

        self.horizontalLayout_PriorVariance.addWidget(self.priorVarianceLabel)

        self.priorVarianceSpinBox = QDoubleSpinBox(self.parametersMenu)
        self.priorVarianceSpinBox.setObjectName(u"priorVarianceSpinBox")
        self.priorVarianceSpinBox.setMinimum(0.000000000000000)
        self.priorVarianceSpinBox.setMaximum(0.000000000000000)
        self.priorVarianceSpinBox.setValue(0.000000000000000)

        self.horizontalLayout_PriorVariance.addWidget(self.priorVarianceSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_PriorVariance)

        self.horizontalLayout_Alpha = QHBoxLayout()
        self.horizontalLayout_Alpha.setObjectName(u"horizontalLayout_Alpha")
        self.alphaLabel = QLabel(self.parametersMenu)
        self.alphaLabel.setObjectName(u"alphaLabel")

        self.horizontalLayout_Alpha.addWidget(self.alphaLabel)

        self.alphaSpinBox = QDoubleSpinBox(self.parametersMenu)
        self.alphaSpinBox.setObjectName(u"alphaSpinBox")
        self.alphaSpinBox.setMinimum(0.000000000000000)
        self.alphaSpinBox.setMaximum(0.000000000000000)
        self.alphaSpinBox.setValue(0.000000000000000)

        self.horizontalLayout_Alpha.addWidget(self.alphaSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Alpha)

        self.horizontalLayout_Beta = QHBoxLayout()
        self.horizontalLayout_Beta.setObjectName(u"horizontalLayout_Beta")
        self.betaLabel = QLabel(self.parametersMenu)
        self.betaLabel.setObjectName(u"betaLabel")

        self.horizontalLayout_Beta.addWidget(self.betaLabel)

        self.betaSpinBox = QSpinBox(self.parametersMenu)
        self.betaSpinBox.setObjectName(u"betaSpinBox")
        self.betaSpinBox.setMinimum(1)
        self.betaSpinBox.setMaximum(10)
        self.betaSpinBox.setValue(2)

        self.horizontalLayout_Beta.addWidget(self.betaSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Beta)

        self.horizontalLayout_Depth = QHBoxLayout()
        self.horizontalLayout_Depth.setObjectName(u"horizontalLayout_Depth")
        self.treeDepthLabel = QLabel(self.parametersMenu)
        self.treeDepthLabel.setObjectName(u"treeDepthLabel")

        self.horizontalLayout_Depth.addWidget(self.treeDepthLabel)

        self.treeDepthSpinBox = QSpinBox(self.parametersMenu)
        self.treeDepthSpinBox.setObjectName(u"treeDepthSpinBox")
        self.treeDepthSpinBox.setMinimum(1)
        self.treeDepthSpinBox.setMaximum(10)
        self.treeDepthSpinBox.setValue(3)

        self.horizontalLayout_Depth.addWidget(self.treeDepthSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_Depth)

        self.horizontalLayout_NodeSize = QHBoxLayout()
        self.horizontalLayout_NodeSize.setObjectName(u"horizontalLayout_NodeSize")
        self.nodeSizeLabel = QLabel(self.parametersMenu)
        self.nodeSizeLabel.setObjectName(u"nodeSizeLabel")

        self.horizontalLayout_NodeSize.addWidget(self.nodeSizeLabel)

        self.nodeSizeSpinBox = QSpinBox(self.parametersMenu)
        self.nodeSizeSpinBox.setObjectName(u"nodeSizeSpinBox")
        self.nodeSizeSpinBox.setMinimum(1)
        self.nodeSizeSpinBox.setMaximum(100)
        self.nodeSizeSpinBox.setValue(5)

        self.horizontalLayout_NodeSize.addWidget(self.nodeSizeSpinBox)


        self.verticalLayout_Parameters.addLayout(self.horizontalLayout_NodeSize)


        self.horizontalLayout_6.addWidget(self.parametersMenu)


        self.verticalLayout.addWidget(self.frame_6)


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
        self.codeGenPushButton.setText(QCoreApplication.translate("TrainTab", u"Code Gen", None))
        self.parametersPushButton.setText(QCoreApplication.translate("TrainTab", u"Parameters", None))
        self.no_dataset_label.setText(QCoreApplication.translate("TrainTab", u"Please select a dataset from the \"Browse\" tab", None))
        self.parametersTitleLabel.setText(QCoreApplication.translate("TrainTab", u"Model Parameters", None))
        self.treesLabel.setText(QCoreApplication.translate("TrainTab", u"Number of Trees (M)", None))
        self.burnInLabel.setText(QCoreApplication.translate("TrainTab", u"Burn-in Iterations", None))
        self.drawsLabel.setText(QCoreApplication.translate("TrainTab", u"Number of Draws", None))
        self.thinningLabel.setText(QCoreApplication.translate("TrainTab", u"Thinning", None))
        self.priorMeanLabel.setText(QCoreApplication.translate("TrainTab", u"Prior Mean of Leaf Parameters", None))
        self.priorVarianceLabel.setText(QCoreApplication.translate("TrainTab", u"Prior Variance of Leaf Parameters", None))
        self.alphaLabel.setText(QCoreApplication.translate("TrainTab", u"Alpha (\u03b1) for Tree Structure Prior", None))
        self.betaLabel.setText(QCoreApplication.translate("TrainTab", u"Beta (\u03b2) for Tree Structure Prior", None))
        self.treeDepthLabel.setText(QCoreApplication.translate("TrainTab", u"Tree Depth (D)", None))
        self.nodeSizeLabel.setText(QCoreApplication.translate("TrainTab", u"Minimum Node Size", None))
        pass
    # retranslateUi

