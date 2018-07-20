import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import pyqtgraph as pg
import numpy as np
import random
from pubsub import pub
from colors import distinct_colors


class FilePicker(QtGui.QWidget):
    def __init__(self, parent):
        super(FilePicker, self).__init__(parent)

        horizontalLayout = QtGui.QHBoxLayout(self)
        horizontalLayout.setContentsMargins(0, 0, 0, 0)

        label = QtGui.QLabel(self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        label.setSizePolicy(sizePolicy)
        label.setText('File path')
        horizontalLayout.addWidget(label)

        self.filePathEdit = QtGui.QLineEdit(self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        self.filePathEdit.setSizePolicy(sizePolicy)
        horizontalLayout.addWidget(self.filePathEdit)

        button = QtGui.QPushButton(self)
        button.setText('...')
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        button.setSizePolicy(sizePolicy)
        horizontalLayout.addWidget(button)

        button.clicked.connect(self.selectFile)

        self.setLayout(horizontalLayout)

    def selectFile(self):
        fileDialog = QtGui.QFileDialog(self)

        if not fileDialog.exec():
            return

        selectedFiles = fileDialog.selectedFiles()
        self.filePathEdit.setText(selectedFiles[0])
        print(selectedFiles)


class PointSelectedEvent:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class ImageWidget(QtGui.QWidget):
    pointSelected = QtCore.pyqtSignal(PointSelectedEvent)

    def __init__(self, parent, fullColor):
        super(ImageWidget, self).__init__(parent)

        self.imageView = pg.ImageView()

        if fullColor:
            data = np.random.randint(0, 255, (200, 200, 3))
            data = np.random.normal(size=(200, 200))
        else:
            dataIndicies = np.random.choice(distinct_colors.shape[0], (200, 200))
            data = distinct_colors[dataIndicies, :]

        self.image = data
        self.imageView.ui.roiBtn.hide()
        self.imageView.ui.menuBtn.hide()
        self.imageView.setImage(self.image)
        self.imageView.view.setLimits(
            xMin=-10,
            xMax=self.image.shape[0] + 10,
            yMin=-10,
            yMax=self.image.shape[1] + 10
        )

        self.blink = 0
        self.pen = QtGui.QPen(QtGui.QColor(0xff, 0x00, 0x00))
        self.pen.setWidthF(0.25)
        self.roi = pg.CircleROI([30.5, 40.5], [2, 2], movable=False, pen=self.pen)
        self.imageView.addItem(self.roi)
        self.roi.removeHandle(0)

        verticalLayout = QtGui.QVBoxLayout()
        verticalLayout.addWidget(self.imageView)
        self.setLayout(verticalLayout)

        self.imageView.scene.sigMouseClicked.connect(self.onClick)

    def onBlink(self):
        self.blink = self.blink + 25
        if self.blink >= 2 * 256:
            self.blink = 0

        if self.blink < 256:
            self.pen.setColor(QtGui.QColor(self.blink, self.blink, self.blink))
        else:
            self.pen.setColor(QtGui.QColor(512 - self.blink, 512 - self.blink, 512 - self.blink))
        self.roi.setPen(self.pen)
        self.imageView.view.update()

    def onPointSelected(self, event):
        self.roi.setPos(event.x - 0.5, event.y - 0.5)

    def onClick(self, event):
        if event.buttons() != QtCore.Qt.LeftButton:
            return
        items = self.imageView.view.mapSceneToView(event.scenePos())
        self.pointSelected.emit(PointSelectedEvent(int(items.x()), int(items.y())))
        self.imageView.view.update()


class DataToolbox(QtGui.QWidget):
    def __init__(self, parent):
        super(DataToolbox, self).__init__(parent)

        self.setWindowTitle('Data Toolbox')

        self.gridLayout = QtGui.QGridLayout()

        self.filePicker = FilePicker(self)
        self.gridLayout.addWidget(self.filePicker, 0, 0, 1, 3)

        self.methodComboBox = QtGui.QComboBox(self)
        self.methodComboBox.addItem('Band - Channel mapping')
        self.gridLayout.addWidget(self.methodComboBox, 1, 0, 1, 3)

        self.redLabel = QtGui.QLabel(self)
        self.redLabel.setText('Red')
        self.gridLayout.addWidget(self.redLabel, 2, 0, 1, 1)

        self.greenLabel = QtGui.QLabel(self)
        self.greenLabel.setText('Green')
        self.gridLayout.addWidget(self.greenLabel, 3, 0, 1, 1)

        self.blueLabel = QtGui.QLabel(self)
        self.blueLabel.setText('Blue')
        self.gridLayout.addWidget(self.blueLabel, 4, 0, 1, 1)

        self.redComboBox = QtGui.QComboBox(self)
        self.redComboBox.addItem('401 nm')
        self.gridLayout.addWidget(self.redComboBox, 2, 1, 1, 2)

        self.greenComboBox = QtGui.QComboBox(self)
        self.greenComboBox.addItem('889 nm')
        self.gridLayout.addWidget(self.greenComboBox, 3, 1, 1, 2)

        self.blueComboBox = QtGui.QComboBox(self)
        self.blueComboBox.addItem('573 nm')
        self.gridLayout.addWidget(self.blueComboBox, 4, 1, 1, 2)

        spacerItem = QtGui.QSpacerItem(
            20,
            40,
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem, 5, 0, 1, 3)

        self.setLayout(self.gridLayout)


class LearningToolbox(QtGui.QWidget):
    def __init__(self, parent):
        super(LearningToolbox, self).__init__(parent)

        self.gridLayout = QtGui.QGridLayout()

        self.filePicker = FilePicker(self)
        self.gridLayout.addWidget(self.filePicker, 0, 0, 1, 3)

        self.legendLabel = QtGui.QLabel(self)
        self.legendLabel.setText('Legend')

        self.gridLayout.addWidget(self.legendLabel, 1, 0, 1, 1)

        scrollArea = QtGui.QScrollArea(self)
        self.frame = QtGui.QFrame(self)

        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        scrollArea.setWidget(self.frame)
        scrollArea.setWidgetResizable(True)
        self.frame.setLayout(self.verticalLayout)

        self.gridLayout.addWidget(scrollArea, 2, 0, 1, 1)

        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])

        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])
        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])
        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])
        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])

        # self.gridLayout.addLayout(self.verticalLayout, 2, 0, 1, 1)

        spacerItem = QtGui.QSpacerItem(
            20,
            40,
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)

        self.learnButton = QtGui.QPushButton(self)
        self.learnButton.setText('Start Learning')
        self.gridLayout.addWidget(self.learnButton, 4, 0, 1, 1)

        self.setLayout(self.gridLayout)

    def addLayer(self, name, color):
        horizontalLayout = QtGui.QHBoxLayout()
        colorButton = QtGui.QPushButton(self.frame)
        colorButton.clicked.connect(self.changeColor)

        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill()
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(pixmap.rect(), QtGui.QColor(color[0], color[1], color[2]))
        painter.end()

        colorButton.setIcon(QtGui.QIcon(pixmap))
        colorButton.setIconSize(QtCore.QSize(16, 16))

        horizontalLayout.addWidget(colorButton)
        nameEdit = QtGui.QLineEdit(self.frame)
        nameEdit.setText(name)
        horizontalLayout.addWidget(nameEdit)

        self.verticalLayout.addLayout(horizontalLayout)

    def changeColor(self, event):
        QtGui.QColorDialog.getColor()


class AnalysisToolbox(QtGui.QWidget):
    def __init__(self, parent):
        super(AnalysisToolbox, self).__init__(parent)

        self.gridLayout = QtGui.QGridLayout()

        self.errorLabel = QtGui.QLabel(self)
        self.errorLabel.setText('Error')
        self.gridLayout.addWidget(self.errorLabel, 0, 0, 1, 1)
        self.errorEdit = QtGui.QLineEdit(self)
        self.errorEdit.setText('0.03%')
        self.errorEdit.setDisabled(True)
        self.gridLayout.addWidget(self.errorEdit, 0, 1, 1, 1)

        self.legendLabel = QtGui.QLabel(self)
        self.legendLabel.setText('Class probability')
        self.gridLayout.addWidget(self.legendLabel, 1, 0, 1, 2)

        self.verticalLayout = QtGui.QVBoxLayout()

        self.addLayer('Asphalt', distinct_colors[0])
        self.addLayer('Meadows', distinct_colors[1])
        self.addLayer('Gravel', distinct_colors[2])
        self.addLayer('Trees', distinct_colors[3])
        self.addLayer('Painted metal sheets', distinct_colors[4])
        self.addLayer('Bare Soil', distinct_colors[5])
        self.addLayer('Bitumen', distinct_colors[6])
        self.addLayer('Self-Blocking Bricks', distinct_colors[7])
        self.addLayer('Shadows', distinct_colors[8])

        self.gridLayout.addLayout(self.verticalLayout, 2, 0, 1, 2)

        spacerItem = QtGui.QSpacerItem(
            20,
            40,
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding
        )
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 2)

        self.analyseButton = QtGui.QPushButton(self)
        self.analyseButton.setText('Start Analysis')
        self.gridLayout.addWidget(self.analyseButton, 4, 0, 1, 2)

        self.setLayout(self.gridLayout)

    def addLayer(self, name, color):
        horizontalLayout = QtGui.QHBoxLayout()
        colorButton = QtGui.QPushButton(self)

        pixmap = QtGui.QPixmap(32, 32)
        pixmap.fill()
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(pixmap.rect(), QtGui.QColor(color[0], color[1], color[2]))
        painter.end()

        colorButton.setIcon(QtGui.QIcon(pixmap))
        colorButton.setIconSize(QtCore.QSize(16, 16))
        horizontalLayout.addWidget(colorButton)

        nameEdit = QtGui.QLineEdit(self)
        nameEdit.setText(name)
        nameEdit.setDisabled(True)
        horizontalLayout.addWidget(nameEdit)

        probabilityEdit = QtGui.QLineEdit(self)
        probabilityEdit.setText(str(np.random.normal()))
        probabilityEdit.setDisabled(True)
        horizontalLayout.addWidget(probabilityEdit)

        self.verticalLayout.addLayout(horizontalLayout)


class ImageWithToolbox(QtGui.QWidget):
    def __init__(self, parent, image, toolbox):
        super(ImageWithToolbox, self).__init__(parent)

        verticalLayout = QtGui.QVBoxLayout(self)

        verticalLayout.addWidget(image)
        verticalLayout.addWidget(toolbox)

        self.setLayout(verticalLayout)


class dockdemo(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(dockdemo, self).__init__(parent)

        bar = self.menuBar()
        file = bar.addMenu("File")
        file.addAction("New")
        file.addAction("save")
        file.addAction("quit")

        self.setCentralWidget(None)

        dataImage = ImageWidget(self, True)
        dataToolbox = DataToolbox(self)
        dataDockWidget = QtGui.QDockWidget(self)
        dataDockWidget.setFeatures(
            QtGui.QDockWidget.DockWidgetMovable | QtGui.QDockWidget.DockWidgetFloatable
        )
        dataDockWidget.setWidget(ImageWithToolbox(self, dataImage, dataToolbox))
        dataDockWidget.setWindowTitle('Input')

        learningImage = ImageWidget(self, False)
        learningToolbox = LearningToolbox(self)
        learningDockWidget = QtGui.QDockWidget(self)
        learningDockWidget.setFeatures(
            QtGui.QDockWidget.DockWidgetMovable | QtGui.QDockWidget.DockWidgetFloatable
        )
        learningDockWidget.setWidget(ImageWithToolbox(self, learningImage, learningToolbox))
        learningDockWidget.setWindowTitle('Ground-truth')

        analysisImage = ImageWidget(self, False)
        analysisToolbox = AnalysisToolbox(self)
        analysisDockWidget = QtGui.QDockWidget(self)
        analysisDockWidget.setFeatures(
            QtGui.QDockWidget.DockWidgetMovable | QtGui.QDockWidget.DockWidgetFloatable
        )
        analysisDockWidget.setWidget(ImageWithToolbox(self, analysisImage, analysisToolbox))
        analysisDockWidget.setWindowTitle('Analysis')

        timer = QtCore.QTimer(self)
        timer.timeout.connect(dataImage.onBlink)
        timer.timeout.connect(learningImage.onBlink)
        timer.timeout.connect(analysisImage.onBlink)
        timer.start(50)

        dataImage.pointSelected.connect(dataImage.onPointSelected)
        dataImage.pointSelected.connect(learningImage.onPointSelected)
        dataImage.pointSelected.connect(analysisImage.onPointSelected)
        learningImage.pointSelected.connect(dataImage.onPointSelected)
        learningImage.pointSelected.connect(learningImage.onPointSelected)
        learningImage.pointSelected.connect(analysisImage.onPointSelected)
        analysisImage.pointSelected.connect(dataImage.onPointSelected)
        analysisImage.pointSelected.connect(learningImage.onPointSelected)
        analysisImage.pointSelected.connect(analysisImage.onPointSelected)

        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, dataDockWidget)
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, learningDockWidget)
        self.addDockWidget(QtCore.Qt.TopDockWidgetArea, analysisDockWidget)

        dataImage.imageView.view.setXLink(learningImage.imageView.view)
        dataImage.imageView.view.setYLink(learningImage.imageView.view)

        learningImage.imageView.view.setXLink(analysisImage.imageView.view)
        learningImage.imageView.view.setYLink(analysisImage.imageView.view)

        analysisImage.imageView.view.setXLink(dataImage.imageView.view)
        analysisImage.imageView.view.setYLink(dataImage.imageView.view)

        self.setWindowTitle('HyperNet')


def main():
    app = QtGui.QApplication(sys.argv)
    ex = dockdemo()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
