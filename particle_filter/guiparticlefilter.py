from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLabel, QVBoxLayout, QLineEdit
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QIntValidator
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import particle_filter.particlefilter as pfilter


class MyPlotCanvas(FigureCanvas):
    """Class for embedding Matplotlib features in PyQt.
       In this manner enabling to plot and display in Qt widget """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Particle filter simulator")
        self.setGeometry(10, 10, 1000, 600)
        # self.file_menu = QtWidgets.QMenu('&File', self)
        # self.file_menu.addAction('&Quit', self.fileQuit,
        #                          QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        # self.menuBar().addMenu(self.file_menu)

        # self.help_menu = QtWidgets.QMenu('&Help', self)
        # self.menuBar().addSeparator()
        # self.menuBar().addMenu(self.help_menu)

        # self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        self.base_hlayout = QHBoxLayout()
        self.vlayout_left = QVBoxLayout()
        self.vlayout_right = QVBoxLayout()

        self.button_startsim = QPushButton('Start Simulation')
        self.label_mparticle = QLabel('M number of particles')
        self.lined_mparticle = QLineEdit()
        self.lined_mparticle.setValidator(QIntValidator())
        self.label_odovar = QLabel('Odometry variances')
        self.lined_odovar = QLineEdit()
        self.label_landmarkvar = QLabel('Landmark variances')
        self.lined_landmarkvar = QLineEdit()

        self.button_startsim.setFixedSize(150, 30)
        self.label_mparticle.setFixedSize(150, 30)
        self.lined_mparticle.setFixedSize(150, 30)
        self.label_odovar.setFixedSize(150, 30)
        self.lined_odovar.setFixedSize(150, 30)
        self.label_landmarkvar.setFixedSize(150, 30)
        self.lined_landmarkvar.setFixedSize(150, 30)

        self.base_hlayout.addLayout(self.vlayout_left)
        self.base_hlayout.addLayout(self.vlayout_right)

        self.vlayout_left.addWidget(self.button_startsim)
        self.vlayout_left.addWidget(self.label_mparticle)
        self.vlayout_left.addWidget(self.lined_mparticle)
        self.vlayout_left.addWidget(self.label_odovar)
        self.vlayout_left.addWidget(self.lined_odovar)
        self.vlayout_left.addWidget(self.label_landmarkvar)
        self.vlayout_left.addWidget(self.lined_landmarkvar)
        self.vlayout_left.addStretch(1)

        self.plot_canvas = MyPlotCanvas(self.main_widget, width=7, height=4, dpi=100)

        self.vlayout_right.addWidget(self.plot_canvas)

        self.main_widget.setLayout(self.base_hlayout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        """self.x0 = np.array([0,0,0], dtype=float)
        self.m_particles = 0
        self.var_odom = np.array([0, 0, 0], dtype=float)
        self.var_landmark = np.array([0, 0, 0, 0], dtype=float)"""

        self.button_startsim.clicked.connect(self.buttonStartSimClick)

        print(f'bt startsim size x,y =  {self.button_startsim.size().width()} , {self.button_startsim.size().height()}')
        print(
            f' label m particle size x,y =  {self.label_mparticle.size().width()} , {self.label_mparticle.size().height()}')
        print(
            f'line edit_mparticle size x,y =  {self.lined_mparticle.size().width()} , {self.lined_mparticle.size().height()}')

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def buttonStartSimClick(self):

        pfilter.particle_filter_simulation(0, 0, 0, 0, 0, 0, "",self.plot_canvas)
        self.plot_canvas.draw()
        parameters = 1

        if (self.lined_mparticle.text() != ""):
            self.m_particles = int(self.lined_mparticle.text())
            print(f'Number of particles {self.m_particles}')
        else:
            parameters = 0
            print(f'Incorrect value inserted in Number of Particles field')

        if (self.lined_odovar.text() != ""):
            vec_lined_odovar = self.lined_odovar.text().split(",")
            if (len(vec_lined_odovar) == 4):
                self.var_odom = np.array((float(vec_lined_odovar[0]), float(vec_lined_odovar[1]),
                                          float(vec_lined_odovar[2]), float(vec_lined_odovar[3])), dtype=float)
                #print(f'Odometry Variances are: {self.var_odom[0]}, {self.var_odom[1]}, {self.var_odom[2]}, {self.var_odom[3]}.')

            else:
                print(f'Must have 4 values separated by comma (,)')
                parameters = 0
        else:
            print(f'Must have 4 values separated by comma (,)')
            parameters = 0

        if self.lined_landmarkvar.text() != "" :
            vec_lined_landmarkvar = self.lined_landmarkvar.text().split(",")
            if len(vec_lined_odovar) == 3:
                self.var_landmark = np.array((float(vec_lined_landmarkvar[0]), float(vec_lined_landmarkvar[1]),
                                          float(vec_lined_landmarkvar[2])), dtype=float)
            else:
                print(f'Landmark variance: Must have 3 values separated by comma (,)')
                parameters = 0
        else:
            print(f'Landmark variance: Must have 3 values separated by comma (,)')
            parameters = 0

        #if parameters == 1:
         #   pfilter.particle_filter_simulation(0, 0, 0, 0, 0, 0, "", self.plot_canvas)

if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow()
    """t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2 * pi * t)
    aw.plot_canvas.axes.plot(t, s)"""

    aw.setWindowTitle("%s" % "Particle Filter Simulator")
    aw.show()

    sys.exit(qApp.exec_())