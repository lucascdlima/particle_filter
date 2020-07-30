from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLabel, QVBoxLayout, QLineEdit
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QIntValidator
import matplotlib
from matplotlib.figure import Figure
import sys
import numpy as np
from numpy import pi
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MyPlotCanvas(FigureCanvas):
    """Class for embbeding Matplotlib features in PyQt.
       In this manner enabling to plot and dysplay in Qt widget """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        #self.compute_initial_figure()

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

        #self.file_menu = QtWidgets.QMenu('&File', self)
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

        self.base_hlayout.addLayout(self.vlayout_left)
        self.base_hlayout.addLayout(self.vlayout_right)

        self.vlayout_left.addWidget(self.button_startsim)
        self.vlayout_left.addWidget(self.label_mparticle)
        self.vlayout_left.addWidget( self.lined_mparticle)
        self.vlayout_left.addWidget(self.label_odovar)
        self.vlayout_left.addWidget(self.lined_odovar)
        self.vlayout_left.addWidget(self.label_landmarkvar)
        self.vlayout_left.addWidget(self.lined_landmarkvar)

        self.plot_canvas = MyPlotCanvas(self.main_widget, width=7, height=4, dpi=100)

        self.vlayout_right.addWidget(self.plot_canvas)

        self.main_widget.setLayout(self.base_hlayout)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.x0 = np.array([0,0,0], dtype=float)
        self.m_particles = 0
        self.var_odom = np.array([0, 0, 0], dtype=float)
        self.var_landmark = np.array([0, 0, 0, 0], dtype=float)

        self.button_startsim.clicked.connect(self.buttonStartSimClick)






    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def buttonStartSimClick(self):

        if(self.lined_mparticle.text() != ""):
            self.m_particles = int(self.lined_mparticle.text())
            print(f'Number of particles {self.m_particles}')
        else:
            print(f'Incorrect value inserted in Number of Particles field')

        if(self.lined_odovar.text()!= ""):
            vec_lined_odovar = self.lined_odovar.text().split(",")
            if(len(vec_lined_odovar)==4):
                a = float(vec_lined_odovar[0])
                self.var_odom = np.array((float(vec_lined_odovar[0]),float(vec_lined_odovar[1]),float(vec_lined_odovar[2]),float(vec_lined_odovar[3])), dtype=float)
                print(f'Odometry Variances are: {self.var_odom[0]}, {self.var_odom[1]}, {self.var_odom[2]}, {self.var_odom[3]}.')
            else:
                print(f'Must have 4 values separated by comma (,)')





qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
t = np.arange(0.0, 3.0, 0.01)
s = np.sin(2 * pi * t)
aw.plot_canvas.axes.plot(t, s)

aw.setWindowTitle("%s" % "Particle Filter Simulator")
aw.show()
sys.exit(qApp.exec_())