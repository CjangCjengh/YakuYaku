import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTranslator, QSettings
from PyQt6.QtGui import QIcon, QGuiApplication
from ui import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QApplication.setWindowIcon(QIcon("yakuyaku.ico"))
    settings = QSettings("settings.ini", QSettings.Format.IniFormat)
    if settings.contains('language'):
        translator = QTranslator()
        translator.load(settings.value('language'))
        app.installTranslator(translator)
    if settings.contains('style'):
        QApplication.setStyle(settings.value('style'))
    window = MainWindow(settings)
    window.show()
    window.move(QGuiApplication.primaryScreen().availableGeometry().center() - window.rect().center())
    sys.exit(app.exec())
