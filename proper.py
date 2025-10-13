import sys
from main_window import MainWindow
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QPixmap, QIcon
from PySide6.QtWidgets import QApplication, QSplashScreen
from config import APP_NAME, APP_VERSION, ORG
from processing.resource_loader import setup_logging, load_icon

if __name__ == '__main__':
    log = setup_logging()
    log.info("Application starting...")

    app = QApplication(sys.argv)
    app.setOrganizationName(ORG)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(load_icon())) # Set program Icon
    app.setStyle('Fusion')

    # Splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(app.palette().color(QPalette.Window))
    splash = QSplashScreen(splash_pix)
    message = f"{APP_NAME} Loading...\n\n v{APP_VERSION}"
    splash.showMessage(message, Qt.AlignCenter | Qt.AlignCenter, app.palette().color(QPalette.Text))
    splash.show()

    app.processEvents()
    win = MainWindow()
    win.show()
    splash.finish(win)    # Close splash when ready

    sys.exit(app.exec())
