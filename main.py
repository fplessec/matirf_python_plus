import sys
import argparse

from PyQt5.QtWidgets import QStyleFactory
import settings


def open_gui():
    from PyQt5.QtWidgets import QApplication
    from gui import ControlWindow
    app = QApplication(sys.argv)
    app_style = QStyleFactory.create(settings.app_style)
    app.setStyle(app_style)  # <- force the style for any OS to macintosh
    palette_style = QStyleFactory.create(settings.app_palette)
    app.setPalette(palette_style.standardPalette())  # <- force the colors for any OS to macintosh
    CW = ControlWindow()
    CW.show()
    sys.exit(app.exec_())


def main():
    parser = argparse.ArgumentParser(description="Implémentation seulement avec un GUI pour l'instant")
    parser.add_argument('mode', choices=['gui'], help=".")
    args = parser.parse_args()
    if args.mode == 'gui':
        open_gui()


if __name__ == '__main__':
    main()