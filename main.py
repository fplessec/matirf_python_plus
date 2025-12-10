import sys
import argparse


def open_gui():
    from PyQt5.QtWidgets import QApplication
    from gui import ControlWindow
    app = QApplication(sys.argv)
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