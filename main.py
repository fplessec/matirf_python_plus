import sys
import argparse


def fonction1():
    print("Vous avez exécuté real.")


def fonction2():
    print("Vous avez exécuté test.")
    open_gui()

def open_gui():
    from PyQt5.QtWidgets import QApplication
    from gui import ControlWindow
    app = QApplication(sys.argv)
    CW = ControlWindow()
    CW.show()
    sys.exit(app.exec_())


def main():
    parser = argparse.ArgumentParser(description="Ouvrir la fenêtre pour une mesure réelle ou "
                                                 "pour faire des tests sur un objet synthétique")
    parser.add_argument('mode', choices=['real', 'test'],
                        help=".")

    args = parser.parse_args()

    if args.mode == 'real':
        fonction1()
    elif args.mode == 'test':
        fonction2()

if __name__ == '__main__':
    main()