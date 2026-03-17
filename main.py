import sys
import argparse

import torch
import numpy as np
import random
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QStyleFactory, QApplication

from gui import ControlWindow
import settings


def fixe_randomness(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def center_window_on_cursor_screen(window):
    cursor_pos = QCursor.pos()
    screen = QApplication.screenAt(cursor_pos)

    if screen is None:
        screen = QApplication.primaryScreen()

    screen_geometry = screen.availableGeometry()
    window_geometry = window.frameGeometry()

    window_geometry.moveCenter(screen_geometry.center())
    window.move(window_geometry.topLeft())


def open_gui():

    app = QApplication(sys.argv)
    app_style = QStyleFactory.create(settings.app_style)
    app.setStyle(app_style)  # <- force the style for any OS to macintosh
    # palette_style = QStyleFactory.create(settings.app_palette)
    # app.setPalette(palette_style.standardPalette())
    app.setPalette(settings.app_palette())  # <- force the colors for any OS to macintosh


    CW = ControlWindow()
    center_window_on_cursor_screen(CW)
    CW.show()
    sys.exit(app.exec_())


def main():
    fixe_randomness()
    parser = argparse.ArgumentParser(description="Implémentation seulement avec un GUI pour l'instant")
    parser.add_argument('mode', choices=['gui'], help=".")
    args = parser.parse_args()
    if args.mode == 'gui':
        open_gui()


if __name__ == '__main__':
    main()