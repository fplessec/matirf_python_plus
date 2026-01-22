from PyQt5.QtGui import QPixmap, QColor, QPalette
from PyQt5.QtWidgets import QApplication, QStyleFactory, QGroupBox



def extract_colors(app_style, app_palette):
    app = QApplication([])
    style1 = QStyleFactory.create(app_style)
    style2 = QStyleFactory.create(app_palette)
    app.setStyle(style1)  # <- force the style for any OS
    app.setPalette(style2.standardPalette())  # <- force the color palette for any OS

    gb = QGroupBox()
    gb.resize(100, 50)
    pix = QPixmap(gb.size())
    gb.render(pix)

    # default color used for the chosen app palette:
    qgroupbox_color = QColor(pix.toImage().pixel(10, 25)).name()  # color of the background of the qapp object
    qapp_color = app.palette().color(QPalette.Window).name()  # color of the background of the qgroupbox object

    app.quit()
    #del app, pix, gb

    gray_color = QColor("gray").name()

    return qapp_color, qgroupbox_color, gray_color