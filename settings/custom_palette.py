from PyQt5.QtGui import QPalette, QColor


def dark_palette():
    palette = QPalette()
    # Backgrounds
    palette.setColor(QPalette.Window, QColor("#3a3a3a"))
    palette.setColor(QPalette.Base, QColor("#2b2b2b"))
    palette.setColor(QPalette.AlternateBase, QColor("#3a3a3a"))
    palette.setColor(QPalette.ToolTipBase, QColor("#2b2b2b"))
    # Text
    palette.setColor(QPalette.WindowText, QColor("#e6e6e6"))
    palette.setColor(QPalette.Text, QColor("#e6e6e6"))
    palette.setColor(QPalette.ToolTipText, QColor("#e6e6e6"))
    palette.setColor(QPalette.PlaceholderText, QColor("#9a9a9a"))
    # Buttons
    palette.setColor(QPalette.Button, QColor("#4a4a4a"))
    palette.setColor(QPalette.ButtonText, QColor("#e6e6e6"))
    # Selection / Highlight
    palette.setColor(QPalette.Highlight, QColor("#5a5a5a"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    # Disabled
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#7a7a7a"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#7a7a7a"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#7a7a7a"))
    # QPalette.Mid is used to match the background of QGroup with the Matplolib Canvas for figure:
    palette.setColor(QPalette.Mid, QColor("#393939"))
    return palette


def light_palette():
    palette = QPalette()
    # Backgrounds
    palette.setColor(QPalette.Window, QColor("#f2f2f2"))          # general background
    palette.setColor(QPalette.Base, QColor("#ffffff"))            # text background
    palette.setColor(QPalette.AlternateBase, QColor("#ededed"))   # alternate lignes
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    # Text
    palette.setColor(QPalette.WindowText, QColor("#2b2b2b"))
    palette.setColor(QPalette.Text, QColor("#2b2b2b"))
    palette.setColor(QPalette.ToolTipText, QColor("#2b2b2b"))
    palette.setColor(QPalette.PlaceholderText, QColor("#9a9a9a"))
    # Buttons
    palette.setColor(QPalette.Button, QColor("#e6e6e6"))
    palette.setColor(QPalette.ButtonText, QColor("#2b2b2b"))
    # Selection / Highlight
    palette.setColor(QPalette.Highlight, QColor("#cfd8e3"))   # bluish gray
    palette.setColor(QPalette.HighlightedText, QColor("#1a1a1a"))
    # Links
    palette.setColor(QPalette.Link, QColor("#4a6fa5"))
    palette.setColor(QPalette.LinkVisited, QColor("#6b4fa5"))
    # Disabled
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#9a9a9a"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#9a9a9a"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9a9a9a"))
    # QPalette.Mid is used to match the background of QGroup with the Matplolib Canvas for figure:
    palette.setColor(QPalette.Mid, QColor("#efefef"))
    return palette
