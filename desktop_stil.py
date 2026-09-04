# -*- coding: utf-8 -*-
"""Ana pencerenin koyu tema QSS'i.

desktop.py'den ayrildi -- MainWindow.__init__ 150 satirlik bir stil
stringi tasiyordu, bu da asil pencere kurulum mantigini gormeyi
zorlastiriyordu. Icerik degismedi, sadece kendi dosyasina tasindi."""

KOYU_TEMA = """
            QMainWindow, QWidget {
                background-color: #17191c;
                color: #f3f4f6;
            }

            QLabel {
                color: #f3f4f6;
                font-size: 14px;
            }

            QPushButton {
                color: #f8fafc;
                background-color: #2b2f34;
                border: 1px solid #4a5058;
                border-radius: 8px;
                padding: 8px 14px;
                min-height: 38px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #363b42;
                border-color: #68707a;
            }

            QPushButton:pressed {
                background-color: #22262b;
            }

            QPushButton:disabled {
                color: #777d86;
                background-color: #222529;
                border-color: #34383e;
            }

            QPushButton:checked {
                background-color: #3a4149;
                border: 2px solid #aeb7c2;
            }

            QCheckBox {
                color: #f3f4f6;
                font-size: 14px;
                spacing: 8px;
            }

            QCheckBox:disabled {
                color: #777d86;
            }

            QTableWidget {
                color: #f3f4f6;
                background-color: #1e2125;
                alternate-background-color: #24282d;
                gridline-color: #3a3f46;
                border: 1px solid #3d4249;
                font-size: 13px;
                selection-background-color: #3d4652;
                selection-color: #ffffff;
            }

            QTableWidget::item {
                padding: 5px;
            }

            QHeaderView::section {
                color: #ffffff;
                background-color: #30343a;
                border: 0px;
                border-right: 1px solid #474c54;
                border-bottom: 1px solid #474c54;
                padding: 8px 6px;
                font-size: 13px;
                font-weight: 700;
            }

            QTableCornerButton::section {
                background-color: #30343a;
                border: 1px solid #474c54;
            }

            QProgressBar {
                color: #ffffff;
                background-color: #24282d;
                border: 1px solid #444a52;
                border-radius: 6px;
                min-height: 22px;
                text-align: center;
                font-size: 12px;
            }

            QProgressBar::chunk {
                background-color: #7d8794;
                border-radius: 5px;
            }

            QTextEdit {
                color: #f1f5f9;
                background-color: #111316;
                border: 1px solid #454b53;
                border-radius: 7px;
                padding: 8px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 13px;
            }

            QScrollArea {
                background-color: #111316;
                border: none;
            }

            QScrollBar:vertical {
                background: #1e2125;
                width: 13px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #555d67;
                min-height: 30px;
                border-radius: 6px;
            }

            QScrollBar:horizontal {
                background: #1e2125;
                height: 13px;
                margin: 0;
            }

            QScrollBar::handle:horizontal {
                background: #555d67;
                min-width: 30px;
                border-radius: 6px;
            }

            QSplitter::handle {
                background-color: #4b5159;
            }

            QSplitter::handle:horizontal {
                width: 5px;
            }

            QSplitter::handle:vertical {
                height: 5px;
            }
"""
