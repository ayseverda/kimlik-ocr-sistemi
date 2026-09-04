# -*- coding: utf-8 -*-
"""Onizleme panelindeki, uzerinde fareyle bolge secilebilen gorsel etiket.

desktop.py'den ayrildi -- tek basina, MainWindow'a bagimli olmayan bir
QLabel alt sinifi."""

from PySide6.QtCore import Qt, QRect, QSize, QPoint, Signal
from PySide6.QtWidgets import QLabel, QRubberBand


class OnizlemeEtiketi(QLabel):
    """Üzerinde fareyle dikdörtgen seçilebilen önizleme alanı.

    Tespitin kaçırdığı bir kimliği kullanıcı kendisi çerçeveleyebilsin diye;
    seçim, gösterilen pixmap'in kendi koordinatlarında bildirilir."""

    alan_secildi = Signal(object)   # QRect (pixmap koordinatı) veya None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._baslangic = None
        self._band = QRubberBand(QRubberBand.Rectangle, self)
        # Kart (perspektifi düzeltilmiş, tek kimlik) görünümünde alan seçimi
        # anlamsız — seçilen koordinat sayfa değil kartın kendi pikselinde
        # olur. Bu görünümdeyken seçim kapatılıyor.
        self.secim_acik = True

    def pixmap_ofseti(self):
        """Pixmap ortalanmış çiziliyor; sol üst köşesinin widget içindeki yeri."""
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        return QPoint(max(0, (self.width() - pm.width()) // 2),
                      max(0, (self.height() - pm.height()) // 2))

    def secimi_temizle(self):
        self._baslangic = None
        self._band.hide()

    def mousePressEvent(self, olay):
        if not self.secim_acik or olay.button() != Qt.LeftButton or self.pixmap() is None:
            return
        self._baslangic = olay.position().toPoint()
        self._band.setGeometry(QRect(self._baslangic, QSize()))
        self._band.show()

    def mouseMoveEvent(self, olay):
        if self._baslangic is not None:
            self._band.setGeometry(
                QRect(self._baslangic, olay.position().toPoint()).normalized()
            )

    def mouseReleaseEvent(self, olay):
        if self._baslangic is None:
            return
        kutu = QRect(self._baslangic, olay.position().toPoint()).normalized()
        self._baslangic = None

        ofset = self.pixmap_ofseti()
        pm = self.pixmap()
        if ofset is None or kutu.width() < 12 or kutu.height() < 12:
            self._band.hide()
            self.alan_secildi.emit(None)
            return

        kutu.translate(-ofset.x(), -ofset.y())
        kutu = kutu.intersected(QRect(0, 0, pm.width(), pm.height()))
        if kutu.width() < 10 or kutu.height() < 10:
            self._band.hide()
            self.alan_secildi.emit(None)
            return

        self.alan_secildi.emit(kutu)


