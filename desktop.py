import sys
import os
import time
from datetime import datetime
import tempfile
import subprocess
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

from PySide6.QtCore import Qt, QThread, Signal, QEvent, QRect, QSize, QPoint, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QCheckBox, QSplitter, QHeaderView,
    QAbstractItemView, QTextEdit, QScrollArea, QSizePolicy, QFrame,
    QRubberBand, QDialog
)

from goruntu_isleme import kartlari_tespit_et_ve_duzelt, sayfa_sirasina_diz
from metin_ayiklama import bilgileri_cimbizla, normalize_text, benzerlik
import gecmis
import excel_kaynak
import karsilastirma

BUILD_ID = "DESKTOP-IMAGE-FIX-V10"
print(f"[DESKTOP BUILD] {BUILD_ID}")
print(f"[DESKTOP FILE] {__file__}")
print(f"[CV2 FILE] {getattr(cv2, '__file__', '?')}")
print(f"[CV2 VERSION] {getattr(cv2, '__version__', '?')}")



# Tespit cozemedigi halde "kart buyuklugunde" duran bloklar, dogrudan
# OCR'lanip kurtarilmaya calisilir: gecerli bir TC numarasi cikarsa o kimlik
# sessizce kaybolmak yerine tabloya "kurtarildi" satiri olarak girer.
KURTARMA_AZAMI_BOLGE = 3
KURTARMA_HEDEF_GENISLIK = 1000

PDF_RENDER_DPI = 170
MAX_GORUNTU_PIKSEL = 40_000_000
DESTEKLENEN = {".jpg", ".jpeg", ".png", ".pdf", ".xlsx", ".xlsm"}

# Excel'de yazan ad ile okunan ad bu orandan az benzerse uyuşmazlık bildirilir.
# Karşılaştırma öncesi Türkçe harfler sadeleştiği için (Ş→S, Ü→U…) doğru
# okumalar 1.00 veriyor; eşik bu yüzden yüksek tutulabiliyor.
AD_BENZERLIK_ESIGI = 0.95


def pixmap_to_bgr(pix):
    """
    PyMuPDF Pixmap -> OpenCV BGR.

    PNG'ye yeniden encode/decode etmez; pixmap'in stride/padding
    değerini doğrudan hesaba katar. Farklı PyMuPDF sürümlerinde
    daha uyumludur.
    """
    if pix is None:
        raise ValueError("Pixmap boş.")

    h = int(pix.height)
    w = int(pix.width)
    n = int(pix.n)
    stride = int(pix.stride)

    if h <= 0 or w <= 0 or n <= 0 or stride <= 0:
        raise ValueError(
            f"Geçersiz pixmap: w={w}, h={h}, n={n}, stride={stride}"
        )

    ham = np.frombuffer(
        pix.samples,
        dtype=np.uint8,
    )

    beklenen = h * stride
    if ham.size < beklenen:
        raise ValueError(
            f"Pixmap byte sayısı yetersiz: {ham.size} < {beklenen}"
        )

    # Her satırdaki olası padding'i at.
    satirlar = ham[:beklenen].reshape(h, stride)
    piksel_bayt = w * n

    if stride < piksel_bayt:
        raise ValueError(
            f"Pixmap stride beklenenden küçük: stride={stride}, "
            f"w*n={piksel_bayt}"
        )

    img = satirlar[:, :piksel_bayt].reshape(
        h,
        w,
        n,
    )

    if n == 4:
        return cv2.cvtColor(
            img,
            cv2.COLOR_RGBA2BGR,
        )

    if n == 3:
        return cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR,
        )

    if n == 1:
        return cv2.cvtColor(
            img[:, :, 0],
            cv2.COLOR_GRAY2BGR,
        )

    raise ValueError(
        f"Desteklenmeyen pixmap kanal sayısı: {n}"
    )


def dosyayi_goruntulere_ayir(yol):
    """(sayfa_no, görüntü, hata, ek_bilgi) üretir.

    ek_bilgi yalnız Excel kaynağında dolu: o bantta Excel'in kendi yazdığı
    kişi adını taşır."""
    ext = os.path.splitext(yol)[1].lower()

    if ext in (".xlsx", ".xlsm"):
        # Her bant (kişi) tek sayfa gibi ele alınır: bandın fotoğrafları
        # Excel'deki gibi yan yana birleştirilip verilir.
        try:
            bantlar = excel_kaynak.bantlari_bul(yol)
        except Exception as e:
            yield None, None, f"Excel okunamadı: {type(e).__name__}: {e}", None
            return

        if not bantlar:
            yield None, None, "Excel dosyasında gömülü kimlik fotoğrafı bulunamadı.", None
            return

        for bant in bantlar:
            ek = {"excel_adi": bant.get("ad", ""), "excel_satiri": bant.get("excel_satiri")}
            try:
                resim = excel_kaynak.bandi_birlestir(bant["gorseller"])
            except Exception as e:
                yield bant["no"], None, f"Excel görüntüsü çözülemedi: {e}", ek
                continue
            if resim is None:
                yield bant["no"], None, "Excel görüntüsü okunamadı.", ek
            else:
                yield bant["no"], resim, None, ek
        return

    if ext == ".pdf":
        doc = pymupdf.open(yol)
        try:
            for sayfa_no, sayfa in enumerate(doc, start=1):
                tahmini_w = max(1, int(sayfa.rect.width * PDF_RENDER_DPI / 72))
                tahmini_h = max(1, int(sayfa.rect.height * PDF_RENDER_DPI / 72))
                if tahmini_w * tahmini_h > MAX_GORUNTU_PIKSEL:
                    yield sayfa_no, None, "Sayfa çözünürlüğü güvenli sınırı aşıyor.", None
                    continue
                try:
                    # Streamlit'te çalışan çağrının aynısı.
                    # colorspace sabiti vermiyoruz; eski/yeni PyMuPDF
                    # sürümleri arasındaki uyumsuzluğu da böyle önlüyoruz.
                    pix = sayfa.get_pixmap(
                        dpi=PDF_RENDER_DPI,
                        alpha=False,
                    )
                    resim = pixmap_to_bgr(
                        pix
                    )
                    yield sayfa_no, resim, None, None

                except Exception as e:
                    hata = (
                        "PDF sayfası görüntüye çevrilemedi: "
                        f"{type(e).__name__}: {e}"
                    )

                    print(
                        f"[PDF HATA] {os.path.basename(yol)} "
                        f"/ sayfa {sayfa_no}: {hata}",
                        flush=True,
                    )

                    yield sayfa_no, None, hata, None
        finally:
            doc.close()
    else:
        try:
            # Windows'ta cv2.imread, Türkçe/özel karakterli dosya
            # yollarında zaman zaman başarısız olabiliyor.
            # JPG/PNG'yi Pillow ile okuyup OpenCV BGR'ye çeviriyoruz.
            from PIL import ImageOps

            with Image.open(yol) as im:
                im = ImageOps.exif_transpose(im)

                w, h = im.size

                if w * h > MAX_GORUNTU_PIKSEL:
                    yield (
                        None,
                        None,
                        f"Görüntü çözünürlüğü güvenli sınırı aşıyor ({w}×{h}).",
                        None,
                    )
                    return

                im = im.convert("RGB")
                rgb = np.array(im)

            if rgb is None or rgb.size == 0:
                yield None, None, "Görüntü boş veya okunamadı.", None
                return

            resim = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            yield None, resim, None, None

        except Exception as e:
            hata = (
                "Görüntü okunamadı: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[GÖRÜNTÜ HATA] {os.path.basename(yol)}: {hata}",
                flush=True,
            )

            yield None, None, hata, None


def resmi_pdfe_ekle(pdf_doc, resim):
    if resim is None:
        return False
    try:
        h, w = resim.shape[:2]
        ok, encoded = cv2.imencode(".jpg", resim, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return False
        pw = 720.0
        ph = pw * h / w
        page = pdf_doc.new_page(width=pw, height=ph)
        page.insert_image(pymupdf.Rect(0, 0, pw, ph), stream=encoded.tobytes(), keep_proportion=True)
        return True
    except Exception:
        return False


def bgr_to_pixmap(img, max_w=650, max_h=700):
    if img is None:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def dosyayi_sistemde_ac(yol):
    """
    Oluşturulan Excel/PDF dosyasını varsayılan masaüstü uygulamasıyla açar.

    Windows:
        os.startfile
    macOS:
        open
    Linux:
        xdg-open
    """
    if not yol or not os.path.exists(yol):
        raise FileNotFoundError(
            f"Dosya bulunamadı: {yol}"
        )

    if sys.platform.startswith("win"):
        os.startfile(yol)
        return

    if sys.platform == "darwin":
        subprocess.Popen(
            ["open", yol]
        )
        return

    subprocess.Popen(
        ["xdg-open", yol]
    )




class KarsilastirmaPenceresi(QDialog):
    """Dış listeyle karşılaştırma sonucunu gösterir.

    En önemli satırlar "Bizde yok" olanlar: dış listede kayıtlı olduğu halde
    bizim çıkardığımız sonuçlarda karşılığı bulunmayan kimlikler."""

    RENKLER = {
        "Eşleşti": "#86efac",
        "Ad farklı": "#f0b429",
        "Bizde yok": "#ff6b6b",
        "Listede yok": "#93c5fd",
    }

    def __init__(self, sonuc, bilgi, dosya_adi, ebeveyn=None):
        super().__init__(ebeveyn)
        self.setWindowTitle("Liste karşılaştırması")
        self.resize(1020, 620)
        self.sonuc = sonuc
        self.satirlar = karsilastirma.satirlara_don(sonuc)

        duzen = QVBoxLayout(self)

        ozet = QLabel(karsilastirma.ozet_metni(sonuc, dosya_adi))
        ozet.setWordWrap(True)
        ozet.setStyleSheet("color: #f3f4f6; font-size: 15px; font-weight: 700;")
        duzen.addWidget(ozet)

        sutun_yazi = ", ".join(f"{ad}={sutun}" for ad, sutun in
                               sorted((bilgi.get("sutunlar") or {}).items()))
        ayrinti = QLabel(
            f"Okunan sayfa: {bilgi.get('sayfa', '-')}   •   "
            f"başlık satırı: {bilgi.get('baslik_satiri') or 'yok'}   •   "
            f"sütunlar: {sutun_yazi or '-'}"
        )
        ayrinti.setWordWrap(True)
        ayrinti.setStyleSheet("color: #9aa3ad; font-size: 12px;")
        duzen.addWidget(ayrinti)

        self.tablo = QTableWidget()
        self.basliklar = ["Durum", "Kimlik No", "Listedeki Ad Soyad",
                          "Bizdeki Ad Soyad", "Not"]
        self.tablo.setColumnCount(len(self.basliklar))
        self.tablo.setHorizontalHeaderLabels(self.basliklar)
        self.tablo.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tablo.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tablo.setAlternatingRowColors(True)
        self.tablo.verticalHeader().setVisible(False)
        for i, genislik in enumerate([110, 130, 230, 230, 280]):
            self.tablo.setColumnWidth(i, genislik)
        duzen.addWidget(self.tablo, 1)

        dugmeler = QHBoxLayout()
        self.filtre_btn = QPushButton("Yalnız sorunluları göster")
        self.filtre_btn.setCheckable(True)
        self.aktar_btn = QPushButton("Excel’e aktar ve aç")
        self.kapat_btn = QPushButton("Kapat")
        for b in (self.filtre_btn, self.aktar_btn, self.kapat_btn):
            b.setMinimumHeight(36)
        dugmeler.addWidget(self.filtre_btn)
        dugmeler.addStretch(1)
        dugmeler.addWidget(self.aktar_btn)
        dugmeler.addWidget(self.kapat_btn)
        duzen.addLayout(dugmeler)

        self.filtre_btn.toggled.connect(self.tabloyu_doldur)
        self.aktar_btn.clicked.connect(self.excele_aktar)
        self.kapat_btn.clicked.connect(self.accept)

        self.tabloyu_doldur()

    def gosterilecek_satirlar(self):
        if self.filtre_btn.isChecked():
            return [s for s in self.satirlar if s["Durum"] != "Eşleşti"]
        return self.satirlar

    def tabloyu_doldur(self):
        satirlar = self.gosterilecek_satirlar()
        # Sorunlular üstte olsun: bizde olmayanlar, ad farklılıkları, sonra kalanlar.
        oncelik = {"Bizde yok": 0, "Ad farklı": 1, "Listede yok": 2, "Eşleşti": 3}
        satirlar = sorted(satirlar, key=lambda s: oncelik.get(s["Durum"], 9))

        self.tablo.setRowCount(0)
        for satir in satirlar:
            r = self.tablo.rowCount()
            self.tablo.insertRow(r)
            for c, baslik in enumerate(self.basliklar):
                item = QTableWidgetItem(str(satir.get(baslik, "")))
                if c == 0:
                    item.setForeground(QColor(self.RENKLER.get(satir["Durum"], "#f3f4f6")))
                self.tablo.setItem(r, c, item)

    def excele_aktar(self):
        if not self.satirlar:
            return
        try:
            fd, yol = tempfile.mkstemp(prefix="karsilastirma_", suffix=".xlsx")
            os.close(fd)

            df = pd.DataFrame(self.satirlar, columns=self.basliklar)
            df.to_excel(yol, index=False, engine="openpyxl")

            wb = load_workbook(yol)
            ws = wb.active
            ws.title = "Karşılaştırma"
            for hucre in ws[1]:
                hucre.font = Font(bold=True)

            dolgular = {
                "Bizde yok": PatternFill("solid", fgColor="FFC7CE"),
                "Ad farklı": PatternFill("solid", fgColor="FFD9A0"),
                "Listede yok": PatternFill("solid", fgColor="DDEBF7"),
            }
            for i, satir in enumerate(self.satirlar, start=2):
                dolgu = dolgular.get(satir["Durum"])
                if dolgu:
                    for c in range(1, len(self.basliklar) + 1):
                        ws.cell(i, c).fill = dolgu

            for harf, genislik in zip("ABCDE", (14, 16, 28, 28, 34)):
                ws.column_dimensions[harf].width = genislik

            wb.save(yol)
            dosyayi_sistemde_ac(yol)
        except Exception as e:
            QMessageBox.critical(self, "Aktarma hatası", str(e))


class GecmisPenceresi(QDialog):
    """Kayıtlı taramaları listeler; seçileni açar, siler veya hepsini temizler."""

    def __init__(self, ebeveyn=None):
        super().__init__(ebeveyn)
        self.setWindowTitle("Geçmiş taramalar")
        self.resize(760, 460)
        self.secilen_yol = None

        duzen = QVBoxLayout(self)

        self.bilgi = QLabel("")
        self.bilgi.setWordWrap(True)
        duzen.addWidget(self.bilgi)

        self.liste = QTableWidget()
        self.basliklar = ["Tarih", "Dosyalar", "Kimlik", "Satır", "Boyut"]
        self.liste.setColumnCount(len(self.basliklar))
        self.liste.setHorizontalHeaderLabels(self.basliklar)
        self.liste.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.liste.setSelectionMode(QAbstractItemView.SingleSelection)
        self.liste.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.liste.setAlternatingRowColors(True)
        self.liste.verticalHeader().setVisible(False)
        for i, genislik in enumerate([160, 330, 80, 70, 90]):
            self.liste.setColumnWidth(i, genislik)
        self.liste.doubleClicked.connect(self.ac)
        duzen.addWidget(self.liste, 1)

        dugmeler = QHBoxLayout()
        self.ac_btn = QPushButton("Aç")
        self.sil_btn = QPushButton("Seçileni sil")
        self.temizle_btn = QPushButton("Geçmişi temizle")
        self.kapat_btn = QPushButton("Kapat")
        for b in (self.ac_btn, self.sil_btn, self.temizle_btn, self.kapat_btn):
            b.setMinimumHeight(36)
        self.temizle_btn.setStyleSheet(
            "QPushButton { background-color: #4a2020; border-color: #7a3030; }"
        )

        dugmeler.addWidget(self.ac_btn)
        dugmeler.addWidget(self.sil_btn)
        dugmeler.addStretch(1)
        dugmeler.addWidget(self.temizle_btn)
        dugmeler.addWidget(self.kapat_btn)
        duzen.addLayout(dugmeler)

        self.ac_btn.clicked.connect(self.ac)
        self.sil_btn.clicked.connect(self.sil)
        self.temizle_btn.clicked.connect(self.temizle)
        self.kapat_btn.clicked.connect(self.reject)

        self.yenile()

    def yenile(self):
        self.kayitlar = gecmis.taramalari_listele()
        self.liste.setRowCount(0)

        for kayit in self.kayitlar:
            r = self.liste.rowCount()
            self.liste.insertRow(r)
            try:
                tarih = datetime.fromisoformat(kayit.get("tarih", "")).strftime("%d.%m.%Y %H:%M")
            except (ValueError, TypeError):
                tarih = kayit.get("tarih", "")
            if kayit.get("guncellendi"):
                try:
                    tarih += datetime.fromisoformat(kayit["guncellendi"]).strftime(
                        "  (düzenlendi %H:%M)")
                except (ValueError, TypeError):
                    tarih += "  (düzenlendi)"
            dosyalar = ", ".join(kayit.get("dosyalar", []))
            degerler = [
                tarih,
                dosyalar,
                str(kayit.get("kimlik_sayisi", "")),
                str(kayit.get("satir_sayisi", "")),
                f"{kayit.get('boyut', 0) / 1024 / 1024:.1f} MB",
            ]
            for c, deger in enumerate(degerler):
                item = QTableWidgetItem(deger)
                if c == 1:
                    item.setToolTip(dosyalar)
                self.liste.setItem(r, c, item)

        toplam = sum(k.get("boyut", 0) for k in self.kayitlar)
        self.bilgi.setText(
            f"{len(self.kayitlar)} tarama saklanıyor (toplam {toplam / 1024 / 1024:.1f} MB). "
            f"En yeni {gecmis.AZAMI_KAYIT} tarama tutulur, eskiler kendiliğinden silinir.\n"
            f"Klasör: {gecmis.gecmis_dizini()}"
        )
        self.ac_btn.setEnabled(bool(self.kayitlar))
        self.sil_btn.setEnabled(bool(self.kayitlar))
        self.temizle_btn.setEnabled(bool(self.kayitlar))
        if self.kayitlar:
            self.liste.selectRow(0)

    def secili_kayit(self):
        r = self.liste.currentRow()
        if 0 <= r < len(self.kayitlar):
            return self.kayitlar[r]
        return None

    def ac(self):
        kayit = self.secili_kayit()
        if kayit:
            self.secilen_yol = kayit["yol"]
            self.accept()

    def sil(self):
        kayit = self.secili_kayit()
        if not kayit:
            return
        if QMessageBox.question(
            self, "Kaydı sil",
            f"{', '.join(kayit.get('dosyalar', []))} taraması silinsin mi?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes:
            gecmis.taramayi_sil(kayit["yol"])
            self.yenile()

    def temizle(self):
        if QMessageBox.question(
            self, "Geçmişi temizle",
            "Kayıtlı bütün taramalar silinsin mi?\n"
            "Bu dosyalar kimlik numarası ve isim içerir; silme geri alınamaz.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes:
            gecmis.gecmisi_temizle()
            self.yenile()


class OnizlemeEtiketi(QLabel):
    """Üzerinde fareyle dikdörtgen seçilebilen önizleme alanı.

    Tespitin kaçırdığı bir kimliği kullanıcı kendisi çerçeveleyebilsin diye;
    seçim, gösterilen pixmap'in kendi koordinatlarında bildirilir."""

    alan_secildi = Signal(object)   # QRect (pixmap koordinatı) veya None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._baslangic = None
        self._band = QRubberBand(QRubberBand.Rectangle, self)

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
        if olay.button() != Qt.LeftButton or self.pixmap() is None:
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


class AlanOkuyucu(QThread):
    """Kullanıcının seçtiği alanı arka planda okur (arayüz donmasın).

    Önce seçilen parçada normal kart tespiti denenir — bulunursa kart
    hizalanmış haliyle okunur, bu en iyi sonucu verir. Bulunamazsa seçilen
    alan doğrudan büyütülüp okunur."""

    bitti = Signal(dict)
    hata = Signal(str)

    def __init__(self, kirpma, debug=False):
        super().__init__()
        self.kirpma = kirpma
        self.debug = debug

    def run(self):
        try:
            kart = None
            belge_tipi = "tc"
            hizalandi = False
            try:
                adaylar = [
                    k for k in kartlari_tespit_et_ve_duzelt(self.kirpma, debug_kart=self.debug)
                    if k.get("basarili")
                ]
            except Exception:
                adaylar = []

            if adaylar:
                kart = adaylar[0].get("kart")
                belge_tipi = adaylar[0].get("belge_tipi", "tc")
                hizalandi = True
            else:
                kart = self.kirpma
                if kart.shape[1] < KURTARMA_HEDEF_GENISLIK:
                    olcek = KURTARMA_HEDEF_GENISLIK / float(kart.shape[1])
                    kart = cv2.resize(kart, None, fx=olcek, fy=olcek,
                                      interpolation=cv2.INTER_CUBIC)
                kart = np.ascontiguousarray(kart)

            ocr_sonuc = bilgileri_cimbizla(kart, debug=self.debug, belge_tipi=belge_tipi)
            self.bitti.emit({
                "kart": kart, "ocr": ocr_sonuc,
                "belge_tipi": belge_tipi, "hizalandi": hizalandi,
            })
        except Exception as exc:
            self.hata.emit(f"{type(exc).__name__}: {exc}")


class Worker(QThread):
    progress = Signal(int, int, str)
    row_ready = Signal(dict)
    finished_ok = Signal(list, object, bool)
    failed = Signal(str)

    def __init__(self, dosyalar, debug=False, kurtar=True, coklu=True, derin=True):
        super().__init__()
        self.dosyalar = dosyalar
        self.debug = debug
        self.kurtar = kurtar
        self.coklu = coklu
        self.derin = derin
        self._durduruldu = False

    def durdur(self):
        """Dışarıdan (ana thread'den) çağrılır. Basit bir bool bayrak —
        Python'da GIL sayesinde tek bir attribute yazımı thread-safe kabul
        edilir, ekstra kilide gerek yok. run() döngüsü bunu her sayfa
        aralığında kontrol eder; şu an işlenmekte olan sayfa yarıda
        kesilmez, bir SONRAKİ sayfaya geçmeden döngü sonlanır."""
        self._durduruldu = True

    def run(self):
        try:
            isler = []
            for yol in self.dosyalar:
                if yol.lower().endswith(".pdf"):
                    try:
                        with pymupdf.open(yol) as d:
                            for p in range(1, d.page_count + 1):
                                isler.append((yol, p))
                    except Exception:
                        isler.append((yol, None))
                elif excel_kaynak.excel_mi(yol):
                    try:
                        for bant in excel_kaynak.bantlari_bul(yol):
                            isler.append((yol, bant["no"]))
                    except Exception:
                        isler.append((yol, None))
                else:
                    isler.append((yol, None))

            toplam = max(1, len(isler))
            sonuclar = []
            sayac = 0

            for yol in self.dosyalar:
                if self._durduruldu:
                    break
                for sayfa_no, resim, girdi_hatasi, ek_bilgi in dosyayi_goruntulere_ayir(yol):
                    if self._durduruldu:
                        break
                    sayac += 1
                    self.progress.emit(sayac, toplam, f"{os.path.basename(yol)} işleniyor")

                    baslangic = time.perf_counter()
                    kart_sonuclari = []
                    tespit_hatasi = girdi_hatasi

                    if resim is not None:
                        try:
                            # Bir sayfada birden fazla kimlik olabilir; hepsi
                            # ayrı ayrı tespit edilip ayrı satır olarak işlenir.
                            kart_sonuclari = kartlari_tespit_et_ve_duzelt(
                                resim, debug_kart=self.debug, ek_tarama=self.coklu
                            )
                        except Exception as e:
                            tespit_hatasi = f"Kart tespit hatası: {e}"

                    if not kart_sonuclari:
                        kart_sonuclari = [{}]

                    tespit_suresi = time.perf_counter() - baslangic
                    kart_sayisi = sum(1 for k in kart_sonuclari if k.get("basarili"))

                    sayfa_satirlari = []
                    for sira, kart_sonuc in enumerate(kart_sonuclari, start=1):
                        sayfa_satirlari.append(self._satir_olustur(
                            yol=yol,
                            sayfa_no=sayfa_no,
                            resim=resim,
                            kart_sonuc=kart_sonuc,
                            sira=sira,
                            kart_sayisi=kart_sayisi,
                            tespit_hatasi=tespit_hatasi,
                            # Tespit süresi sayfanın tamamı için bir kez harcanır;
                            # sayfanın ilk satırına yazılır, diğerlerine yazılmaz.
                            ek_sure=tespit_suresi if sira == 1 else 0.0,
                            ek_bilgi=ek_bilgi,
                        ))

                    if ek_bilgi is not None:
                        sayfa_satirlari = self._excel_bandini_sadelestir(
                            sayfa_satirlari, ek_bilgi
                        )

                    # Tespitin çözemediği kart bloklarını doğrudan okumayı dene.
                    # Sadece sayfadan HİÇ kart çıkmadığında: kart bulunan
                    # sayfalarda arta kalan bloklar çoğunlukla kimliğin arka
                    # yüzü oluyor ve oradan üretilen satır hayalet kayıt olur.
                    if self.kurtar and resim is not None and kart_sayisi == 0:
                        kurtarilan = self._kurtarma_satirlari(
                            yol=yol,
                            sayfa_no=sayfa_no,
                            resim=resim,
                            bolgeler=(kart_sonuclari[0] or {}).get("kurtarma_bolgeleri", []),
                            mevcut_satirlar=sayfa_satirlari,
                        )
                        if kurtarilan:
                            # Sayfada hiç kart bulunamadıysa "bulunamadı" satırı
                            # artık yanıltıcı olur; kurtarılan kimlikler onun yerini alır.
                            if kart_sayisi == 0:
                                sayfa_satirlari = []
                            sayfa_satirlari.extend(kurtarilan)

                    for row in sayfa_satirlari:
                        sonuclar.append(row)
                        self.row_ready.emit(row)

            # PDF artık burada üretilmiyor: kullanıcı elle kimlik ekleyip
            # sırayı değiştirebildiği için, kaydetme anında tablodaki güncel
            # sıradan üretiliyor (bkz. MainWindow.pdf_kaydet).
            self.finished_ok.emit(sonuclar, None, self._durduruldu)
        except Exception as e:
            self.failed.emit(repr(e))

    def _excel_bandini_sadelestir(self, satirlar, ek_bilgi):
        """Excel'de bir bant tek kişidir: banttan TEK satır bırakılır.

        Kartın arka yüzü nadiren kart olarak da tespit edilebiliyor ve boş bir
        ikinci satır üretiyordu. En çok alanı okunan satır tutulur; ad/soyad
        eksikse Excel'in kendi yazdığı isimden tamamlanır, okunduysa onunla
        karşılaştırılır."""
        if not satirlar:
            return satirlar

        def dolu_alan(satir):
            return sum(
                1 for alan in ("Kimlik No", "Ad", "Soyad")
                if str(satir.get(alan, "")).strip() not in ("", "Bulunamadi")
            )

        satir = max(satirlar, key=dolu_alan)
        satir["Kart"] = "1/1" if satir.get("_kart_bulundu") else "-"

        excel_adi = (ek_bilgi or {}).get("excel_adi") or ""
        excelden = []
        ad_uyusmazligi = ""

        if excel_adi:
            e_ad, e_soyad = excel_kaynak.adi_ayir(excel_adi)
            if str(satir.get("Ad")) == "Bulunamadi" and e_ad:
                satir["Ad"] = e_ad
                excelden.append("Ad")
            if str(satir.get("Soyad")) == "Bulunamadi" and e_soyad:
                satir["Soyad"] = e_soyad
                excelden.append("Soyad")
            if not excelden:
                okunan = normalize_text(f"{satir.get('Ad')} {satir.get('Soyad')}")
                if benzerlik(okunan, normalize_text(excel_adi)) < AD_BENZERLIK_ESIGI:
                    ad_uyusmazligi = excel_adi

        satir["_excelden_alanlar"] = excelden
        satir["_ad_uyusmazligi"] = bool(ad_uyusmazligi)
        if excelden:
            satir["_kart_bulundu"] = True

        # Durum, tamamlanan alanlardan sonra yeniden yazılıyor.
        eksikler = [
            alan for alan in ("Kimlik No", "Ad", "Soyad")
            if str(satir.get(alan, "")).strip() in ("", "Bulunamadi")
        ]
        if satir.get("_hata") and not satir.get("_kart_bulundu"):
            durum = str(satir["_hata"])
        elif eksikler:
            durum = "Eksik alan: " + ", ".join(eksikler)
        elif satir.get("_tc_supheli"):
            durum = "Kimlik No doğrulanamadı — kontrol edin"
        else:
            durum = "Başarılı"
        if excelden:
            durum += f" ({'/'.join(excelden)} Excel'den alındı)"
        if ad_uyusmazligi:
            durum += f" — Excel'de: {ad_uyusmazligi}"
        satir["Durum"] = durum

        return [satir]

    def _kurtarma_satirlari(self, yol, sayfa_no, resim, bolgeler, mevcut_satirlar):
        """Tespitin kart çıkaramadığı blokları doğrudan OCR'lar.

        Kimlik kartı hizalanamadığında eskiden o kimlik tamamen kayboluyordu.
        Burada blok kırpılıp büyütülüyor ve normal alan okuma akışından
        geçiriliyor; GEÇERLİ (checksum'ı tutan) bir kimlik numarası çıkarsa
        satır olarak ekleniyor.

        Kartın arka yüzü de kart büyüklüğünde bir bloktur ve MRZ'sinde aynı
        numara yazar; o yüzden sayfada ZATEN görülmüş numaralar atlanıyor —
        arka yüzler kendiliğinden elenmiş oluyor."""
        gorulen_tcler = {
            str(r.get("Kimlik No", "")).strip()
            for r in mevcut_satirlar
            if str(r.get("Kimlik No", "")).strip() not in ("", "Bulunamadi")
        }

        satirlar = []
        for x0, y0, x1, y1 in list(bolgeler)[:KURTARMA_AZAMI_BOLGE]:
            if self._durduruldu:
                break

            baslangic = time.perf_counter()
            blok = resim[int(y0):int(y1), int(x0):int(x1)]
            if blok is None or blok.size == 0 or blok.shape[0] < 2 or blok.shape[1] < 2:
                continue

            # OCR, düzeltilmiş kart boyutlarında eğitilmiş oranlara göre
            # çalışıyor; küçük bloğu o ölçeğe büyütüyoruz.
            if blok.shape[1] < KURTARMA_HEDEF_GENISLIK:
                olcek = KURTARMA_HEDEF_GENISLIK / float(blok.shape[1])
                blok = cv2.resize(blok, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_CUBIC)
            blok = np.ascontiguousarray(blok)

            try:
                ocr_sonuc = bilgileri_cimbizla(
                    blok, sayfa_no=sayfa_no, debug=self.debug, belge_tipi="tc",
                    derin=self.derin,
                )
            except Exception:
                continue

            kimlik_no = str(ocr_sonuc.get("tc_no", "Bulunamadi")).strip()
            if kimlik_no in ("", "Bulunamadi") or kimlik_no in gorulen_tcler:
                continue
            # Kurtarmada yalnız doğrulanmış numaraya güveniyoruz; hizalanmamış
            # bloktan gelen şüpheli okuma yanlış kişi kaydı üretebilir.
            if not ocr_sonuc.get("tc_dogrulandi", True):
                continue
            gorulen_tcler.add(kimlik_no)

            ad = ocr_sonuc.get("ad", "Bulunamadi")
            soyad = ocr_sonuc.get("soyad", "Bulunamadi")

            eksikler = [
                alan for alan, deger in (("Ad", ad), ("Soyad", soyad))
                if deger == "Bulunamadi"
            ]
            durum = "Kart hizalanamadı, sayfadan okundu"
            if eksikler:
                durum += " — eksik alan: " + ", ".join(eksikler)

            satirlar.append({
                "Dosya": os.path.basename(yol),
                "Sayfa": sayfa_no if sayfa_no is not None else "-",
                "Kart": "K",
                "Belge Türü": "T.C. Kimlik",
                "Kimlik No": kimlik_no,
                "Ad": ad,
                "Soyad": soyad,
                "Bitiş Tarihi": "-",
                "Geçerlilik": "-",
                "Durum": durum,
                "Süre": round(time.perf_counter() - baslangic, 2),
                "_belge_tipi": "tc",
                "_belge_gecerli": None,
                "_kart_bulundu": True,
                "_kurtarildi": True,
                "_kaynak_yol": yol,
                "_koseler": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "_ad_conf": float(ocr_sonuc.get("ad_conf", 0.0) or 0.0),
                "_soyad_conf": float(ocr_sonuc.get("soyad_conf", 0.0) or 0.0),
                "_preview": blok,
                "_debug_resmi": ocr_sonuc.get("debug_resmi"),
                "_kart_debug": None,
                "_kart_sonuc": {
                    "belge_tipi": "tc",
                    "mesaj": "Kart tespit edilemedi; blok doğrudan okundu (kurtarma).",
                    "arama_modu": "kurtarma",
                    "referans_hatalari": {},
                },
                "_ocr_sonuc": {
                    "guven": ocr_sonuc.get("guven"),
                    "kimlik_no_conf": ocr_sonuc.get("kimlik_no_conf", 0.0),
                    "ad_conf": ocr_sonuc.get("ad_conf", 0.0),
                    "soyad_conf": ocr_sonuc.get("soyad_conf", 0.0),
                    "ocr_suresi": ocr_sonuc.get("ocr_suresi", 0.0),
                    "tum_ocr": ocr_sonuc.get("tum_ocr", []) if self.debug else [],
                },
            })

        return satirlar

    def _satir_olustur(self, yol, sayfa_no, resim, kart_sonuc, sira, kart_sayisi,
                       tespit_hatasi, ek_sure, ek_bilgi=None):
        """Tek bir kimlik (sayfadaki bir kart) için OCR'ı çalıştırıp tablo
        satırını üretir. Sayfada hiç kart bulunamadıysa da tek bir
        "bulunamadı" satırı üretilir."""
        baslangic = time.perf_counter()

        hata = tespit_hatasi
        belge_tipi = "bilinmiyor"
        kart = None
        ocr_sonuc = {}

        if kart_sonuc.get("basarili", False):
            kart = kart_sonuc.get("kart")
            belge_tipi = kart_sonuc.get("belge_tipi", "tc")
        else:
            eslesen = kart_sonuc.get("belge_tipi")
            if eslesen in {"tc", "eski_tc", "gocmen"}:
                belge_tipi = eslesen
            if not hata and kart_sonuc:
                hata = kart_sonuc.get("mesaj") or "Kart tespit edilemedi."

        # Streamlit backend'indeki kritik bağlantı:
        # OCR yalnızca tespit edilen/düzeltilen kart üzerinde ve belge_tipi ile çalışır.
        if kart is not None:
            try:
                ocr_sonuc = bilgileri_cimbizla(
                    kart,
                    sayfa_no=sayfa_no,
                    debug=self.debug,
                    belge_tipi=belge_tipi,
                    derin=self.derin,
                )
                if ocr_sonuc.get("hata"):
                    hata = str(ocr_sonuc["hata"])
            except Exception as e:
                hata = f"OCR hatası: {e}"

        kimlik_no = ocr_sonuc.get("tc_no", "Bulunamadi")
        ad = ocr_sonuc.get("ad", "Bulunamadi")
        soyad = ocr_sonuc.get("soyad", "Bulunamadi")
        bitis = ocr_sonuc.get("bitis_tarihi", "")
        belge_gecerli = ocr_sonuc.get("belge_gecerli")
        ad_conf = float(ocr_sonuc.get("ad_conf", 0.0) or 0.0)
        soyad_conf = float(ocr_sonuc.get("soyad_conf", 0.0) or 0.0)

        # Numara okundu ama checksum'ı tutmuyor: kaybetmek yerine
        # "doğrulanamadı" diye gösteriyoruz, kullanıcı karta bakıp düzeltebilsin.
        tc_dogrulandi = bool(ocr_sonuc.get("tc_dogrulandi", True))
        tc_supheli = kimlik_no != "Bulunamadi" and not tc_dogrulandi

        excel_adi = (ek_bilgi or {}).get("excel_adi") or ""

        eksikler = []
        if kart is None:
            eksikler.append("Kimlik")
        else:
            if kimlik_no == "Bulunamadi":
                eksikler.append("Kimlik No")
            if ad == "Bulunamadi":
                eksikler.append("Ad")
            if soyad == "Bulunamadi":
                eksikler.append("Soyad")
            if belge_tipi == "gocmen" and not bitis:
                eksikler.append("Bitiş Tarihi")

        if hata:
            durum = hata
        elif eksikler:
            durum = "Eksik alan: " + ", ".join(eksikler)
        elif tc_supheli:
            durum = "Kimlik No doğrulanamadı — kontrol edin"
        else:
            durum = "Başarılı"

        if tc_supheli and eksikler:
            durum += " (Kimlik No doğrulanamadı)"

        belge_yazi = {
            "tc": "T.C. Kimlik",
            "eski_tc": "Eski T.C. Kimlik",
            "gocmen": "Göçmen / Yabancı",
        }.get(belge_tipi, "Bilinmiyor")

        if belge_tipi == "gocmen":
            if belge_gecerli is False:
                gecerlilik = "GEÇERSİZ"
            elif belge_gecerli is True:
                gecerlilik = "Geçerli"
            else:
                gecerlilik = "Kontrol Edilemedi"
        else:
            gecerlilik = "-"

        return {
            "Dosya": os.path.basename(yol),
            "Sayfa": sayfa_no if sayfa_no is not None else "-",
            "Kart": f"{sira}/{kart_sayisi}" if kart_sayisi else "-",
            "Belge Türü": belge_yazi,
            "Kimlik No": kimlik_no,
            "Ad": ad,
            "Soyad": soyad,
            "Bitiş Tarihi": bitis if belge_tipi == "gocmen" else "-",
            "Geçerlilik": gecerlilik,
            "Durum": durum,
            "Süre": round(time.perf_counter() - baslangic + ek_sure, 2),
            "_belge_tipi": belge_tipi,
            "_belge_gecerli": belge_gecerli,
            "_kart_bulundu": kart is not None,
            "_tc_supheli": tc_supheli,
            "_excel_adi": excel_adi,
            "_hata": hata,
            # Sayfanın tamamını işaretli göstermek için: kaynak dosya + kartın
            # sayfa üzerindeki köşeleri. Sayfa görüntüsü saklanmıyor, gerektiğinde
            # dosyadan yeniden üretiliyor (yüzlerce sayfada bellek şişmesin).
            "_kaynak_yol": yol,
            "_koseler": (kart_sonuc.get("koseler").tolist()
                         if kart_sonuc.get("koseler") is not None else None),
            "_ad_conf": ad_conf,
            "_soyad_conf": soyad_conf,
            "_preview": kart if kart is not None else resim,
            "_debug_resmi": ocr_sonuc.get("debug_resmi"),
            "_kart_debug": kart_sonuc.get("debug_resmi"),
            "_kart_sonuc": {
                "belge_tipi": kart_sonuc.get("belge_tipi"),
                "kart_sirasi": kart_sonuc.get("kart_sirasi", sira),
                "kart_sayisi": kart_sonuc.get("kart_sayisi", kart_sayisi),
                "arama_modu": kart_sonuc.get("arama_modu"),
                "duzeltildi": kart_sonuc.get("duzeltildi", False),
                "duzeltme_yontemi": kart_sonuc.get("duzeltme_yontemi"),
                "fallback": kart_sonuc.get("fallback", False),
                "iyi_eslesme": kart_sonuc.get("iyi_eslesme", 0),
                "inlier": kart_sonuc.get("inlier", 0),
                "inlier_orani": kart_sonuc.get("inlier_orani", 0.0),
                "skor": kart_sonuc.get("skor", 0.0),
                "mesaj": kart_sonuc.get("mesaj", ""),
                "aday_sirasi": kart_sonuc.get("aday_sirasi"),
                "referans_hatalari": kart_sonuc.get("referans_hatalari", {}),
            },
            "_ocr_sonuc": {
                "guven": ocr_sonuc.get("guven"),
                "tc_dogrulandi": tc_dogrulandi,
                "tc_kaynak": ocr_sonuc.get("tc_kaynak"),
                "kimlik_no_conf": ocr_sonuc.get("kimlik_no_conf", 0.0),
                "ad_conf": ocr_sonuc.get("ad_conf", 0.0),
                "soyad_conf": ocr_sonuc.get("soyad_conf", 0.0),
                "baslangic_tarihi": ocr_sonuc.get("baslangic_tarihi", ""),
                "bitis_tarihi": ocr_sonuc.get("bitis_tarihi", ""),
                "gecerlilik_durumu": ocr_sonuc.get("gecerlilik_durumu"),
                "ocr_suresi": ocr_sonuc.get("ocr_suresi", 0.0),
                "hizli_yol_kullanildi": ocr_sonuc.get("hizli_yol_kullanildi"),
                "hizli_hucre_ocr_suresi": ocr_sonuc.get("hizli_hucre_ocr_suresi", 0.0),
                "detector_fallback_kullanildi": ocr_sonuc.get("detector_fallback_kullanildi"),
                "detector_ocr_suresi": ocr_sonuc.get("detector_ocr_suresi", 0.0),
                "fallback_denenen_alanlar": ocr_sonuc.get("fallback_denenen_alanlar", []),
                "fallback_kullanilan_alanlar": ocr_sonuc.get("fallback_kullanilan_alanlar", []),
                "fallback_ocr_suresi": ocr_sonuc.get("fallback_ocr_suresi", 0.0),
                "tum_ocr": ocr_sonuc.get("tum_ocr", []) if self.debug else [],
            },
        }


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kimlik Okuyucu — Masaüstü")
        self.resize(1280, 760)
        self.setMinimumSize(980, 620)

        self.dosyalar = []
        self.sonuclar = []
        self.canli_sonuclar = []
        self.pdf_bytes = None
        self.worker = None

        # Önizleme: zoom = None ise pencereye sığdırılır, sayı ise o oranda gösterilir.
        self.zoom = None
        self._sayfa_onbellek = {}
        self._isaretli_onbellek = (None, None)

        # Tablodaki sonuçların yazıldığı geçmiş kaydı. Düzenleme yapıldıkça
        # AYNI kayıt güncelleniyor; her düzenleme için yeni kayıt açılmıyor.
        self.gecmis_yolu = None
        self.gecmis_zamanlayici = QTimer(self)
        self.gecmis_zamanlayici.setSingleShot(True)
        self.gecmis_zamanlayici.setInterval(2500)
        self.gecmis_zamanlayici.timeout.connect(self.gecmisi_guncelle)

        # Fareyle seçilen alan ve o alanı okuyan iş parçacığı
        self._secim = None                # görüntü koordinatında (x0, y0, x1, y1)
        self._onizleme_ham = None         # işaretsiz görüntü (kırpma buradan alınır)
        self._onizleme_olcek = 1.0
        self._onizleme_ofset = (0, 0)     # gösterilen görüntünün sayfadaki yeri
        self.alan_okuyucu = None

        # "kart": satıra tıklanınca o kimliğe yakınlaşılır.
        # "sayfa": Sayfa sütununa tıklanınca sayfanın tamamı gösterilir.
        self._onizleme_kipi = "kart"

        self.edit_mode = False
        self._table_updating = False

        self.setStyleSheet("""
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
        """)

        root = QWidget()
        self.setCentralWidget(root)
        ana = QVBoxLayout(root)

        ust = QHBoxLayout()
        ust.setSpacing(10)

        self.sec_btn = QPushButton("Dosya Seç")
        self.klasor_btn = QPushButton("Klasör Seç")
        self.baslat_btn = QPushButton("İşlemi Başlat")
        self.baslat_btn.setEnabled(False)

        self.karsilastir_btn = QPushButton("Liste Karşılaştır")
        self.karsilastir_btn.setToolTip(
            "Dışarıdan gelen kimlik listesini (Excel) yükleyip tablodaki sonuçlarla "
            "karşılaştırır: kaç kayıt tuttu, hangileri tutmadı, hangi kimlikler eksik."
        )
        self.karsilastir_btn.setEnabled(False)

        self.gecmis_btn = QPushButton("Geçmiş")
        self.gecmis_btn.setToolTip(
            "Yakın zamanlı taramalar burada saklanır; eskisini açıp Excel/PDF alabilirsiniz."
        )

        self.edit_btn = QPushButton("Düzenle")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setEnabled(False)

        # Tespitin kaçırdığı bir kimliği elle eklemek / fazladan satırı silmek için.
        self.satir_ekle_btn = QPushButton("＋ Satır")
        self.satir_ekle_btn.setToolTip(
            "Seçili satırın sayfasına, elle doldurulacak boş bir kimlik satırı ekler."
        )
        self.satir_sil_btn = QPushButton("－ Satır")
        self.satir_sil_btn.setToolTip("Seçili satırı listeden siler.")
        self.satir_ekle_btn.setEnabled(False)
        self.satir_sil_btn.setEnabled(False)

        self.coklu_cb = QCheckBox("Çok kimlikli sayfa")
        self.coklu_cb.setChecked(True)
        self.coklu_cb.setToolTip(
            "Açıkken sayfanın tamamı taranır ve satır seçilince sayfa, bulunan\n"
            "kimlikler işaretli halde gösterilir. Her sayfada tek kimlik olduğu\n"
            "biliniyorsa kapatmak işlemi hızlandırır."
        )

        self.derin_cb = QCheckBox("Derin okuma")
        self.derin_cb.setChecked(True)
        self.derin_cb.setToolTip(
            "Kimlik numarası ilk denemede okunamazsa ek okuma geçişleri yapar. "
            "Solgun fotokopilerde okunan numara sayısını belirgin arttırır; "
            "kapatılırsa işlem yaklaşık üçte bir oranında hızlanır."
        )

        self.kurtar_cb = QCheckBox("Kurtarma")
        self.kurtar_cb.setChecked(True)
        self.kurtar_cb.setToolTip(
            "Kart olarak hizalanamayan blokları da doğrudan okur; kimliğin sessizce "
            "atlanmasını önler. Kapatılırsa işlem bir miktar hızlanır."
        )
        self.debug_cb = QCheckBox("Debug")
        self.debug_cb.setToolTip("Seçilen satırın belge tespit / OCR ayrıntılarını gösterir.")

        for btn in (
            self.sec_btn,
            self.klasor_btn,
            self.baslat_btn,
            self.gecmis_btn,
            self.karsilastir_btn,
            self.edit_btn,
            self.satir_ekle_btn,
            self.satir_sil_btn,
        ):
            btn.setMinimumHeight(42)

        for btn in (self.satir_ekle_btn, self.satir_sil_btn):
            btn.setMaximumWidth(110)

        ust.addWidget(self.sec_btn)
        ust.addWidget(self.klasor_btn)
        ust.addWidget(self.baslat_btn)
        ust.addWidget(self.gecmis_btn)
        ust.addWidget(self.karsilastir_btn)
        ust.addWidget(self.edit_btn)
        ust.addWidget(self.satir_ekle_btn)
        ust.addWidget(self.satir_sil_btn)
        ust.addStretch(1)
        ust.addWidget(self.coklu_cb)
        ust.addWidget(self.derin_cb)
        ust.addWidget(self.kurtar_cb)
        ust.addWidget(self.debug_cb)
        ana.addLayout(ust)

        self.bilgi = QLabel("Dosya seçilmedi.")
        self.bilgi.setWordWrap(True)
        ana.addWidget(self.bilgi)

        # Okunamayan alanların sayısı buraya yazılır
        # ("3 Kimlik No bulunamadı, 1 Ad bulunamadı..."), satırlar
        # geldikçe canlı güncellenir.
        self.ozet = QLabel("")
        self.ozet.setWordWrap(True)
        self.ozet.setStyleSheet("color: #cbd5e1; font-size: 13px;")
        ana.addWidget(self.ozet)

        self.progress = QProgressBar()
        ana.addWidget(self.progress)

        # Kaydetme butonlarını ana içerikten bağımsız ve her zaman görünür tut.
        kaydet_bar = QHBoxLayout()
        kaydet_bar.setSpacing(10)
        kaydet_bar.addStretch(1)

        self.excel_btn = QPushButton("Excel’e Çevir ve Aç")
        self.pdf_btn = QPushButton("PDF’e Çevir ve Aç")
        self.excel_btn.setEnabled(False)
        self.pdf_btn.setEnabled(False)
        self.excel_btn.setMinimumHeight(42)
        self.pdf_btn.setMinimumHeight(42)

        kaydet_bar.addWidget(self.excel_btn)
        kaydet_bar.addWidget(self.pdf_btn)
        ana.addLayout(kaydet_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        ana.addWidget(splitter, 1)

        self.table = QTableWidget()
        self.headers = [
            "Dosya", "Sayfa", "Kart", "Belge Türü", "Kimlik No", "Ad", "Soyad",
            "Bitiş Tarihi", "Geçerlilik", "Durum", "Süre",
        ]
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setDefaultSectionSize(32)

        header = self.table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setStretchLastSection(False)

        for idx in range(len(self.headers)):
            header.setSectionResizeMode(idx, QHeaderView.Interactive)

        genislikler = [190, 65, 60, 145, 145, 175, 175, 120, 115, 230, 70]
        for idx, width in enumerate(genislikler):
            self.table.setColumnWidth(idx, width)

        splitter.addWidget(self.table)

        sag = QWidget()
        saglay = QVBoxLayout(sag)
        saglay.setContentsMargins(6, 0, 0, 0)

        self.sag_splitter = QSplitter(Qt.Vertical)
        self.sag_splitter.setChildrenCollapsible(False)

        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        # Sayfada kaç kimlik bulunduğu + zoom kontrolleri
        onizleme_bar = QHBoxLayout()
        onizleme_bar.setSpacing(6)

        # Tek satır kalsın: sarmalanırsa panelin yarısını kaplayıp görüntüye
        # yer bırakmıyordu. Sığmayan kısım "…" ile kısaltılıyor.
        self.sayfa_bilgi = QLabel("")
        self.sayfa_bilgi.setWordWrap(False)
        self.sayfa_bilgi.setMinimumWidth(80)
        self.sayfa_bilgi.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.sayfa_bilgi.setStyleSheet("color: #dce1e7; font-size: 13px; font-weight: 600;")
        self._sayfa_bilgi_metni = ""

        self.alan_oku_btn = QPushButton("🔍 Seçili alanı oku")
        self.alan_oku_btn.setEnabled(False)
        self.alan_oku_btn.setToolTip(
            "İşaretlenmemiş bir kimliği fareyle çerçeveleyin, sonra bu düğmeye basın."
        )
        self.alan_oku_btn.setMinimumHeight(30)
        self.alan_oku_btn.setMaximumHeight(30)

        self.zoom_azalt_btn = QPushButton("−")
        self.zoom_arttir_btn = QPushButton("+")
        self.zoom_sigdir_btn = QPushButton("Sığdır")
        for btn in (self.zoom_azalt_btn, self.zoom_arttir_btn, self.zoom_sigdir_btn):
            btn.setMinimumHeight(30)
            btn.setMaximumHeight(30)
        self.zoom_azalt_btn.setMaximumWidth(38)
        self.zoom_arttir_btn.setMaximumWidth(38)
        self.zoom_sigdir_btn.setMaximumWidth(78)

        # Bilgi yazısı kendi satırında: düğmelerle aynı satırda sıkışıp
        # "Sayfa 2 — 3…" gibi kesiliyordu.
        preview_layout.addWidget(self.sayfa_bilgi)

        onizleme_bar.addWidget(self.alan_oku_btn)
        onizleme_bar.addStretch(1)
        onizleme_bar.addWidget(self.zoom_azalt_btn)
        onizleme_bar.addWidget(self.zoom_arttir_btn)
        onizleme_bar.addWidget(self.zoom_sigdir_btn)
        preview_layout.addLayout(onizleme_bar)

        self.preview = OnizlemeEtiketi("Bir satır seçince kimlik burada gösterilecek.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 260)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet(
            "border: 1px solid #454b53; background: #111316; color: #dce1e7; border-radius: 7px;"
        )

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFrameShape(QFrame.NoFrame)
        self.preview_scroll.setWidget(self.preview)
        self.preview_scroll.viewport().installEventFilter(self)
        preview_layout.addWidget(self.preview_scroll, 1)

        # Uzun açıklama artık başlıkta değil; görüntünün altında tek satır ipucu.
        self.onizleme_ipucu = QLabel("")
        self.onizleme_ipucu.setWordWrap(False)
        self.onizleme_ipucu.setMinimumWidth(80)
        self.onizleme_ipucu.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.onizleme_ipucu.setStyleSheet("color: #9aa3ad; font-size: 12px;")
        self._onizleme_ipucu_metni = ""
        preview_layout.addWidget(self.onizleme_ipucu)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setPlaceholderText(
            "Debug açıksa seçilen satırın belge tespit / OCR ayrıntıları burada görünür."
        )
        self.debug_text.setMinimumHeight(140)

        self.sag_splitter.addWidget(preview_frame)
        self.sag_splitter.addWidget(self.debug_text)

        # Debug kapalı başlar; böylece kimlik önizlemesi bütün sağ paneli kullanır.
        self.debug_text.hide()
        self.sag_splitter.setSizes([1, 0])

        saglay.addWidget(self.sag_splitter, 1)
        splitter.addWidget(sag)
        # Sağ panel artık sayfa görüntüsünü de gösteriyor; biraz daha geniş başlasın.
        splitter.setSizes([780, 540])

        self.sec_btn.clicked.connect(self.dosya_sec)
        self.klasor_btn.clicked.connect(self.klasor_sec)
        self.baslat_btn.clicked.connect(self.baslat)
        self.edit_btn.toggled.connect(self.duzenleme_modu_degisti)
        self.debug_cb.toggled.connect(self.debug_gorunumu_degisti)
        self.table.itemSelectionChanged.connect(self.onizleme_goster)
        self.table.cellClicked.connect(self.hucreye_tiklandi)
        self.table.cellChanged.connect(self.hucre_degisti)
        self.excel_btn.clicked.connect(self.excel_kaydet)
        self.pdf_btn.clicked.connect(self.pdf_kaydet)
        self.gecmis_btn.clicked.connect(self.gecmisi_goster)
        self.karsilastir_btn.clicked.connect(self.listeyle_karsilastir)
        self.satir_ekle_btn.clicked.connect(self.elle_satir_ekle)
        self.satir_sil_btn.clicked.connect(self.satiri_sil)
        self.zoom_arttir_btn.clicked.connect(lambda: self.zoom_degistir(1.25))
        self.zoom_azalt_btn.clicked.connect(lambda: self.zoom_degistir(0.8))
        self.zoom_sigdir_btn.clicked.connect(self.zoom_sigdir)
        self.coklu_cb.toggled.connect(lambda _: self.onizleme_goster())
        self.preview.alan_secildi.connect(self.alan_secildi)
        self.alan_oku_btn.clicked.connect(self.secili_alani_oku)

    def dosya_sec(self):
        yollar, _ = QFileDialog.getOpenFileNames(
            self, "Kimlik dosyalarını seç", "",
            "Kimlik / PDF / Excel (*.jpg *.jpeg *.png *.pdf *.xlsx *.xlsm)"
        )
        if yollar:
            self.dosyalar = yollar
            self.bilgi.setText(f"{len(yollar)} dosya seçildi.")
            self.baslat_btn.setEnabled(True)

    def klasor_sec(self):
        klasor = QFileDialog.getExistingDirectory(self, "Klasör seç")
        if not klasor:
            return
        yollar = []
        for kok, _, dosyalar in os.walk(klasor):
            for ad in dosyalar:
                yol = os.path.join(kok, ad)
                if os.path.splitext(ad)[1].lower() in DESTEKLENEN:
                    yollar.append(yol)
        self.dosyalar = sorted(yollar)
        self.bilgi.setText(f"{len(yollar)} dosya bulundu.")
        self.baslat_btn.setEnabled(bool(yollar))

    def baslat(self):
        if not self.dosyalar:
            return
        self.gecmisi_guncelle()      # önceki taramanın bekleyen düzenlemesi kaybolmasın

        self.table.setRowCount(0)
        self.sonuclar = []
        self.canli_sonuclar = []
        self.ozet.setText("")
        self.gecmis_yolu = None
        self.pdf_bytes = None
        self.excel_btn.setEnabled(False)
        self.pdf_btn.setEnabled(False)
        self.edit_btn.setChecked(False)
        self.edit_btn.setEnabled(False)
        self.baslat_btn.setEnabled(False)
        self.karsilastir_btn.setEnabled(False)
        self.progress.setValue(0)
        self.debug_text.clear()

        self.satir_ekle_btn.setEnabled(False)
        self.satir_sil_btn.setEnabled(False)
        self._sayfa_onbellek = {}

        self.worker = Worker(
            self.dosyalar,
            self.debug_cb.isChecked(),
            self.kurtar_cb.isChecked(),
            self.coklu_cb.isChecked(),
            self.derin_cb.isChecked(),
        )
        self.worker.progress.connect(self.progress_guncelle)
        self.worker.row_ready.connect(self.satir_ekle)
        self.worker.finished_ok.connect(self.bitti)
        self.worker.failed.connect(self.hata)
        self.worker.start()

    def progress_guncelle(self, n, toplam, mesaj):
        self.progress.setMaximum(max(1, toplam))
        self.progress.setValue(n)
        self.bilgi.setText(f"{mesaj} — {n}/{toplam}")

    def satir_ekle(self, row):
        # Geçersiz belgelerde satırı sarıya boyamıyoruz (koyu temada okunmuyordu);
        # renk kuralları tabloya_satir_yaz içinde toplandı.
        self.canli_sonuclar.append(row)
        self._table_updating = True
        try:
            self.tabloya_satir_yaz(row)
        finally:
            self._table_updating = False

        self.ozet_guncelle(self.canli_sonuclar)


    def eksik_sayilari(self, satirlar):
        """Okunamayan alanları sayar: (alan -> adet, kartı hiç bulunamayan sayfa
        sayısı). Bitiş tarihi yalnız göçmen belgelerinde beklenir."""
        sayac = {alan: 0 for alan in ("Kimlik No", "Ad", "Soyad", "Bitiş Tarihi")}
        kart_bulunamayan = 0
        supheli = 0

        for r in satirlar:
            if not r.get("_kart_bulundu", True):
                kart_bulunamayan += 1
                continue
            if r.get("_tc_supheli"):
                supheli += 1
            for alan in ("Kimlik No", "Ad", "Soyad"):
                if str(r.get(alan, "")).strip() in ("", "-", "Bulunamadi"):
                    sayac[alan] += 1
            if r.get("_belge_tipi") == "gocmen":
                if str(r.get("Bitiş Tarihi", "")).strip() in ("", "-"):
                    sayac["Bitiş Tarihi"] += 1

        return sayac, kart_bulunamayan, supheli

    def ozet_guncelle(self, satirlar):
        """Bulunamayan alanların özetini bilgi çubuğunun altına yazar."""
        if not satirlar:
            self.ozet.setText("")
            return

        sayac, kart_bulunamayan, supheli = self.eksik_sayilari(satirlar)

        parcalar = []
        if kart_bulunamayan:
            parcalar.append(f"{kart_bulunamayan} sayfada kimlik bulunamadı")
        parcalar.extend(
            f"{adet} {alan} bulunamadı" for alan, adet in sayac.items() if adet
        )
        if supheli:
            parcalar.append(f"{supheli} Kimlik No doğrulanamadı")

        if parcalar:
            self.ozet.setText("Bulunamayanlar →  " + "   ·   ".join(parcalar))
            self.ozet.setStyleSheet("color: #f0b429; font-size: 13px; font-weight: 600;")
        else:
            okunan = len(satirlar) - kart_bulunamayan
            self.ozet.setText(
                f"Bulunamayan alan yok — {okunan} kimliğin tüm alanları okundu."
            )
            self.ozet.setStyleSheet("color: #86efac; font-size: 13px; font-weight: 600;")

    def debug_gorunumu_degisti(self, aktif):
        """
        Debug alanını yalnız checkbox açıkken göster.
        Kapalıyken sağ panelin tamamını kimlik önizlemesine bırak.
        """
        if aktif:
            self.debug_text.show()
            self.sag_splitter.setSizes([620, 240])
        else:
            self.debug_text.hide()
            self.sag_splitter.setSizes([1, 0])

        # Seçili kimliği yeni kullanılabilir alana yeniden sığdır.
        self.onizleme_goster()

    def duzenleme_modu_degisti(self, aktif):
        self.edit_mode = bool(aktif)

        if self.edit_mode:
            self.edit_btn.setText("Düzenlemeyi Bitir")
            self.table.setEditTriggers(
                QAbstractItemView.DoubleClicked
                | QAbstractItemView.SelectedClicked
                | QAbstractItemView.EditKeyPressed
            )
            self.bilgi.setText(
                "Düzenleme modu açık — Kimlik No, Ad ve Soyad hücrelerini değiştirebilirsin."
            )
        else:
            self.edit_btn.setText("Düzenle")
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        editable_headers = {"Kimlik No", "Ad", "Soyad"}

        self._table_updating = True
        try:
            for r in range(self.table.rowCount()):
                for c, baslik in enumerate(self.headers):
                    item = self.table.item(r, c)
                    if item is None:
                        continue

                    if self.edit_mode and baslik in editable_headers:
                        item.setFlags(item.flags() | Qt.ItemIsEditable)
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        finally:
            self._table_updating = False

    def hucre_degisti(self, row, column):
        if self._table_updating:
            return

        if row < 0 or row >= len(self.sonuclar):
            return

        baslik = self.headers[column]
        if baslik not in {"Kimlik No", "Ad", "Soyad"}:
            return

        item = self.table.item(row, column)
        if item is None:
            return

        yeni_deger = item.text().strip() or "Bulunamadi"

        # Ana veri modelini de güncelle:
        # Excel kaydı self.sonuclar'dan üretildiği için değişiklik doğrudan yansır.
        self.sonuclar[row][baslik] = yeni_deger
        self.sonuclar[row]["_manuel_duzenlendi"] = True
        # Hangi alanların elle değiştirildiği saklanıyor ki tablo yeniden
        # kurulduğunda (sıralama değişince) vurgu kaybolmasın.
        duzenlenenler = set(self.sonuclar[row].get("_duzenlenen_alanlar") or [])
        duzenlenenler.add(baslik)
        self.sonuclar[row]["_duzenlenen_alanlar"] = sorted(duzenlenenler)
        # Durum yazısı OCR sonucunu anlatıyordu; elle doldurulan alandan sonra
        # "Eksik alan: Ad" gibi yanıltıcı kalıyordu. Yeniden hesaplanıyor.
        satir = self.sonuclar[row]
        satir["_tc_supheli"] = False
        eksikler = [
            alan for alan in ("Kimlik No", "Ad", "Soyad")
            if str(satir.get(alan, "")).strip() in ("", "Bulunamadi")
        ]
        onek = "Elle eklendi" if satir.get("_manuel_eklendi") else "Elle düzeltildi"
        satir["Durum"] = onek if not eksikler else f"{onek} — eksik: " + ", ".join(eksikler)

        self._table_updating = True
        try:
            durum_item = self.table.item(row, self.headers.index("Durum"))
            if durum_item is not None:
                durum_item.setText(satir["Durum"])
                durum_item.setForeground(QColor("#93c5fd"))
        finally:
            self._table_updating = False

        self.duzenlenmis_hucreyi_boya(item)
        self.ozet_guncelle(self.sonuclar)
        self.gecmisi_guncellemeyi_planla()

        if self.table.currentRow() == row:
            self.onizleme_goster()

    def bitti(self, sonuclar, pdf_bytes, durduruldu=False):
        self.sonuclar = sonuclar
        self.pdf_bytes = pdf_bytes
        self.baslat_btn.setEnabled(True)
        self.excel_btn.setEnabled(bool(sonuclar))
        self.pdf_btn.setEnabled(bool(sonuclar))
        self.edit_btn.setEnabled(bool(sonuclar))
        self.satir_ekle_btn.setEnabled(bool(sonuclar))
        self.satir_sil_btn.setEnabled(bool(sonuclar))
        self.karsilastir_btn.setEnabled(bool(sonuclar))
        self.canli_sonuclar = list(sonuclar)

        # Bir sayfada birden fazla kimlik olabildiği için satır sayısı ile
        # sayfa sayısı artık aynı değil; ikisini de ayrı ayrı gösteriyoruz.
        kimlik_sayisi = sum(1 for r in sonuclar if r.get("_kart_bulundu"))
        kurtarilan = sum(1 for r in sonuclar if r.get("_kurtarildi"))
        sayfa_sayisi = len({(r.get("Dosya"), r.get("Sayfa")) for r in sonuclar})
        kurtarma_yazi = f" ({kurtarilan} tanesi kurtarıldı)" if kurtarilan else ""

        gecmis_yazi = self.gecmise_kaydet(sonuclar)

        if durduruldu:
            self.bilgi.setText(
                f"⏹ Durduruldu — {sayfa_sayisi} sayfada {kimlik_sayisi} kimlik okundu"
                f"{kurtarma_yazi} (kalanlar işlenmedi).{gecmis_yazi}"
            )
        else:
            self.bilgi.setText(
                f"Tamamlandı — {sayfa_sayisi} sayfada {kimlik_sayisi} kimlik okundu"
                f"{kurtarma_yazi}.{gecmis_yazi}"
            )

        self.ozet_guncelle(sonuclar)
        if self.table.rowCount():
            self.table.selectRow(0)

    def gecmise_kaydet(self, sonuclar):
        """Tamamlanan taramayı kendiliğinden geçmişe yazar."""
        if not sonuclar:
            return ""
        try:
            self.gecmis_yolu = gecmis.taramayi_kaydet(
                sonuclar, kaynak_dosyalar=self.dosyalar
            )
        except Exception as exc:
            print(f"[GEÇMİŞ] kaydedilemedi: {exc!r}", flush=True)
            self.gecmis_yolu = None
            return "  (geçmişe kaydedilemedi)"
        return "  •  geçmişe kaydedildi" if self.gecmis_yolu else ""

    def gecmisi_guncellemeyi_planla(self):
        """Düzenleme sonrası geçmiş kaydını tazeler.

        Doğrudan yazmak yerine kısa bir gecikme kullanılıyor: hücre hücre
        düzenleme yapılırken her tuş için dosya yeniden yazılmasın, art arda
        gelen değişiklikler tek yazmada toplansın."""
        if self.gecmis_yolu and self.sonuclar:
            self.gecmis_zamanlayici.start()

    def gecmisi_guncelle(self):
        """Tablodaki güncel hali, açık olan geçmiş kaydının üzerine yazar."""
        self.gecmis_zamanlayici.stop()
        if not self.gecmis_yolu or not self.sonuclar:
            return
        try:
            # kaynak_dosyalar geçilmiyor: self.dosyalar kullanıcının BİR SONRAKİ
            # tarama için seçtiği dosyaları tutuyor olabilir; kaydın kendi dosya
            # listesi korunmalı.
            gecmis.taramayi_kaydet(self.sonuclar, yol=self.gecmis_yolu)
        except Exception as exc:
            print(f"[GEÇMİŞ] güncellenemedi: {exc!r}", flush=True)

    def listeyle_karsilastir(self):
        """Dışarıdan gelen listeyi yükleyip tablodaki sonuçlarla karşılaştırır."""
        if not self.sonuclar:
            return

        yol, _ = QFileDialog.getOpenFileName(
            self, "Karşılaştırılacak listeyi seç", "", "Excel (*.xlsx *.xlsm)"
        )
        if not yol:
            return

        try:
            kayitlar, bilgi = karsilastirma.listeyi_oku(yol)
        except Exception as exc:
            QMessageBox.warning(self, "Liste okunamadı", str(exc))
            return

        sonuc = karsilastirma.karsilastir(self.sonuclar, kayitlar)
        self.bilgi.setText(karsilastirma.ozet_metni(sonuc, os.path.basename(yol)))

        pencere = KarsilastirmaPenceresi(sonuc, bilgi, os.path.basename(yol), self)
        pencere.exec()

    def gecmisi_goster(self):
        pencere = GecmisPenceresi(self)
        if pencere.exec() == QDialog.Accepted and pencere.secilen_yol:
            self.gecmis_kaydini_ac(pencere.secilen_yol)

    def gecmis_kaydini_ac(self, yol):
        """Kayıtlı bir taramayı tabloya yükler."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self, "İşlem sürüyor",
                "Önce mevcut işlemin bitmesini bekleyin.",
            )
            return

        try:
            satirlar = gecmis.taramayi_yukle(yol)
        except Exception as exc:
            QMessageBox.critical(self, "Geçmiş açılamadı", f"{type(exc).__name__}: {exc}")
            return

        self.gecmisi_guncelle()      # açık kayıtta bekleyen değişiklik varsa önce onu yaz

        self.table.setRowCount(0)
        self.sonuclar = satirlar
        self.canli_sonuclar = list(satirlar)
        self.gecmis_yolu = yol
        self.pdf_bytes = None
        self._sayfa_onbellek = {}
        self._isaretli_onbellek = (None, None)

        self.tabloyu_yenile()
        self.ozet_guncelle(self.sonuclar)

        kimlik_sayisi = sum(1 for r in satirlar if r.get("_kart_bulundu"))
        sayfa_sayisi = len({(r.get("Dosya"), r.get("Sayfa")) for r in satirlar})
        self.bilgi.setText(
            f"Geçmişten açıldı — {sayfa_sayisi} sayfada {kimlik_sayisi} kimlik. "
            "Excel/PDF alabilir, satır ekleyip düzenleyebilirsiniz."
        )

        self.excel_btn.setEnabled(bool(satirlar))
        self.pdf_btn.setEnabled(bool(satirlar))
        self.edit_btn.setEnabled(bool(satirlar))
        self.satir_ekle_btn.setEnabled(bool(satirlar))
        self.satir_sil_btn.setEnabled(bool(satirlar))
        self.karsilastir_btn.setEnabled(bool(satirlar))
        if self.table.rowCount():
            self.table.selectRow(0)

    def hata(self, mesaj):
        self.baslat_btn.setEnabled(True)
        QMessageBox.critical(self, "Hata", mesaj)

    def sayfa_goruntusu_getir(self, yol, sayfa_no):
        """Sayfanın tam görüntüsünü kaynak dosyadan üretir.

        Yüzlerce sayfanın görüntüsünü bellekte tutmak yerine, satır seçildiğinde
        yeniden okunuyor; son birkaç sayfa küçük bir önbellekte tutuluyor."""
        if not yol or not os.path.exists(yol):
            return None

        anahtar = (yol, sayfa_no)
        if anahtar in self._sayfa_onbellek:
            return self._sayfa_onbellek[anahtar]

        resim = None
        try:
            if excel_kaynak.excel_mi(yol):
                resim = excel_kaynak.bant_goruntusu(yol, sayfa_no)
            elif yol.lower().endswith(".pdf"):
                with pymupdf.open(yol) as doc:
                    indeks = (sayfa_no or 1) - 1
                    if 0 <= indeks < doc.page_count:
                        resim = pixmap_to_bgr(
                            doc[indeks].get_pixmap(dpi=PDF_RENDER_DPI, alpha=False)
                        )
            else:
                for _, img, _, _ in dosyayi_goruntulere_ayir(yol):
                    resim = img
                    break
        except Exception:
            return None

        if resim is not None:
            # Küçük bir FIFO önbellek: art arda aynı sayfanın satırları gezilirken
            # dosya tekrar tekrar açılmasın.
            if len(self._sayfa_onbellek) >= 3:
                self._sayfa_onbellek.pop(next(iter(self._sayfa_onbellek)))
            self._sayfa_onbellek[anahtar] = resim
        return resim

    def sayfanin_satirlari(self, sonuc):
        """Aynı dosya + sayfaya ait bütün satırlar (tablodaki sırayla)."""
        anahtar = (sonuc.get("Dosya"), sonuc.get("Sayfa"))
        return [r for r in self.sonuclar if (r.get("Dosya"), r.get("Sayfa")) == anahtar]

    def sayfayi_isaretle(self, sayfa_resmi, satirlar, secili):
        """Bulunan kimliklerin sayfa üzerindeki yerlerini numaralandırarak çizer."""
        img = sayfa_resmi.copy()
        h, w = img.shape[:2]
        kalinlik = max(2, int(min(h, w) * 0.004))
        yazi_olcek = max(0.8, min(h, w) / 900.0)

        for sira, satir in enumerate(satirlar, start=1):
            koseler = satir.get("_koseler")
            if not koseler:
                continue

            secili_mi = satir is secili
            if satir.get("_manuel_eklendi"):
                renk = (255, 170, 60)
            elif satir.get("_kurtarildi"):
                renk = (0, 170, 255)
            else:
                renk = (80, 220, 80)

            poly = np.asarray(koseler, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], True, renk,
                          kalinlik * (2 if secili_mi else 1), cv2.LINE_AA)

            x, y = poly[0][0]
            etiket = str(sira)
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, yazi_olcek, 2)
            x = int(max(0, min(x, w - tw - 12)))
            y = int(max(th + 12, min(y, h - 6)))
            cv2.rectangle(img, (x, y - th - 10), (x + tw + 12, y + 4), renk, -1)
            cv2.putText(img, etiket, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        yazi_olcek, (20, 20, 20), 2, cv2.LINE_AA)

        return img

    def onizleme_gorselleri(self, sonuc):
        """(gösterilecek görüntü, işaretsiz görüntü, sayfadaki ofset) üçlüsü.

        İkisi de AYNI kırpmadan gelir: ekranda işaretli hali görünür, fareyle
        seçilen alan ise işaretsiz halden kırpılır (yeşil çerçeveler OCR'a
        karışmasın). Ofset, seçimi tekrar sayfa koordinatına çevirmek için."""
        if self.coklu_cb.isChecked():
            sayfa = self.sayfa_goruntusu_getir(
                sonuc.get("_kaynak_yol"), sonuc.get("Sayfa")
            )
            if sayfa is not None:
                satirlar = self.sayfanin_satirlari(sonuc)
                anahtar = (sonuc.get("_kaynak_yol"), sonuc.get("Sayfa"), id(sonuc),
                           tuple(id(r) for r in satirlar), self._onizleme_kipi)
                if self._isaretli_onbellek[0] == anahtar:
                    return self._isaretli_onbellek[1]

                isaretli = self.sayfayi_isaretle(sayfa, satirlar, sonuc)
                ham = sayfa
                ofset = (0, 0)

                # Satıra tıklanmışsa o kimliğe yakınlaş; Sayfa sütununa
                # tıklanmışsa sayfanın tamamı kalsın.
                if self._onizleme_kipi == "kart" and sonuc.get("_koseler"):
                    isaretli, ofset = self.karta_yakinlas(isaretli, sonuc["_koseler"])
                    ham, _ = self.karta_yakinlas(sayfa, sonuc["_koseler"])

                gorseller = (isaretli, ham, ofset)
                self._isaretli_onbellek = (anahtar, gorseller)
                return gorseller

        if self.debug_cb.isChecked():
            for anahtar in ("_debug_resmi", "_kart_debug"):
                if sonuc.get(anahtar) is not None:
                    return sonuc[anahtar], sonuc[anahtar], (0, 0)
        onizleme = sonuc.get("_preview")
        return onizleme, onizleme, (0, 0)

    def etiketi_sigdir(self, etiket, tam_metin):
        """Tek satırlık etikete sığmayan metni "…" ile kısaltarak yazar."""
        genislik = max(60, etiket.width() - 4)
        etiket.setText(etiket.fontMetrics().elidedText(tam_metin, Qt.ElideRight, genislik))
        etiket.setToolTip(tam_metin if tam_metin else "")

    def onizleme_yazilarini_tazele(self):
        self.etiketi_sigdir(self.sayfa_bilgi, self._sayfa_bilgi_metni)
        self.etiketi_sigdir(self.onizleme_ipucu, self._onizleme_ipucu_metni)

    def sayfa_bilgisini_guncelle(self, sonuc):
        if not self.coklu_cb.isChecked():
            self._sayfa_bilgi_metni = ""
            self._onizleme_ipucu_metni = ""
            self.onizleme_yazilarini_tazele()
            return

        satirlar = self.sayfanin_satirlari(sonuc)
        bulunan = sum(1 for r in satirlar if r.get("_kart_bulundu"))
        elle = sum(1 for r in satirlar if r.get("_manuel_eklendi"))
        kurtarilan = sum(1 for r in satirlar if r.get("_kurtarildi"))

        sayfa_yazi = f"Sayfa {sonuc.get('Sayfa')}" if sonuc.get("Sayfa") != "-" else "Görüntü"
        metin = f"{sayfa_yazi} — {bulunan} kimlik" if bulunan else f"{sayfa_yazi} — kimlik yok"

        notlar = []
        if kurtarilan:
            notlar.append(f"{kurtarilan} kurtarıldı")
        if elle:
            notlar.append(f"{elle} elle eklendi")
        if notlar:
            metin += " (" + ", ".join(notlar) + ")"

        if self._onizleme_kipi == "kart" and sonuc.get("_koseler"):
            metin += "  •  yakınlaştırıldı"
        self._sayfa_bilgi_metni = metin
        self._onizleme_ipucu_metni = (
            "Sayfanın tamamı için Sayfa sütununa tıklayın.  "
            "İşaretlenmemiş kimliği fareyle çerçeveleyip “Seçili alanı oku” ile okutun."
        )
        self.onizleme_yazilarini_tazele()

    def pixmapi_yerlestir(self, img, ham=None, ofset=(0, 0)):
        self.preview.secimi_temizle()
        self._secim = None
        self.alan_oku_btn.setEnabled(False)
        self._onizleme_ham = ham if ham is not None else img
        self._onizleme_ofset = ofset

        if img is None:
            self.preview.clear()
            self.preview.setMinimumSize(320, 220)
            self.preview.setText("Görüntü bulunamadı.")
            return

        if self.zoom is None:
            alan_w = max(260, self.preview_scroll.viewport().width() - 8)
            alan_h = max(180, self.preview_scroll.viewport().height() - 8)
            self.preview.setMinimumSize(1, 1)
            pixmap = bgr_to_pixmap(img, max_w=alan_w, max_h=alan_h)
        else:
            h, w = img.shape[:2]
            # Ölçeklenmiş görüntü belleği patlatmasın: kenar en fazla 6000 piksel.
            azami_olcek = 6000.0 / max(h, w)
            self.zoom = min(self.zoom, azami_olcek)
            pixmap = bgr_to_pixmap(img, max_w=int(w * self.zoom), max_h=int(h * self.zoom))
            self.preview.setMinimumSize(pixmap.size())

        self.preview.setPixmap(pixmap)
        self.preview.setAlignment(Qt.AlignCenter)
        # Pixmap ile görüntü arasındaki oran: seçimi görüntü koordinatına çevirmek için.
        self._onizleme_olcek = pixmap.width() / float(img.shape[1]) if img.shape[1] else 1.0

    def alan_secildi(self, kutu):
        """Önizlemede fareyle çizilen dikdörtgeni görüntü koordinatına çevirir."""
        if kutu is None or self._onizleme_ham is None or self._onizleme_olcek <= 0:
            self._secim = None
            self.alan_oku_btn.setEnabled(False)
            return

        h, w = self._onizleme_ham.shape[:2]
        olcek = self._onizleme_olcek
        x0 = max(0, min(w - 1, int(kutu.left() / olcek)))
        y0 = max(0, min(h - 1, int(kutu.top() / olcek)))
        x1 = max(0, min(w, int(kutu.right() / olcek)))
        y1 = max(0, min(h, int(kutu.bottom() / olcek)))

        if x1 - x0 < 40 or y1 - y0 < 25:
            self._secim = None
            self.alan_oku_btn.setEnabled(False)
            self._sayfa_bilgi_metni = "Seçilen alan çok küçük — kimliği tamamen çerçeveleyin."
            self.onizleme_yazilarini_tazele()
            return

        self._secim = (x0, y0, x1, y1)
        self.alan_oku_btn.setEnabled(self.alan_okuyucu is None)

    def secili_alani_oku(self):
        """Kullanıcının çerçevelediği alanı okuyup yeni bir satır olarak ekler."""
        if self._secim is None or self._onizleme_ham is None:
            return
        if self.alan_okuyucu is not None:
            return

        r = self.table.currentRow()
        if r < 0 or r >= len(self.sonuclar):
            return

        x0, y0, x1, y1 = self._secim
        kirpma = np.ascontiguousarray(self._onizleme_ham[y0:y1, x0:x1])
        if kirpma.size == 0:
            return

        self._okunan_satir = r
        self._okunan_kutu = self._secim
        self._okunan_ofset = self._onizleme_ofset
        self.alan_oku_btn.setEnabled(False)
        self.alan_oku_btn.setText("Okunuyor…")
        self._sayfa_bilgi_metni = "Seçilen alan okunuyor…"
        self.onizleme_yazilarini_tazele()

        self.alan_okuyucu = AlanOkuyucu(kirpma, debug=self.debug_cb.isChecked())
        self.alan_okuyucu.bitti.connect(self.alan_okundu)
        self.alan_okuyucu.hata.connect(self.alan_okuma_hatasi)
        self.alan_okuyucu.finished.connect(self.alan_okuma_bitti)
        self.alan_okuyucu.start()

    def alan_okuma_bitti(self):
        self.alan_okuyucu = None
        self.alan_oku_btn.setText("🔍 Seçili alanı oku")
        self.alan_oku_btn.setEnabled(self._secim is not None)

    def alan_okuma_hatasi(self, mesaj):
        QMessageBox.warning(self, "Alan okunamadı", mesaj)

    def alan_okundu(self, sonuc):
        """Seçili alandan okunanları yeni bir satır olarak tabloya ekler."""
        kaynak = self.sonuclar[self._okunan_satir]
        ocr_sonuc = sonuc.get("ocr", {}) or {}
        # Seçim, ekranda gösterilen (belki kırpılmış) görüntünün koordinatındaydı;
        # işaretlemenin sayfada doğru yere düşmesi için ofset geri ekleniyor.
        ox, oy = self._okunan_ofset
        x0, y0, x1, y1 = self._okunan_kutu
        x0, y0, x1, y1 = x0 + ox, y0 + oy, x1 + ox, y1 + oy

        kimlik_no = str(ocr_sonuc.get("tc_no", "Bulunamadi")).strip() or "Bulunamadi"
        ad = ocr_sonuc.get("ad", "Bulunamadi")
        soyad = ocr_sonuc.get("soyad", "Bulunamadi")
        tc_supheli = kimlik_no != "Bulunamadi" and not ocr_sonuc.get("tc_dogrulandi", True)

        eksikler = [
            alan for alan, deger in
            (("Kimlik No", kimlik_no), ("Ad", ad), ("Soyad", soyad))
            if deger == "Bulunamadi"
        ]
        if eksikler:
            durum = "Seçilen alandan okundu — eksik: " + ", ".join(eksikler)
        elif tc_supheli:
            durum = "Seçilen alandan okundu — Kimlik No doğrulanamadı"
        else:
            durum = "Seçilen alandan okundu"

        belge_yazi = {
            "tc": "T.C. Kimlik", "eski_tc": "Eski T.C. Kimlik", "gocmen": "Göçmen / Yabancı",
        }.get(sonuc.get("belge_tipi", "tc"), "T.C. Kimlik")

        yeni = {
            "Dosya": kaynak.get("Dosya", ""),
            "Sayfa": kaynak.get("Sayfa", "-"),
            "Kart": "seçim",
            "Belge Türü": belge_yazi,
            "Kimlik No": kimlik_no,
            "Ad": ad,
            "Soyad": soyad,
            "Bitiş Tarihi": ocr_sonuc.get("bitis_tarihi", "") or "-",
            "Geçerlilik": "-",
            "Durum": durum,
            "Süre": round(float(ocr_sonuc.get("ocr_suresi", 0.0) or 0.0), 2),
            "_belge_tipi": sonuc.get("belge_tipi", "tc"),
            "_belge_gecerli": ocr_sonuc.get("belge_gecerli"),
            "_kart_bulundu": True,
            "_manuel_eklendi": True,
            "_tc_supheli": tc_supheli,
            "_kaynak_yol": kaynak.get("_kaynak_yol"),
            "_koseler": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
            "_preview": sonuc.get("kart"),
            "_debug_resmi": ocr_sonuc.get("debug_resmi"),
            "_kart_sonuc": {
                "belge_tipi": sonuc.get("belge_tipi"),
                "arama_modu": "elle seçim",
                "mesaj": ("Seçilen alanda kart hizalandı."
                          if sonuc.get("hizalandi") else "Seçilen alan doğrudan okundu."),
                "referans_hatalari": {},
            },
            "_ocr_sonuc": {
                "guven": ocr_sonuc.get("guven"),
                "tc_dogrulandi": ocr_sonuc.get("tc_dogrulandi"),
                "tc_kaynak": ocr_sonuc.get("tc_kaynak"),
                "kimlik_no_conf": ocr_sonuc.get("kimlik_no_conf", 0.0),
                "ad_conf": ocr_sonuc.get("ad_conf", 0.0),
                "soyad_conf": ocr_sonuc.get("soyad_conf", 0.0),
                "ocr_suresi": ocr_sonuc.get("ocr_suresi", 0.0),
                "tum_ocr": ocr_sonuc.get("tum_ocr", []),
            },
        }

        # Aynı sayfanın son satırından sonraya ekle; sıralama ve numaralandırma
        # zaten konuma göre yeniden yapılacak.
        anahtar = (yeni["Dosya"], yeni["Sayfa"])
        hedef = len(self.sonuclar)
        for i in range(len(self.sonuclar) - 1, -1, -1):
            if (self.sonuclar[i].get("Dosya"), self.sonuclar[i].get("Sayfa")) == anahtar:
                hedef = i + 1
                break

        self.sonuclar.insert(hedef, yeni)
        self.sayfalari_duzenle()
        self.tabloyu_yenile(secilecek=yeni)
        self.ozet_guncelle(self.sonuclar)
        self.gecmisi_guncellemeyi_planla()

    def zoom_degistir(self, carpan):
        if self.zoom is None:
            # Sığdırma modundayken mevcut görünen boyuttan devam et.
            pixmap = self.preview.pixmap()
            img = self.aktif_onizleme_resmi()
            if pixmap is None or img is None:
                return
            self.zoom = pixmap.width() / float(img.shape[1])
        self.zoom = max(0.1, min(8.0, self.zoom * carpan))
        self.onizleme_goster()

    def zoom_sigdir(self):
        self.zoom = None
        self.onizleme_goster()

    def aktif_onizleme_resmi(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.sonuclar):
            return None
        return self.onizleme_gorselleri(self.sonuclar[r])[0]

    def eventFilter(self, nesne, olay):
        """Ctrl + fare tekerleği ile yakınlaştırma."""
        if (nesne is self.preview_scroll.viewport()
                and olay.type() == QEvent.Wheel
                and olay.modifiers() & Qt.ControlModifier):
            self.zoom_degistir(1.15 if olay.angleDelta().y() > 0 else 1 / 1.15)
            return True
        return super().eventFilter(nesne, olay)

    def elle_satir_ekle(self):
        """Tespitin kaçırdığı bir kimliği elle girmek için boş satır açar."""
        if not self.sonuclar:
            return

        r = self.table.currentRow()
        if r < 0:
            r = len(self.sonuclar) - 1
        kaynak = self.sonuclar[r]

        yeni = {
            "Dosya": kaynak.get("Dosya", ""),
            "Sayfa": kaynak.get("Sayfa", "-"),
            "Kart": "elle",
            "Belge Türü": kaynak.get("Belge Türü", "T.C. Kimlik"),
            "Kimlik No": "",
            "Ad": "",
            "Soyad": "",
            "Bitiş Tarihi": "-",
            "Geçerlilik": "-",
            "Durum": "Elle eklendi — alanları doldurun",
            "Süre": 0.0,
            "_belge_tipi": kaynak.get("_belge_tipi", "tc"),
            "_belge_gecerli": None,
            "_kart_bulundu": True,
            "_manuel_eklendi": True,
            "_kaynak_yol": kaynak.get("_kaynak_yol"),
            "_koseler": None,
            # Konumu yok; sıralamada seçili satırın hemen ardında dursun.
            "_ardil_oldugu": id(kaynak),
            "_preview": None,
            "_kart_sonuc": {},
            "_ocr_sonuc": {},
        }

        self.sonuclar.insert(r + 1, yeni)
        self.sayfalari_duzenle()

        # Elle satır girilecek: düzenleme modunu kendiliğinden aç.
        if not self.edit_mode:
            self.edit_btn.setChecked(True)

        self.tabloyu_yenile(secilecek=yeni)
        self.table.editItem(
            self.table.item(self.table.currentRow(), self.headers.index("Kimlik No"))
        )
        self.ozet_guncelle(self.sonuclar)
        self.gecmisi_guncellemeyi_planla()

    def satiri_sil(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.sonuclar):
            return

        satir = self.sonuclar[r]
        cevap = QMessageBox.question(
            self,
            "Satırı sil",
            f"{satir.get('Dosya')} / sayfa {satir.get('Sayfa')} satırı silinsin mi?\n"
            f"Kimlik No: {satir.get('Kimlik No')}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if cevap != QMessageBox.Yes:
            return

        self.sonuclar.pop(r)
        # Silinen kimlikten sonrakilerin numarası kaysın.
        self.sayfalari_duzenle()
        self.tabloyu_yenile()

        self.ozet_guncelle(self.sonuclar)
        self.gecmisi_guncellemeyi_planla()
        if self.table.rowCount():
            self.table.selectRow(min(r, self.table.rowCount() - 1))
        else:
            self.preview.clear()
            self._sayfa_bilgi_metni = ""
            self._onizleme_ipucu_metni = ""
            self.onizleme_yazilarini_tazele()

    def sayfa_satirlarini_sirala(self, satirlar):
        """Bir sayfanın satırlarını kimliğin sayfadaki YERİNE göre dizer:
        yukarıdan aşağıya, aynı hizadakiler soldan sağa. Konumu olmayan
        (elle açılmış boş) satırlar sona alınır."""
        konumlu = [r for r in satirlar if r.get("_koseler")]
        konumsuz = [r for r in satirlar if not r.get("_koseler")]

        if len(konumlu) > 1:
            # Tespit tarafındaki sıralama ile aynı kural kullanılsın diye
            # satırlar geçici olarak o fonksiyonun beklediği şekle sarılıyor.
            sarmal = [{"koseler": np.asarray(r["_koseler"], dtype=np.float32), "_satir": r}
                      for r in konumlu]
            konumlu = [k["_satir"] for k in sayfa_sirasina_diz(sarmal)]

        if not konumsuz:
            return konumlu

        # Konumu olmayan (elle açılmış boş) satırlar, açıldıkları satırın hemen
        # ardına yerleşir; böylece numarası da oradan devam eder. Dayandığı
        # satır kalmadıysa sayfanın sonuna eklenir.
        sonuc = list(konumlu)
        artakalan = []
        for satir in konumsuz:
            ardil = satir.get("_ardil_oldugu")
            yerlesti = False
            if ardil is not None:
                for i, mevcut in enumerate(sonuc):
                    if id(mevcut) == ardil:
                        sonuc.insert(i + 1, satir)
                        yerlesti = True
                        break
            if not yerlesti:
                artakalan.append(satir)

        return sonuc + artakalan

    def sayfalari_duzenle(self):
        """Tüm satırları dosya+sayfa gruplarında konuma göre sıralar ve Kart
        numaralarını (1/4, 2/4 …) baştan verir.

        Araya elle bir kimlik eklendiğinde numarasını konumundan alır,
        kendisinden sonraki kimliklerin numarası da bir artar."""
        gruplar = {}
        sira = []
        for satir in self.sonuclar:
            anahtar = (satir.get("Dosya"), satir.get("Sayfa"))
            if anahtar not in gruplar:
                gruplar[anahtar] = []
                sira.append(anahtar)
            gruplar[anahtar].append(satir)

        yeni_liste = []
        for anahtar in sira:
            sirali = self.sayfa_satirlarini_sirala(gruplar[anahtar])
            toplam = sum(1 for r in sirali if r.get("_kart_bulundu"))
            sayac = 0
            for satir in sirali:
                if satir.get("_kart_bulundu"):
                    sayac += 1
                    satir["Kart"] = f"{sayac}/{toplam}"
                else:
                    satir["Kart"] = "-"
            yeni_liste.extend(sirali)

        self.sonuclar = yeni_liste
        self.canli_sonuclar = list(self.sonuclar)
        self._isaretli_onbellek = (None, None)

    def tabloyu_yenile(self, secilecek=None):
        """Tabloyu self.sonuclar'dan baştan kurar (sıra değiştiğinde).
        `secilecek` verilen satır sözlüğü tekrar seçili hale gelir."""
        self._table_updating = True
        try:
            self.table.setRowCount(0)
            for satir in self.sonuclar:
                self.tabloya_satir_yaz(satir)
        finally:
            self._table_updating = False

        if secilecek is not None:
            for i, satir in enumerate(self.sonuclar):
                if satir is secilecek:
                    self.table.selectRow(i)
                    break

    @staticmethod
    def duzenlenmis_hucreyi_boya(item):
        """Elle değiştirilen hücrenin vurgusu.

        Eskiden açık mavi bir ZEMİN veriliyordu; koyu temada yazı da açık
        olduğu için hücre okunmaz hale geliyordu. Artık koyu lacivert zemin +
        açık mavi yazı kullanılıyor."""
        item.setBackground(QColor("#1f3350"))
        item.setForeground(QColor("#93c5fd"))

    def tabloya_satir_yaz(self, row, hedef=None):
        """Tek bir satırı tabloya ekler (renklendirme kurallarıyla birlikte)."""
        r = self.table.rowCount() if hedef is None else hedef
        self.table.insertRow(r)

        editable_headers = {"Kimlik No", "Ad", "Soyad"}
        for c, h in enumerate(self.headers):
            item = QTableWidgetItem(str(row.get(h, "")))

            if self.edit_mode and h in editable_headers:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

            if row.get("_belge_gecerli") is False and h == "Geçerlilik":
                item.setForeground(QColor("#ff6b6b"))
            if row.get("_kurtarildi") and h in ("Kart", "Durum"):
                item.setForeground(QColor("#f0b429"))
            if row.get("_tc_supheli") and h in ("Kimlik No", "Durum"):
                item.setForeground(QColor("#f0b429"))
            if row.get("_manuel_eklendi"):
                item.setForeground(QColor("#ffb454"))
            # Excel'den tamamlanan ad/soyad ve ad uyuşmazlığı
            if h in (row.get("_excelden_alanlar") or ()):
                item.setForeground(QColor("#a5b4fc"))
            if row.get("_ad_uyusmazligi") and h in ("Ad", "Soyad", "Durum"):
                item.setForeground(QColor("#f0b429"))
            if h in (row.get("_duzenlenen_alanlar") or ()):
                self.duzenlenmis_hucreyi_boya(item)

            self.table.setItem(r, c, item)
        return r

    def hucreye_tiklandi(self, satir, sutun):
        """Sayfa sütunu tüm sayfayı, diğer sütunlar o satırın kimliğini gösterir."""
        yeni_kip = "sayfa" if self.headers[sutun] == "Sayfa" else "kart"
        if yeni_kip != self._onizleme_kipi:
            self._onizleme_kipi = yeni_kip
            self.zoom = None          # kip değişince sığdırmaya dön
        self.onizleme_goster()

    def karta_yakinlas(self, resim, koseler, pay_orani=0.18):
        """Sayfa görüntüsünü, ilgili kimliğin çevresinden kırpar.
        Dönüş: (kırpılmış görüntü, (ofset_x, ofset_y))"""
        h, w = resim.shape[:2]
        nokta = np.asarray(koseler, dtype=np.float32).reshape(-1, 2)
        x0, y0 = nokta[:, 0].min(), nokta[:, 1].min()
        x1, y1 = nokta[:, 0].max(), nokta[:, 1].max()

        pay = pay_orani * max(x1 - x0, y1 - y0)
        x0 = int(max(0, x0 - pay))
        y0 = int(max(0, y0 - pay))
        x1 = int(min(w, x1 + pay))
        y1 = int(min(h, y1 + pay))

        if x1 - x0 < 20 or y1 - y0 < 20:
            return resim, (0, 0)
        return np.ascontiguousarray(resim[y0:y1, x0:x1]), (x0, y0)

    def onizleme_goster(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.sonuclar):
            return

        sonuc = self.sonuclar[r]

        gosterilecek, ham, ofset = self.onizleme_gorselleri(sonuc)
        self.pixmapi_yerlestir(gosterilecek, ham=ham, ofset=ofset)
        self.sayfa_bilgisini_guncelle(sonuc)

        if not self.debug_cb.isChecked():
            self.debug_text.clear()
            return

        kart = sonuc.get("_kart_sonuc", {}) or {}
        ocr = sonuc.get("_ocr_sonuc", {}) or {}

        satirlar = [
            f"Dosya: {sonuc.get('Dosya')}",
            f"Sayfa: {sonuc.get('Sayfa')}",
            f"Kart: {sonuc.get('Kart')}",
            f"Belge Türü: {sonuc.get('Belge Türü')}",
            "",
            "=== SONUÇ ===",
            f"Kimlik No: {sonuc.get('Kimlik No')}",
            f"Ad: {sonuc.get('Ad')}",
            f"Soyad: {sonuc.get('Soyad')}",
            f"Bitiş Tarihi: {sonuc.get('Bitiş Tarihi')}",
            f"Geçerlilik: {sonuc.get('Geçerlilik')}",
            f"Durum: {sonuc.get('Durum')}",
            "",
            "=== BELGE TESPİT ===",
            f"Seçilen tip: {kart.get('belge_tipi')}",
            f"Düzeltildi: {kart.get('duzeltildi')}",
            f"Düzeltme yöntemi: {kart.get('duzeltme_yontemi')}",
            f"Fallback: {kart.get('fallback')}",
            f"İyi eşleşme: {kart.get('iyi_eslesme')}",
            f"Inlier: {kart.get('inlier')}",
            f"Inlier oranı: {kart.get('inlier_orani')}",
            f"Skor: {kart.get('skor')}",
            f"Sayfadaki kart: {kart.get('kart_sirasi')}/{kart.get('kart_sayisi')}",
            f"Arama modu: {kart.get('arama_modu')}",
            f"Mesaj: {kart.get('mesaj')}",
            f"Aday sırası: {kart.get('aday_sirasi')}",
            "",
            "=== OCR ===",
            f"Güven: {ocr.get('guven')}",
            f"Kimlik No doğrulandı: {ocr.get('tc_dogrulandi')} (kaynak: {ocr.get('tc_kaynak')})",
            f"Kimlik no confidence: {float(ocr.get('kimlik_no_conf', 0.0) or 0.0):.3f}",
            f"Ad confidence: {float(ocr.get('ad_conf', 0.0) or 0.0):.3f}",
            f"Soyad confidence: {float(ocr.get('soyad_conf', 0.0) or 0.0):.3f}",
            f"OCR süresi: {float(ocr.get('ocr_suresi', 0.0) or 0.0):.3f} sn",
        ]

        if sonuc.get("_belge_tipi") == "gocmen":
            satirlar.extend([
                "",
                "=== GÖÇMEN OCR ===",
                f"Başlangıç: {ocr.get('baslangic_tarihi')}",
                f"Bitiş: {ocr.get('bitis_tarihi')}",
                f"Geçerlilik durumu: {ocr.get('gecerlilik_durumu')}",
                f"Hızlı yol: {ocr.get('hizli_yol_kullanildi')}",
                f"Hızlı hücre OCR: {float(ocr.get('hizli_hucre_ocr_suresi', 0.0) or 0.0):.3f} sn",
                f"Detector fallback: {ocr.get('detector_fallback_kullanildi')}",
                f"Detector OCR: {float(ocr.get('detector_ocr_suresi', 0.0) or 0.0):.3f} sn",
                f"Fallback denenen: {ocr.get('fallback_denenen_alanlar')}",
                f"Fallback kullanılan: {ocr.get('fallback_kullanilan_alanlar')}",
                f"Fallback OCR süresi: {float(ocr.get('fallback_ocr_suresi', 0.0) or 0.0):.3f} sn",
            ])

        ham = ocr.get("tum_ocr", []) or []
        if ham:
            satirlar.append("")
            satirlar.append("=== HAM OCR ===")
            for item in ham[:80]:
                try:
                    satirlar.append(
                        f"{item.get('text', '')} — {float(item.get('conf', 0.0) or 0.0):.3f}"
                    )
                except Exception:
                    satirlar.append(str(item))

        self.debug_text.setPlainText(
            "\n".join(satirlar)
        )


    def excel_kaydet(self):
        """
        Sonuçları geçici bir Excel dosyasına çevirir ve
        varsayılan Excel uygulamasıyla doğrudan açar.

        Kullanıcı isterse Excel içinden normal "Farklı Kaydet"
        ile kalıcı konum seçebilir.
        """
        if not self.sonuclar:
            return

        try:
            fd, yol = tempfile.mkstemp(
                prefix="kimlik_sonuclari_",
                suffix=".xlsx",
            )
            os.close(fd)

            # Nihai Excel sadece:
            # Kimlik No | Ad | Soyad
            df = pd.DataFrame([
                {
                    "Kimlik No": r["Kimlik No"],
                    "Ad": r["Ad"],
                    "Soyad": r["Soyad"],
                }
                for r in self.sonuclar
            ])

            df.to_excel(
                yol,
                index=False,
                engine="openpyxl",
            )

            wb = load_workbook(
                yol
            )

            ws = wb.active
            ws.title = "Kimlik Sonuçları"

            # Excel'de GEÇERSİZ kimlikler ve checksum'ı tutmayan
            # (doğrulanamamış) kimlik numaraları renklendirilir; ikincisi
            # kullanıcının gözden geçirmesi gereken tek şey.
            gecersiz_fill = PatternFill(
                "solid",
                fgColor="FFF2CC",
            )

            supheli_fill = PatternFill(
                "solid",
                fgColor="FFD9A0",
            )

            for cell in ws[1]:
                cell.font = Font(
                    bold=True
                )

            for i, r in enumerate(
                self.sonuclar,
                start=2,
            ):
                if r.get(
                    "_belge_gecerli"
                ) is False:
                    # Excel yalnız 3 kolon: Kimlik No | Ad | Soyad
                    # Geçersiz belgenin tamamı tek renkle görünür.
                    for col in range(
                        1,
                        4,
                    ):
                        ws.cell(
                            i,
                            col,
                        ).fill = gecersiz_fill

                if r.get("_tc_supheli"):
                    hucre = ws.cell(i, 1)
                    hucre.fill = supheli_fill
                    hucre.comment = Comment(
                        "Bu numara okundu ama doğrulama hanesi tutmuyor; "
                        "kimlikle karşılaştırıp kontrol edin.",
                        "Kimlik Okuyucu",
                    )

            ws.column_dimensions["A"].width = 18
            ws.column_dimensions["B"].width = 24
            ws.column_dimensions["C"].width = 24

            wb.save(
                yol
            )

            dosyayi_sistemde_ac(
                yol
            )

            self.bilgi.setText(
                "Excel oluşturuldu ve açıldı. "
                "Kalıcı olarak saklamak istersen Excel içinden Farklı Kaydet kullanabilirsin."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Excel oluşturma hatası",
                str(e),
            )


    def pdf_olustur(self):
        """Tablodaki GÜNCEL sıraya göre kimlik PDF'i üretir.

        Eskiden PDF işlem sırasında hazırlanıyordu; elle kesilen/eklenen
        kimlikler ve sıralama değişiklikleri içine girmiyordu. Artık kaydetme
        anında tablodan üretiliyor: her satırın kendi görüntüsü, tabloda
        göründüğü sırayla (sayfa sayfa, her sayfada yukarıdan aşağıya) eklenir.
        """
        doc = pymupdf.open()
        try:
            for satir in self.sonuclar:
                resim = satir.get("_preview")
                if resim is not None:
                    resmi_pdfe_ekle(doc, resim)
            if not doc.page_count:
                return None
            return doc.tobytes(garbage=3, deflate=True)
        finally:
            doc.close()

    def pdf_kaydet(self):
        """
        Kimlik PDF'ini geçici dosyaya yazar ve varsayılan PDF
        görüntüleyiciyle doğrudan açar.
        """
        if not self.sonuclar:
            return

        self.pdf_bytes = self.pdf_olustur()
        if not self.pdf_bytes:
            QMessageBox.information(
                self,
                "PDF oluşturulamadı",
                "Tabloda PDF'e eklenecek görüntü bulunamadı.",
            )
            return

        try:
            fd, yol = tempfile.mkstemp(
                prefix="kimlikler_",
                suffix=".pdf",
            )
            os.close(fd)

            with open(
                yol,
                "wb",
            ) as f:
                f.write(
                    self.pdf_bytes
                )

            dosyayi_sistemde_ac(
                yol
            )

            self.bilgi.setText(
                "PDF oluşturuldu ve açıldı. "
                "Kalıcı olarak saklamak istersen PDF görüntüleyicinden kaydedebilirsin."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "PDF oluşturma hatası",
                str(e),
            )


    def closeEvent(self, event):
        """Pencere kapanırken süren tarama arka planda devam etmesin.
        Worker o an işlediği sayfayı bitirip döngüden çıkar."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.durdur()
            self.worker.wait(5000)

        # Zamanlayıcıda bekleyen düzenleme varsa diske yaz.
        if self.gecmis_zamanlayici.isActive():
            self.gecmisi_guncelle()

        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if hasattr(self, "sayfa_bilgi"):
            self.onizleme_yazilarini_tazele()

        # Pencere boyutu değişince seçili kimliği yeni alana tekrar sığdır.
        if hasattr(self, "table") and self.table.currentRow() >= 0:
            self.onizleme_goster()



if __name__ == "__main__":
    print("[CHECK] desktop.py içinde cv2.imdecode kullanılmıyor.")
    app = QApplication(sys.argv)
    pencere = MainWindow()
    pencere.showMaximized()
    sys.exit(app.exec())