import sys
import os
import time
import tempfile
import subprocess
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils import get_column_letter

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QCheckBox, QSplitter, QHeaderView,
    QAbstractItemView, QTextEdit, QScrollArea, QSizePolicy, QFrame
)

from goruntu_isleme import kart_tespit_et_ve_duzelt
from metin_ayiklama import bilgileri_cimbizla

BUILD_ID = "DESKTOP-IMAGE-FIX-V10"
print(f"[DESKTOP BUILD] {BUILD_ID}")
print(f"[DESKTOP FILE] {__file__}")
print(f"[CV2 FILE] {getattr(cv2, '__file__', '?')}")
print(f"[CV2 VERSION] {getattr(cv2, '__version__', '?')}")



PDF_RENDER_DPI = 170
MAX_GORUNTU_PIKSEL = 40_000_000
DESTEKLENEN = {".jpg", ".jpeg", ".png", ".pdf"}


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
    ext = os.path.splitext(yol)[1].lower()
    if ext == ".pdf":
        doc = pymupdf.open(yol)
        try:
            for sayfa_no, sayfa in enumerate(doc, start=1):
                tahmini_w = max(1, int(sayfa.rect.width * PDF_RENDER_DPI / 72))
                tahmini_h = max(1, int(sayfa.rect.height * PDF_RENDER_DPI / 72))
                if tahmini_w * tahmini_h > MAX_GORUNTU_PIKSEL:
                    yield sayfa_no, None, "Sayfa çözünürlüğü güvenli sınırı aşıyor."
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
                    yield sayfa_no, resim, None

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

                    yield sayfa_no, None, hata
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
                    )
                    return

                im = im.convert("RGB")
                rgb = np.array(im)

            if rgb is None or rgb.size == 0:
                yield None, None, "Görüntü boş veya okunamadı."
                return

            resim = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            yield None, resim, None

        except Exception as e:
            hata = (
                "Görüntü okunamadı: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[GÖRÜNTÜ HATA] {os.path.basename(yol)}: {hata}",
                flush=True,
            )

            yield None, None, hata


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




class Worker(QThread):
    progress = Signal(int, int, str)
    row_ready = Signal(dict)
    finished_ok = Signal(list, object)
    failed = Signal(str)

    def __init__(self, dosyalar, pdf_olustur=False, debug=False):
        super().__init__()
        self.dosyalar = dosyalar
        self.pdf_olustur = pdf_olustur
        self.debug = debug

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
                else:
                    isler.append((yol, None))

            toplam = max(1, len(isler))
            sonuclar = []
            pdf_doc = pymupdf.open() if self.pdf_olustur else None
            sayac = 0

            for yol in self.dosyalar:
                for sayfa_no, resim, girdi_hatasi in dosyayi_goruntulere_ayir(yol):
                    sayac += 1
                    self.progress.emit(sayac, toplam, f"{os.path.basename(yol)} işleniyor")

                    baslangic = time.perf_counter()
                    belge_tipi = "bilinmiyor"
                    kart = None
                    kart_sonuc = {}
                    ocr_sonuc = {}
                    hata = girdi_hatasi

                    if resim is not None:
                        try:
                            kart_sonuc = kart_tespit_et_ve_duzelt(resim, debug_kart=self.debug)
                            if kart_sonuc.get("basarili", False):
                                kart = kart_sonuc.get("kart")
                                belge_tipi = kart_sonuc.get("belge_tipi", "tc")
                            else:
                                eslesen = kart_sonuc.get("belge_tipi")
                                if eslesen in {"tc", "eski_tc", "gocmen"}:
                                    belge_tipi = eslesen
                                hata = kart_sonuc.get("mesaj") or "Kart tespit edilemedi."
                        except Exception as e:
                            hata = f"Kart tespit hatası: {e}"

                    # Streamlit backend'indeki kritik bağlantı:
                    # OCR yalnızca tespit edilen/düzeltilen kart üzerinde ve belge_tipi ile çalışır.
                    if kart is not None:
                        try:
                            ocr_sonuc = bilgileri_cimbizla(
                                kart,
                                sayfa_no=sayfa_no,
                                debug=self.debug,
                                belge_tipi=belge_tipi
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

                    eksikler = []
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
                    else:
                        durum = "Başarılı"

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

                    # PDF: kart bulunduysa düzeltilmiş kart, bulunamadıysa orijinal.
                    if pdf_doc is not None:
                        pdf_resmi = kart if kart is not None else resim
                        if pdf_resmi is not None:
                            resmi_pdfe_ekle(pdf_doc, pdf_resmi)

                    row = {
                        "Dosya": os.path.basename(yol),
                        "Sayfa": sayfa_no if sayfa_no is not None else "-",
                        "Belge Türü": belge_yazi,
                        "Kimlik No": kimlik_no,
                        "Ad": ad,
                        "Soyad": soyad,
                        "Bitiş Tarihi": bitis if belge_tipi == "gocmen" else "-",
                        "Geçerlilik": gecerlilik,
                        "Durum": durum,
                        "Süre": round(time.perf_counter() - baslangic, 2),
                        "_belge_tipi": belge_tipi,
                        "_belge_gecerli": belge_gecerli,
                        "_ad_conf": ad_conf,
                        "_soyad_conf": soyad_conf,
                        "_preview": kart if kart is not None else resim,
                        "_debug_resmi": ocr_sonuc.get("debug_resmi"),
                        "_kart_debug": kart_sonuc.get("debug_resmi"),
                        "_kart_sonuc": {
                            "belge_tipi": kart_sonuc.get("belge_tipi"),
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
                    sonuclar.append(row)
                    self.row_ready.emit(row)

            pdf_bytes = None
            if pdf_doc is not None:
                if pdf_doc.page_count:
                    pdf_bytes = pdf_doc.tobytes(garbage=3, deflate=True)
                pdf_doc.close()

            self.finished_ok.emit(sonuclar, pdf_bytes)
        except Exception as e:
            self.failed.emit(repr(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kimlik Okuyucu — Masaüstü")
        self.resize(1280, 760)
        self.setMinimumSize(980, 620)

        self.dosyalar = []
        self.sonuclar = []
        self.pdf_bytes = None
        self.worker = None

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

        self.edit_btn = QPushButton("Düzenle")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setEnabled(False)

        self.pdf_cb = QCheckBox("Kimlikleri PDF olarak birleştir")
        self.debug_cb = QCheckBox("Debug göster")

        for btn in (
            self.sec_btn,
            self.klasor_btn,
            self.baslat_btn,
            self.edit_btn,
        ):
            btn.setMinimumHeight(42)

        ust.addWidget(self.sec_btn)
        ust.addWidget(self.klasor_btn)
        ust.addWidget(self.baslat_btn)
        ust.addWidget(self.edit_btn)
        ust.addStretch(1)
        ust.addWidget(self.pdf_cb)
        ust.addWidget(self.debug_cb)
        ana.addLayout(ust)

        self.bilgi = QLabel("Dosya seçilmedi.")
        self.bilgi.setWordWrap(True)
        ana.addWidget(self.bilgi)

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
            "Dosya", "Sayfa", "Belge Türü", "Kimlik No", "Ad", "Soyad",
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

        genislikler = [190, 65, 145, 145, 175, 175, 120, 115, 230, 70]
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

        self.preview = QLabel("Bir satır seçince kimlik burada gösterilecek.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(320, 220)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview.setStyleSheet(
            "border: 1px solid #454b53; background: #111316; color: #dce1e7; border-radius: 7px;"
        )

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setFrameShape(QFrame.NoFrame)
        preview_scroll.setWidget(self.preview)
        preview_layout.addWidget(preview_scroll)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setPlaceholderText(
            "Debug açıksa seçilen satırın belge tespit / OCR ayrıntıları burada görünür."
        )
        self.debug_text.setMinimumHeight(180)

        self.sag_splitter.addWidget(preview_frame)
        self.sag_splitter.addWidget(self.debug_text)

        # Debug kapalı başlar; böylece kimlik önizlemesi bütün sağ paneli kullanır.
        self.debug_text.hide()
        self.sag_splitter.setSizes([1, 0])

        saglay.addWidget(self.sag_splitter, 1)
        splitter.addWidget(sag)
        splitter.setSizes([850, 430])

        self.sec_btn.clicked.connect(self.dosya_sec)
        self.klasor_btn.clicked.connect(self.klasor_sec)
        self.baslat_btn.clicked.connect(self.baslat)
        self.edit_btn.toggled.connect(self.duzenleme_modu_degisti)
        self.debug_cb.toggled.connect(self.debug_gorunumu_degisti)
        self.table.itemSelectionChanged.connect(self.onizleme_goster)
        self.table.cellChanged.connect(self.hucre_degisti)
        self.excel_btn.clicked.connect(self.excel_kaydet)
        self.pdf_btn.clicked.connect(self.pdf_kaydet)

    def dosya_sec(self):
        yollar, _ = QFileDialog.getOpenFileNames(
            self, "Kimlik dosyalarını seç", "",
            "Kimlik/PDF (*.jpg *.jpeg *.png *.pdf)"
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
        self.table.setRowCount(0)
        self.sonuclar = []
        self.pdf_bytes = None
        self.excel_btn.setEnabled(False)
        self.pdf_btn.setEnabled(False)
        self.edit_btn.setChecked(False)
        self.edit_btn.setEnabled(False)
        self.baslat_btn.setEnabled(False)
        self.progress.setValue(0)
        self.debug_text.clear()

        self.worker = Worker(self.dosyalar, self.pdf_cb.isChecked(), self.debug_cb.isChecked())
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
        self._table_updating = True
        try:
            r = self.table.rowCount()
            self.table.insertRow(r)

            editable_headers = {"Kimlik No", "Ad", "Soyad"}

            for c, h in enumerate(self.headers):
                item = QTableWidgetItem(str(row.get(h, "")))

                if self.edit_mode and h in editable_headers:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Geçersiz belgelerde satırı sarıya boyamıyoruz:
                # koyu temada okunabilirliği bozuyordu.
                # Bunun yerine yalnız "Geçerlilik" hücresini kırmızı yazıyla vurgula.
                if (
                    row.get("_belge_gecerli") is False
                    and h == "Geçerlilik"
                ):
                    item.setForeground(QColor("#ff6b6b"))

                self.table.setItem(r, c, item)
        finally:
            self._table_updating = False


    def debug_gorunumu_degisti(self, aktif):
        """
        Debug alanını yalnız checkbox açıkken göster.
        Kapalıyken sağ panelin tamamını kimlik önizlemesine bırak.
        """
        if aktif:
            self.debug_text.show()
            self.sag_splitter.setSizes([430, 260])
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

        item.setBackground(QColor("#dbeafe"))

        if self.table.currentRow() == row:
            self.onizleme_goster()

    def bitti(self, sonuclar, pdf_bytes):
        self.sonuclar = sonuclar
        self.pdf_bytes = pdf_bytes
        self.baslat_btn.setEnabled(True)
        self.excel_btn.setEnabled(bool(sonuclar))
        self.pdf_btn.setEnabled(bool(pdf_bytes))
        self.edit_btn.setEnabled(bool(sonuclar))
        self.bilgi.setText(f"Tamamlandı — {len(sonuclar)} kimlik/sayfa işlendi.")
        if self.table.rowCount():
            self.table.selectRow(0)

    def hata(self, mesaj):
        self.baslat_btn.setEnabled(True)
        QMessageBox.critical(self, "Hata", mesaj)

    def onizleme_goster(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.sonuclar):
            return

        sonuc = self.sonuclar[r]

        # Debug açıksa OCR kutulu görsel öncelikli.
        if self.debug_cb.isChecked():
            img = sonuc.get("_debug_resmi")
            if img is None:
                img = sonuc.get("_kart_debug")
            if img is None:
                img = sonuc.get("_preview")
        else:
            img = sonuc.get("_preview")

        if img is None:
            self.preview.clear()
            self.preview.setText("Görüntü bulunamadı.")
        else:
            # Kimliği sağdaki mevcut alana TAM SIĞDIR.
            # Büyük kimlikler küçültülür, oran korunur.
            alan_w = max(260, self.preview.width() - 24)
            alan_h = max(180, self.preview.height() - 24)

            pixmap = bgr_to_pixmap(
                img,
                max_w=alan_w,
                max_h=alan_h,
            )

            self.preview.setPixmap(pixmap)
            self.preview.setAlignment(Qt.AlignCenter)

        if not self.debug_cb.isChecked():
            self.debug_text.clear()
            return

        kart = sonuc.get("_kart_sonuc", {}) or {}
        ocr = sonuc.get("_ocr_sonuc", {}) or {}

        satirlar = [
            f"Dosya: {sonuc.get('Dosya')}",
            f"Sayfa: {sonuc.get('Sayfa')}",
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
            f"Mesaj: {kart.get('mesaj')}",
            f"Aday sırası: {kart.get('aday_sirasi')}",
            "",
            "=== OCR ===",
            f"Güven: {ocr.get('guven')}",
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

            # Excel'de yalnızca GEÇERSİZ kimlikler renklendirilir.
            # Eksik alan, düşük confidence veya "kesin değil" durumları
            # artık herhangi bir renkle işaretlenmez.
            gecersiz_fill = PatternFill(
                "solid",
                fgColor="FFF2CC",
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


    def pdf_kaydet(self):
        """
        İşlem sırasında hazırlanmış kimlik PDF'ini geçici dosyaya
        yazar ve varsayılan PDF görüntüleyiciyle doğrudan açar.
        """
        if not self.pdf_bytes:
            QMessageBox.information(
                self,
                "PDF hazır değil",
                (
                    "Önce 'Kimlikleri PDF olarak birleştir' seçeneğini "
                    "işaretleyip işlemi çalıştırmalısın."
                ),
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


    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Pencere boyutu değişince seçili kimliği yeni alana tekrar sığdır.
        if hasattr(self, "table") and self.table.currentRow() >= 0:
            self.onizleme_goster()



if __name__ == "__main__":
    print("[CHECK] desktop.py içinde cv2.imdecode kullanılmıyor.")
    app = QApplication(sys.argv)
    pencere = MainWindow()
    pencere.showMaximized()
    sys.exit(app.exec())
