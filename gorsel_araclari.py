# -*- coding: utf-8 -*-
"""Goruntu/dosya donusum yardimcilari: PDF/Excel/JPG'yi OpenCV goruntusune
cevirme, OpenCV goruntusunu Qt pixmap'ine cevirme, uretilen Excel/PDF'i
isletim sisteminin varsayilan uygulamasiyla acma.

desktop.py'den ayrildi (tek dosya cok buyumustu) -- bu modul saf yardimci
fonksiyonlar icerir, herhangi bir Qt penceresine bagimli degildir (bgr_to_pixmap
haric: o da yalnizca QImage/QPixmap uretir, bir widget'a dokunmaz)."""

import os
import sys
import subprocess

import cv2
import numpy as np
import pymupdf
from PIL import Image

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

import excel_kaynak


PDF_RENDER_DPI = 170
MAX_GORUNTU_PIKSEL = 40_000_000
DESTEKLENEN = {".jpg", ".jpeg", ".png", ".pdf", ".xlsx", ".xlsm"}


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

