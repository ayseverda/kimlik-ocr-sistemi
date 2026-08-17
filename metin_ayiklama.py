import cv2
import re
import logging
import numpy as np
from paddleocr import PaddleOCR


logging.disable(logging.WARNING)


# =========================================================
# PADDLE OCR
# =========================================================

ocr = PaddleOCR(
    lang="tr",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


# =========================================================
# METİN NORMALİZASYONU
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()

    tablo = str.maketrans({
        "İ": "I",
        "Ş": "S",
        "Ğ": "G",
        "Ü": "U",
        "Ö": "O",
        "Ç": "C"
    })

    text = text.translate(tablo)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# TC DOĞRULAMA
# =========================================================

def tc_kimlik_gecerli_mi(no):
    if not no:
        return False

    if len(no) != 11:
        return False

    if not no.isdigit():
        return False

    if no[0] == "0":
        return False

    d = [int(x) for x in no]

    d10 = (
        (
            sum(d[0:9:2]) * 7
        )
        -
        sum(d[1:8:2])
    ) % 10

    d11 = sum(d[:10]) % 10

    return (
        d[9] == d10
        and
        d[10] == d11
    )


# =========================================================
# TC OCR HATALARINI DÜZELT
# =========================================================

def tc_metni_duzelt(text):
    text = str(text).upper()

    donusum = {
        "O": "0",
        "Q": "0",
        "D": "0",

        "I": "1",
        "İ": "1",
        "L": "1",

        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8"
    }

    sonuc = ""

    for karakter in text:

        if karakter.isdigit():
            sonuc += karakter

        elif karakter in donusum:
            sonuc += donusum[karakter]

    return sonuc


# =========================================================
# PADDLE SONUCUNU OKU
# =========================================================

def paddle_oku(resim):
    """
    PaddleOCR 3.x sonucunu standart hale getirir.
    """

    bulunanlar = []

    try:
        sonuclar = ocr.predict(resim)

    except Exception as e:
        print(
            "PaddleOCR predict hatası:",
            repr(e)
        )
        return []

    for sonuc in sonuclar:

        try:
            data = sonuc.json

        except Exception as e:
            print(
                "OCR JSON alınamadı:",
                repr(e)
            )
            continue

        if (
            isinstance(data, dict)
            and
            "res" in data
        ):
            data = data["res"]

        if not isinstance(data, dict):
            continue

        metinler = data.get(
            "rec_texts",
            []
        )

        guvenler = data.get(
            "rec_scores",
            []
        )

        kutular = data.get(
            "rec_polys",
            []
        )

        for i, text in enumerate(
            metinler
        ):

            text = str(text).strip()

            if not text:
                continue

            try:
                conf = float(
                    guvenler[i]
                )
            except Exception:
                conf = 0.0

            try:
                box = np.asarray(
                    kutular[i],
                    dtype=np.float32
                )

                xs = box[:, 0]
                ys = box[:, 1]

                x1 = int(xs.min())
                y1 = int(ys.min())
                x2 = int(xs.max())
                y2 = int(ys.max())

            except Exception:

                x1 = 0
                y1 = 0
                x2 = 0
                y2 = 0

            bulunanlar.append({
                "text": text,
                "norm": normalize_text(text),
                "conf": conf,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,

                "w": max(0, x2 - x1),
                "h": max(0, y2 - y1)
            })

    return bulunanlar


# =========================================================
# LABEL KONTROLLERİ
# =========================================================

def soyad_label_mi(item):
    t = item["norm"]

    return (
        "SOYADI" in t
        or
        "SURNAME" in t
    )


def ad_label_mi(item):
    t = item["norm"]

    # Soyadı içindeki ADI yüzünden yanlış eşleşmesin
    if (
        "SOYADI" in t
        or
        "SURNAME" in t
    ):
        return False

    return (
        "ADI" in t
        and
        "GIVEN" in t
    )


# =========================================================
# LABEL BUL
# =========================================================

def label_bul(
    ocr_sonuclari,
    kontrol_fonksiyonu
):

    adaylar = [
        item
        for item in ocr_sonuclari
        if kontrol_fonksiyonu(item)
    ]

    if not adaylar:
        return None

    adaylar.sort(
        key=lambda x: x["conf"],
        reverse=True
    )

    return adaylar[0]


# =========================================================
# İSİM ADAYI MI?
# =========================================================

def isim_adayi_mi(item):
    text = item["text"]

    norm = item["norm"]

    # Başka label alanlarını value sanmasın
    label_kelimeleri = [
        "SOYADI",
        "SURNAME",
        "ADI",
        "GIVEN",
        "DOGUM",
        "DATE OF BIRTH",
        "CINSIYET",
        "GENDER",
        "SERI",
        "DOCUMENT",
        "UYRUGU",
        "NATIONALITY",
        "GECERLILIK",
        "VALID",
        "IMZASI",
        "SIGNATURE",
        "KIMLIK",
        "IDENTITY"
    ]

    if any(
        kelime in norm
        for kelime in label_kelimeleri
    ):
        return False

    # Harf içermeli
    if not re.search(
        r"[A-Za-zÇĞİÖŞÜçğıöşü]",
        text
    ):
        return False

    # Fazla rakam içeriyorsa isim değildir
    rakam_sayisi = sum(
        c.isdigit()
        for c in text
    )

    if rakam_sayisi > 1:
        return False

    return True


# =========================================================
# LABEL ALTINDAKİ EN UYGUN DEĞER
# =========================================================

def label_altindaki_degeri_bul(
    ocr_sonuclari,
    label,
    maksimum_dikey=100
):
    """
    Sabit koordinat kullanmaz.
    Label'ın hemen altında, aynı sütunda bulunan value'yu seçer.
    """

    if label is None:
        return ""

    adaylar = []

    for item in ocr_sonuclari:

        if item is label:
            continue

        if not isim_adayi_mi(
            item
        ):
            continue

        # ---------------------------------------------
        # Label'ın altında mı?
        # ---------------------------------------------

        dikey_fark = (
            item["y1"]
            -
            label["y2"]
        )

        if dikey_fark < -10:
            continue

        if dikey_fark > maksimum_dikey:
            continue

        # ---------------------------------------------
        # Aynı sütunda mı?
        # ---------------------------------------------

        yatay_fark = abs(
            item["x1"]
            -
            label["x1"]
        )

        if yatay_fark > 180:
            continue

        # ---------------------------------------------
        # SKOR
        # ---------------------------------------------

        skor = 0

        # Yakın olan daha iyi
        skor -= dikey_fark * 4

        # X hizası
        skor -= yatay_fark * 0.5

        # OCR güveni
        skor += item["conf"] * 100

        adaylar.append(
            (
                skor,
                item
            )
        )

    if not adaylar:
        return ""

    adaylar.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return adaylar[0][1]["text"]


# =========================================================
# İSİM TEMİZLE
# =========================================================

def isim_temizle(text):
    if not text:
        return ""

    text = str(text).upper()

    text = re.sub(
        r"[^A-ZÇĞİIÖŞÜ\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# TC BUL
# =========================================================

def tc_bul(
    ocr_sonuclari
):

    # -----------------------------------------------------
    # 1. Direkt 11 rakam
    # -----------------------------------------------------

    for item in ocr_sonuclari:

        rakamlar = re.sub(
            r"\D",
            "",
            item["text"]
        )

        if len(rakamlar) == 11:

            if tc_kimlik_gecerli_mi(
                rakamlar
            ):
                return rakamlar

    # -----------------------------------------------------
    # 2. OCR harf-rakam hataları
    # -----------------------------------------------------

    for item in ocr_sonuclari:

        aday = tc_metni_duzelt(
            item["text"]
        )

        if len(aday) == 11:

            if tc_kimlik_gecerli_mi(
                aday
            ):
                return aday

    return "Bulunamadi"


# =========================================================
# PADDLE DEBUG
# =========================================================

def paddle_debug_resmi(
    kart,
    ocr_sonuclari
):

    debug = kart.copy()

    for index, item in enumerate(
        ocr_sonuclari
    ):

        # OCR kutusu
        cv2.rectangle(
            debug,
            (
                item["x1"],
                item["y1"]
            ),
            (
                item["x2"],
                item["y2"]
            ),
            (0, 255, 0),
            2
        )

        # sıra numarası
        cv2.putText(
            debug,
            str(index + 1),
            (
                item["x1"],
                max(
                    20,
                    item["y1"] - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    return debug


# =========================================================
# LABEL DEBUG
# =========================================================

def label_debug_resmi(
    kart,
    ocr_sonuclari,
    soyad_label,
    ad_label
):

    debug = paddle_debug_resmi(
        kart,
        ocr_sonuclari
    )

    # -----------------------------------------------------
    # SOYAD LABEL
    # -----------------------------------------------------

    if soyad_label:

        cv2.rectangle(
            debug,
            (
                soyad_label["x1"],
                soyad_label["y1"]
            ),
            (
                soyad_label["x2"],
                soyad_label["y2"]
            ),
            (0, 255, 255),
            4
        )

        cv2.putText(
            debug,
            "SOYAD",
            (
                soyad_label["x1"],
                max(
                    25,
                    soyad_label["y1"] - 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # -----------------------------------------------------
    # AD LABEL
    # -----------------------------------------------------

    if ad_label:

        cv2.rectangle(
            debug,
            (
                ad_label["x1"],
                ad_label["y1"]
            ),
            (
                ad_label["x2"],
                ad_label["y2"]
            ),
            (255, 255, 0),
            4
        )

        cv2.putText(
            debug,
            "AD",
            (
                ad_label["x1"],
                max(
                    25,
                    ad_label["y1"] - 20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def bilgileri_cimbizla(
    orijinal_kart
):

    if orijinal_kart is None:

        return {
            "tc_no": "Bulunamadi",
            "ad": "Bulunamadi",
            "soyad": "Bulunamadi",
            "guven": "dusuk",
            "paddle_debug": None,
            "debug_resmi": None,
            "tum_ocr": []
        }

    # =====================================================
    # OCR
    # =====================================================

    ocr_sonuclari = paddle_oku(
        orijinal_kart
    )

    # =====================================================
    # LABEL'LAR
    # =====================================================

    soyad_label = label_bul(
        ocr_sonuclari,
        soyad_label_mi
    )

    ad_label = label_bul(
        ocr_sonuclari,
        ad_label_mi
    )

    # =====================================================
    # SOYAD
    # =====================================================

    soyad = label_altindaki_degeri_bul(
        ocr_sonuclari,
        soyad_label,
        maksimum_dikey=80
    )

    soyad = isim_temizle(
        soyad
    )

    # =====================================================
    # AD
    # =====================================================

    ad = label_altindaki_degeri_bul(
        ocr_sonuclari,
        ad_label,
        maksimum_dikey=90
    )

    ad = isim_temizle(
        ad
    )

    # =====================================================
    # TC
    # =====================================================

    tc_no = tc_bul(
        ocr_sonuclari
    )

    # =====================================================
    # BOŞLAR
    # =====================================================

    if not soyad:
        soyad = "Bulunamadi"

    if not ad:
        ad = "Bulunamadi"

    # =====================================================
    # GÜVEN
    # =====================================================

    bulunan = 0

    if tc_no != "Bulunamadi":
        bulunan += 1

    if ad != "Bulunamadi":
        bulunan += 1

    if soyad != "Bulunamadi":
        bulunan += 1

    if bulunan == 3:
        guven = "yuksek"

    elif bulunan >= 1:
        guven = "orta"

    else:
        guven = "dusuk"

    # =====================================================
    # DEBUG
    # =====================================================

    paddle_debug = paddle_debug_resmi(
        orijinal_kart,
        ocr_sonuclari
    )

    debug_resmi = label_debug_resmi(
        orijinal_kart,
        ocr_sonuclari,
        soyad_label,
        ad_label
    )

    # =====================================================
    # HAM OCR LİSTESİ
    # =====================================================

    tum_ocr = []

    for index, item in enumerate(
        ocr_sonuclari
    ):

        tum_ocr.append({
            "no": index + 1,

            "metin": item["text"],

            "guven": round(
                item["conf"],
                3
            ),

            "x": item["x1"],
            "y": item["y1"],

            "x2": item["x2"],
            "y2": item["y2"]
        })

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "tc_no": tc_no,

        "ad": ad,

        "soyad": soyad,

        "guven": guven,

        "paddle_debug": paddle_debug,

        "debug_resmi": debug_resmi,

        "tum_ocr": tum_ocr
    }