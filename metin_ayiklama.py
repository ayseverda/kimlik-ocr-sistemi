import cv2
import re
import time
import easyocr
import numpy as np
from difflib import SequenceMatcher


# =========================================================
# EASYOCR
# =========================================================

reader = easyocr.Reader(
    ["tr", "en"],
    gpu=False
)

# İsim ikinci okumada OCR'ı sadece harflere yönlendiriyoruz.
ISIM_ALLOWLIST = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ÇĞİÖŞÜ"
    " "
)


# =========================================================
# METİN YARDIMCILARI
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()

    text = text.translate(
        str.maketrans({
            "İ": "I",
            "Ş": "S",
            "Ğ": "G",
            "Ü": "U",
            "Ö": "O",
            "Ç": "C"
        })
    )

    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def benzerlik(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


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
    )

    return text.strip()


# =========================================================
# TC
# =========================================================

def tc_kimlik_gecerli_mi(no):
    if (
        not no
        or len(no) != 11
        or not no.isdigit()
        or no[0] == "0"
    ):
        return False

    d = [int(x) for x in no]

    d10 = (
        sum(d[0:9:2]) * 7
        -
        sum(d[1:8:2])
    ) % 10

    d11 = sum(d[:10]) % 10

    return (
        d[9] == d10
        and
        d[10] == d11
    )


def tc_metni_duzelt(text):
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

    for karakter in str(text).upper():

        if karakter.isdigit():
            sonuc += karakter

        elif karakter in donusum:
            sonuc += donusum[karakter]

    return sonuc


def tc_bul(ocr_sonuclari):
    # Önce normal
    for item in ocr_sonuclari:

        rakamlar = re.sub(
            r"\D",
            "",
            item["text"]
        )

        if (
            len(rakamlar) == 11
            and tc_kimlik_gecerli_mi(rakamlar)
        ):
            return rakamlar, item

    # OCR hata düzeltmeli
    for item in ocr_sonuclari:

        aday = tc_metni_duzelt(
            item["text"]
        )

        if (
            len(aday) == 11
            and tc_kimlik_gecerli_mi(aday)
        ):
            return aday, item

    return "Bulunamadi", None


# =========================================================
# EASYOCR - ANA OKUMA
# =========================================================

def easyocr_oku(
    resim,
    offset_x=0,
    offset_y=0
):
    baslangic = time.perf_counter()

    bulunanlar = []

    try:
        sonuclar = reader.readtext(
            resim,
            detail=1,
            paragraph=False,
            decoder="greedy"
        )

    except Exception as e:
        print("EasyOCR hatası:", repr(e))
        return [], 0.0


    for sonuc in sonuclar:

        try:
            box, text, conf = sonuc
        except Exception:
            continue

        text = str(text).strip()

        if not text:
            continue

        box = np.asarray(
            box,
            dtype=np.float32
        )

        x1 = int(box[:, 0].min()) + offset_x
        y1 = int(box[:, 1].min()) + offset_y
        x2 = int(box[:, 0].max()) + offset_x
        y2 = int(box[:, 1].max()) + offset_y

        bulunanlar.append({
            "text": text,
            "norm": normalize_text(text),
            "conf": float(conf),

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })


    sure = (
        time.perf_counter()
        -
        baslangic
    )

    return bulunanlar, sure


# =========================================================
# LABEL BUL
# =========================================================

def fuzzy_label_bul(
    ocr_sonuclari,
    hedefler,
    esik=0.40
):
    """
    Eskiden 0.48 idi.

    Biraz bulanık kartlarda:
        Soyadi / Surname
        Adi / Given Name

    kötü okunabildiği için 0.40'a indirdik.
    """

    en_iyi_item = None
    en_iyi_skor = 0.0

    for item in ocr_sonuclari:

        for hedef in hedefler:

            skor = benzerlik(
                item["text"],
                hedef
            )

            if skor > en_iyi_skor:
                en_iyi_skor = skor
                en_iyi_item = item


    if en_iyi_skor < esik:
        return None

    return en_iyi_item


# =========================================================
# SABİT KİMLİK LABEL'I MI?
# =========================================================

def label_metni_mi(text):
    norm = normalize_text(text)

    yasaklar = [
        "TURKIYE",
        "REPUBLIC",
        "IDENTITY",
        "KIMLIK",
        "CARD",

        "SOYADI",
        "SURNAME",

        "ADI",
        "GIVEN",

        "DATE",
        "BIRTH",

        "GENDER",
        "CINSIYET",

        "DOCUMENT",
        "SERI",

        "NATIONALITY",
        "UYRUGU",

        "VALID",
        "GECERLILIK",

        "SIGNATURE",
        "IMZASI"
    ]

    return any(
        kelime in norm
        for kelime in yasaklar
    )


# =========================================================
# LABEL'DAN SONRA DEĞER BUL
# =========================================================

def label_degerini_bul(
    ocr_sonuclari,
    label,
    maksimum_sonraki=5,
    maksimum_dikey=145
):
    """
    Artık confidence 0.70 şartı YOK.

    Label bulunduysa:
    - konum
    - OCR sırası
    - yatay hizalama
    - confidence

    birlikte değerlendirilir.

    Böylece MEHMET = 0.671 gibi doğru sonuçlar
    çöpe gitmez.

    %G0z = 0.127 gibi çok kötü okunmuş ama
    doğru konumdaki UĞUZ kutusu da aday olabilir.
    """

    if label is None:
        return None


    try:
        label_index = ocr_sonuclari.index(
            label
        )

    except ValueError:
        return None


    adaylar = []

    bitis = min(
        label_index + 1 + maksimum_sonraki,
        len(ocr_sonuclari)
    )


    for i in range(
        label_index + 1,
        bitis
    ):

        item = ocr_sonuclari[i]

        text = item["text"].strip()

        if not text:
            continue


        # Tamamen sayıysa isim olamaz
        sadece_rakam = re.sub(
            r"\D",
            "",
            text
        )

        if (
            sadece_rakam
            and len(sadece_rakam) == len(text)
        ):
            continue


        # Güçlü biçimde başka bir label ise alma
        if label_metni_mi(text):
            continue


        # Label'ın altında olmalı
        dikey_fark = (
            item["y1"]
            -
            label["y2"]
        )

        if (
            dikey_fark < -10
            or
            dikey_fark > maksimum_dikey
        ):
            continue


        # Çok uzak başka sütun olmasın
        yatay_fark = abs(
            item["x1"]
            -
            label["x1"]
        )

        if yatay_fark > 230:
            continue


        index_farki = (
            i - label_index
        )


        # =============================================
        # SKOR
        # =============================================
        #
        # Konuma confidence'tan daha fazla önem veriyoruz.
        #
        # Label'ın hemen altındaki düşük confidence'lı
        # kutu bile seçilebilir.
        # =============================================

        skor = 0

        skor += item["conf"] * 35

        skor -= max(
            0,
            dikey_fark
        ) * 1.5

        skor -= yatay_fark * 0.10

        skor -= (
            index_farki - 1
        ) * 6


        # Hemen altındaki kutuya bonus
        if (
            0 <= dikey_fark <= 55
        ):
            skor += 30


        # Aynı sol hizaya yakınsa bonus
        if yatay_fark <= 70:
            skor += 15


        adaylar.append(
            (
                skor,
                item
            )
        )


    if not adaylar:
        return None


    return max(
        adaylar,
        key=lambda x: x[0]
    )[1]


# =========================================================
# KÜÇÜK İSİM KUTUSUNU TEKRAR OKU
# =========================================================

def isim_kutusunu_tekrar_oku(
    kart,
    item
):
    """
    İlk EasyOCR sonucu düşük confidence ise
    SADECE seçilmiş küçük ad/soyad kutusunu tekrar okur.

    Örneğin:

        %G0z   -> UĞUZ

    Bütün kart ikinci kez OCR'a girmez.
    """

    if item is None:
        return None


    h, w = kart.shape[:2]

    pad_x = 14
    pad_y = 10


    x1 = max(
        0,
        item["x1"] - pad_x
    )

    y1 = max(
        0,
        item["y1"] - pad_y
    )

    x2 = min(
        w,
        item["x2"] + pad_x
    )

    y2 = min(
        h,
        item["y2"] + pad_y
    )


    roi = kart[
        y1:y2,
        x1:x2
    ]


    if roi.size == 0:
        return None


    # =====================================================
    # BÜYÜT
    # =====================================================

    roi = cv2.resize(
        roi,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )


    # =====================================================
    # HAFİF KONTRAST
    # =====================================================

    gri = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8)
    )

    gri = clahe.apply(
        gri
    )


    # =====================================================
    # İKİNCİ EASYOCR
    # =====================================================

    try:

        sonuclar = reader.readtext(
            gri,

            detail=1,
            paragraph=False,

            decoder="greedy",

            allowlist=ISIM_ALLOWLIST,

            text_threshold=0.25,
            low_text=0.20,
            link_threshold=0.20,

            contrast_ths=0.05,
            adjust_contrast=0.7,

            mag_ratio=1.5
        )

    except Exception:
        return None


    if not sonuclar:
        return None


    adaylar = []


    for _, text, conf in sonuclar:

        temiz = isim_temizle(
            text
        )

        if not temiz:
            continue


        harf_sayisi = len(
            re.sub(
                r"[^A-ZÇĞİÖŞÜ]",
                "",
                temiz
            )
        )


        if harf_sayisi < 2:
            continue


        adaylar.append(
            (
                float(conf),
                temiz
            )
        )


    if not adaylar:
        return None


    return max(
        adaylar,
        key=lambda x: x[0]
    )


# =========================================================
# SEÇİLEN DEĞERİ İYİLEŞTİR
# =========================================================

def secilen_isimi_iyilestir(
    kart,
    item
):
    if item is None:
        return None


    ilk_text = isim_temizle(
        item["text"]
    )

    ilk_conf = item["conf"]


    # =============================================
    # ZATEN ÇOK İYİYSE HİÇBİR ŞEY YAPMA
    # =============================================

    temiz_mi = bool(
        re.fullmatch(
            r"[A-ZÇĞİÖŞÜ\s\-]+",
            ilk_text
        )
    )


    if (
        ilk_conf >= 0.72
        and
        temiz_mi
    ):
        return ilk_text


    # =============================================
    # DÜŞÜK GÜVEN -> KÜÇÜK ROI SECOND PASS
    # =============================================

    ikinci = isim_kutusunu_tekrar_oku(
        kart,
        item
    )


    if ikinci is None:

        # İlk sonuç tamamen çöp değilse
        # son çare olarak onu kullan.
        if (
            ilk_conf >= 0.45
            and
            temiz_mi
        ):
            return ilk_text

        return None


    ikinci_conf, ikinci_text = (
        ikinci
    )


    # Yeni OCR çok kötüyse kullanma
    if ikinci_conf < 0.25:

        if (
            ilk_conf >= 0.45
            and
            temiz_mi
        ):
            return ilk_text

        return None


    return ikinci_text


# =========================================================
# FALLBACK
# =========================================================

def fallback_isim_adayi_mi(item):

    text = item["text"].strip()

    if not text:
        return False


    if item["conf"] < 0.45:
        return False


    # Tamamen sayı içeren alanları at
    if re.fullmatch(
        r"[\d\W]+",
        text
    ):
        return False


    if label_metni_mi(text):
        return False


    temiz = isim_temizle(
        text
    )


    harf_sayisi = len(
        re.sub(
            r"[^A-ZÇĞİÖŞÜ]",
            "",
            temiz
        )
    )


    return harf_sayisi >= 3


def relative_fallback(
    ocr_sonuclari,
    tc_item=None
):

    tc_index = None


    if tc_item is not None:

        try:

            tc_index = ocr_sonuclari.index(
                tc_item
            )

        except ValueError:

            tc_index = None


    if tc_index is None:

        for i, item in enumerate(
            ocr_sonuclari
        ):

            rakamlar = re.sub(
                r"\D",
                "",
                item["text"]
            )

            if (
                len(rakamlar) == 11
                and
                tc_kimlik_gecerli_mi(
                    rakamlar
                )
            ):

                tc_index = i
                break


    if tc_index is None:
        return None, None


    adaylar = []


    for item in ocr_sonuclari[
        tc_index + 1:
        tc_index + 10
    ]:

        if fallback_isim_adayi_mi(
            item
        ):

            adaylar.append(
                item
            )


    if len(adaylar) < 2:
        return None, None


    # TC sonrası genel sıra:
    # SOYAD -> AD

    return (
        adaylar[1],
        adaylar[0]
    )


# =========================================================
# AYNI KUTU KONTROLÜ
# =========================================================

def ayni_item_mi(a, b):

    if a is None or b is None:
        return False

    return (
        a["x1"] == b["x1"]
        and
        a["y1"] == b["y1"]
        and
        a["x2"] == b["x2"]
        and
        a["y2"] == b["y2"]
    )


# =========================================================
# DEBUG
# =========================================================

def debug_resmi_olustur(
    kart,
    tc_item,
    soyad_item,
    ad_item
):

    debug = kart.copy()


    alanlar = [
        (
            "TC",
            tc_item,
            (0, 255, 0)
        ),

        (
            "SOYAD",
            soyad_item,
            (0, 255, 255)
        ),

        (
            "AD",
            ad_item,
            (255, 255, 0)
        )
    ]


    for isim, item, renk in alanlar:

        if item is None:
            continue


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

            renk,
            4
        )


        cv2.putText(
            debug,
            isim,

            (
                item["x1"],
                max(
                    25,
                    item["y1"] - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            renk,
            2,
            cv2.LINE_AA
        )


    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def bilgileri_cimbizla(
    orijinal_kart,
    debug=False
):

    if orijinal_kart is None:

        return {
            "tc_no": "Bulunamadi",
            "ad": "Bulunamadi",
            "soyad": "Bulunamadi",
            "guven": "dusuk",

            "ocr_suresi": 0.0,

            "debug_resmi": None,
            "tum_ocr": []
        }


    # =====================================================
    # 1. ORTAK OCR ROI
    # =====================================================

    h, w = orijinal_kart.shape[:2]


    x1 = int(w * 0.02)
    x2 = int(w * 0.70)

    y1 = int(h * 0.15)
    y2 = int(h * 0.62)


    bilgi_bolgesi = orijinal_kart[
        y1:y2,
        x1:x2
    ]


    # =====================================================
    # 2. ANA EASYOCR
    # =====================================================

    (
        ocr_sonuclari,
        ocr_suresi
    ) = easyocr_oku(
        bilgi_bolgesi,
        offset_x=x1,
        offset_y=y1
    )


    # =====================================================
    # 3. TC
    # =====================================================

    tc_no, tc_item = tc_bul(
        ocr_sonuclari
    )


    # =====================================================
    # 4. LABEL'LAR
    # =====================================================

    soyad_label = fuzzy_label_bul(
        ocr_sonuclari,
        [
            "SOYADI SURNAME",
            "SURNAME",
            "SOYADI"
        ]
    )


    ad_label = fuzzy_label_bul(
        ocr_sonuclari,
        [
            "ADI GIVEN NAME S",
            "ADI GIVEN NAMES",
            "ADI GIVEN NAME",

            "GIVEN NAME S",
            "GIVEN NAMES",
            "GIVEN NAME"
        ]
    )


    # =====================================================
    # 5. LABEL TABANLI DEĞERLER
    # =====================================================

    soyad_item = label_degerini_bul(
        ocr_sonuclari,
        soyad_label,
        maksimum_sonraki=5,
        maksimum_dikey=145
    )


    ad_item = label_degerini_bul(
        ocr_sonuclari,
        ad_label,
        maksimum_sonraki=5,
        maksimum_dikey=145
    )


    # =====================================================
    # 6. FALLBACK
    # =====================================================

    if (
        soyad_item is None
        or
        ad_item is None
        or
        ayni_item_mi(
            ad_item,
            soyad_item
        )
    ):

        (
            fallback_ad,
            fallback_soyad
        ) = relative_fallback(
            ocr_sonuclari,
            tc_item
        )


        if soyad_item is None:
            soyad_item = fallback_soyad


        if ad_item is None:
            ad_item = fallback_ad


        if ayni_item_mi(
            ad_item,
            soyad_item
        ):

            ad_item = fallback_ad
            soyad_item = fallback_soyad


    # =====================================================
    # 7. LOW CONF SECOND PASS
    # =====================================================

    soyad = secilen_isimi_iyilestir(
        orijinal_kart,
        soyad_item
    )


    ad = secilen_isimi_iyilestir(
        orijinal_kart,
        ad_item
    )


    ad = (
        ad
        if ad
        else "Bulunamadi"
    )


    soyad = (
        soyad
        if soyad
        else "Bulunamadi"
    )


    # =====================================================
    # 8. GÜVEN
    # =====================================================

    bulunan = sum([
        tc_no != "Bulunamadi",
        ad != "Bulunamadi",
        soyad != "Bulunamadi"
    ])


    if bulunan == 3:
        guven = "yuksek"

    elif bulunan > 0:
        guven = "orta"

    else:
        guven = "dusuk"


    # =====================================================
    # 9. DEBUG
    # =====================================================

    debug_resmi = None
    tum_ocr = []


    if debug:

        debug_resmi = debug_resmi_olustur(
            orijinal_kart,
            tc_item,
            soyad_item,
            ad_item
        )


        tum_ocr = [
            {
                "no": index + 1,

                "text": item["text"],

                "conf": round(
                    item["conf"],
                    3
                ),

                "x1": item["x1"],
                "y1": item["y1"],

                "x2": item["x2"],
                "y2": item["y2"]
            }

            for index, item
            in enumerate(
                ocr_sonuclari
            )
        ]


    return {
    "tc_no": tc_no,
    "ad": ad,
    "soyad": soyad,
    "guven": guven,

    # Ad için EasyOCR güven değeri
    "ad_conf": (
        float(ad_item["conf"])
        if ad_item is not None
        else 0.0
    ),

    # Soyad için EasyOCR güven değeri
    "soyad_conf": (
        float(soyad_item["conf"])
        if soyad_item is not None
        else 0.0
    ),

    # OCR işlem süresi
    "ocr_suresi": ocr_suresi,

    # Debug bilgileri
    "debug_resmi": debug_resmi,
    "tum_ocr": tum_ocr
}