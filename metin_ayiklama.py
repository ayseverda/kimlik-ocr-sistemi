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


# =========================================================
# METİN YARDIMCILARI
# =========================================================

def normalize_text(text):
    """
    Sadece karşılaştırma için.
    Gerçek sonuçtaki Türkçe karakterlere dokunmaz.
    """

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

    text = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def isim_temizle(text):

    if not text:
        return ""

    text = str(text).upper().strip()

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


def benzerlik(a, b):

    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


# =========================================================
# BÜYÜK HARF ORANI
# =========================================================

def buyuk_harf_orani(text):
    """
    Kimlikte gerçek AD / SOYAD değerleri:
        MERCAN
        AYNUR
        GÜLCEMAL
        AYŞE VERDA

    gibi büyük harfle yazılıyor.

    Label'lar ise:
        Soyadı
        Surname
        Given Name(s)

    gibi title-case / küçük harf ağırlıklı.
    """

    harfler = [
        c
        for c in str(text)
        if c.isalpha()
    ]

    if not harfler:
        return 0.0

    buyuk = sum(
        1
        for c in harfler
        if c.isupper()
    )

    return (
        buyuk
        /
        len(harfler)
    )


# =========================================================
# KUTU BOYUTLARI
# =========================================================

def kutu_yuksekligi(item):

    if item is None:
        return 0

    return max(
        0,
        item["y2"] - item["y1"]
    )


def kutu_genisligi(item):

    if item is None:
        return 0

    return max(
        0,
        item["x2"] - item["x1"]
    )


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

    d = [
        int(x)
        for x in no
    ]

    d10 = (
        sum(d[0:9:2]) * 7
        -
        sum(d[1:8:2])
    ) % 10

    d11 = (
        sum(d[:10]) % 10
    )

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

            sonuc += donusum[
                karakter
            ]

    return sonuc


def tc_bul(ocr_sonuclari):

    # 1. Direkt
    for item in ocr_sonuclari:

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

            return (
                rakamlar,
                item
            )


    # 2. OCR düzeltmeli
    for item in ocr_sonuclari:

        aday = tc_metni_duzelt(
            item["text"]
        )

        if (
            len(aday) == 11
            and
            tc_kimlik_gecerli_mi(
                aday
            )
        ):

            return (
                aday,
                item
            )


    return (
        "Bulunamadi",
        None
    )


# =========================================================
# EASYOCR
# =========================================================

def easyocr_oku(
    resim,
    offset_x=0,
    offset_y=0
):

    baslangic = (
        time.perf_counter()
    )

    bulunanlar = []


    try:

        sonuclar = reader.readtext(
            resim,
            detail=1,
            paragraph=False,
            decoder="greedy"
        )

    except Exception as e:

        print(
            "EasyOCR hatası:",
            repr(e)
        )

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


        x1 = (
            int(box[:, 0].min())
            +
            offset_x
        )

        y1 = (
            int(box[:, 1].min())
            +
            offset_y
        )

        x2 = (
            int(box[:, 0].max())
            +
            offset_x
        )

        y2 = (
            int(box[:, 1].max())
            +
            offset_y
        )


        bulunanlar.append({
            "text":
                text,

            "norm":
                normalize_text(
                    text
                ),

            "conf":
                float(conf),

            "x1":
                x1,

            "y1":
                y1,

            "x2":
                x2,

            "y2":
                y2
        })


    sure = (
        time.perf_counter()
        -
        baslangic
    )


    return (
        bulunanlar,
        sure
    )


# =========================================================
# LABEL HEDEFLERİ
# =========================================================

SOYAD_LABEL_HEDEFLERI = [
    "SOYADI SURNAME",
    "SURNAME",
    "SOYADI"
]


AD_LABEL_HEDEFLERI = [
    "ADI GIVEN NAME S",
    "ADI GIVEN NAMES",
    "ADI GIVEN NAME",

    "GIVEN NAME S",
    "GIVEN NAMES",
    "GIVEN NAME"
]


# =========================================================
# LABEL BUL
# =========================================================

def fuzzy_label_bul(
    ocr_sonuclari,
    hedefler,
    esik=0.40
):
    """
    Label confidence düşük olabilir.

    Burada OCR confidence'a değil,
    metin benzerliğine bakıyoruz.
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
# LABEL BENZERİ METİN Mİ?
# =========================================================

def label_metni_mi(text):

    norm = normalize_text(
        text
    )

    if not norm:

        return False


    sabitler = [
        "TURKIYE",
        "REPUBLIC",
        "IDENTITY",
        "KIMLIK",
        "CARD",

        "SOYADI",
        "SURNAME",

        "ADI",
        "GIVEN",
        "NAME",

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


    if any(
        kelime in norm
        for kelime in sabitler
    ):

        return True


    return False


# =========================================================
# VALUE ADAYI
# =========================================================

def isim_value_adayi_mi(
    item,
    label_item=None
):
    """
    Buradaki EN ÖNEMLİ YENİ KURAL:

    AD / SOYAD değerleri büyük harfli.

    Böylece:
        Soyadı       ❌
        Surname      ❌
        Given Name   ❌

    ama:
        MERCAN       ✅
        AYNUR        ✅
        GÜLCEMAL     ✅
    """

    if item is None:

        return False


    text = str(
        item["text"]
    ).strip()


    if not text:

        return False


    # -----------------------------------------------------
    # LABEL METNİ İSE ASLA VALUE OLAMAZ
    # -----------------------------------------------------

    if label_metni_mi(
        text
    ):

        return False


    # -----------------------------------------------------
    # TAMAMEN SAYI / SEMBOL
    # -----------------------------------------------------

    if re.fullmatch(
        r"[\d\W]+",
        text
    ):

        return False


    # -----------------------------------------------------
    # RAKAM İÇERİYORSA ALMA
    # -----------------------------------------------------

    if any(
        c.isdigit()
        for c in text
    ):

        return False


    # -----------------------------------------------------
    # HARF SAYISI
    # -----------------------------------------------------

    temiz = isim_temizle(
        text
    )


    harfler = re.sub(
        r"[^A-ZÇĞİÖŞÜ]",
        "",
        temiz
    )


    if len(harfler) < 2:

        return False


    # -----------------------------------------------------
    # BÜYÜK HARF ORANI
    #
    # %80 ve üstünü değer kabul ediyoruz.
    # -----------------------------------------------------

    oran = buyuk_harf_orani(
        text
    )


    if oran < 0.80:

        return False


    # -----------------------------------------------------
    # VALUE GENELDE LABEL'DAN DAHA BÜYÜK YAZILIYOR
    #
    # Bunu sert eleme değil küçük güvenlik olarak kullan.
    # -----------------------------------------------------

    if label_item is not None:

        value_h = kutu_yuksekligi(
            item
        )

        label_h = kutu_yuksekligi(
            label_item
        )


        # Çok bariz şekilde label'dan küçükse alma.
        if (
            label_h > 0
            and
            value_h
            <
            label_h * 0.75
        ):

            return False


    return True


# =========================================================
# LABEL ALTINDAKİ VALUE'YU BUL
# =========================================================

def label_degerini_bul(
    ocr_sonuclari,
    label_item,
    sonraki_label=None,
    maksimum_dikey=130,
    maksimum_yatay=220
):
    """
    Eski çalışan fikre geri dönüş:

    label bul
       ↓
    hemen altındaki en mantıklı BÜYÜK HARFLİ value'yu al
    """

    if label_item is None:

        return None


    adaylar = []


    for item in ocr_sonuclari:

        if item is label_item:

            continue


        if not isim_value_adayi_mi(
            item,
            label_item
        ):

            continue


        # -------------------------------------------------
        # LABEL'IN ALTINDA OLMALI
        # -------------------------------------------------

        dikey_fark = (
            item["y1"]
            -
            label_item["y2"]
        )


        if (
            dikey_fark < -12
            or
            dikey_fark > maksimum_dikey
        ):

            continue


        # -------------------------------------------------
        # SOYAD İÇİN AD LABEL'IN ALTINA GEÇMESİN
        # -------------------------------------------------

        if (
            sonraki_label is not None
            and
            item["y1"]
            >=
            sonraki_label["y1"]
        ):

            continue


        # -------------------------------------------------
        # AYNI SÜTUN
        # -------------------------------------------------

        yatay_fark = abs(
            item["x1"]
            -
            label_item["x1"]
        )


        if yatay_fark > maksimum_yatay:

            continue


        # -------------------------------------------------
        # PUAN
        #
        # 1. Dikey yakınlık
        # 2. Yatay hizalama
        # 3. Büyük harf oranı
        # 4. Kutunun label'dan büyük olması
        # 5. OCR confidence
        # -------------------------------------------------

        value_h = kutu_yuksekligi(
            item
        )

        label_h = max(
            1,
            kutu_yuksekligi(
                label_item
            )
        )


        boyut_orani = (
            value_h
            /
            label_h
        )


        skor = 0.0


        # Yakınlık en önemli
        skor -= (
            max(
                0,
                dikey_fark
            )
            *
            2.5
        )


        skor -= (
            yatay_fark
            *
            0.20
        )


        # Büyük harf bonusu
        skor += (
            buyuk_harf_orani(
                item["text"]
            )
            *
            35
        )


        # Değer daha büyük fontsa bonus
        skor += (
            min(
                boyut_orani,
                2.0
            )
            *
            12
        )


        # OCR confidence artık sadece yardımcı
        skor += (
            item["conf"]
            *
            15
        )


        # Hemen altında
        if (
            0
            <=
            dikey_fark
            <=
            60
        ):

            skor += 35


        # Sol taraflar hizalı
        if yatay_fark <= 70:

            skor += 20


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
# LABEL OKUNAMAZSA FALLBACK
# =========================================================

def fallback_ad_soyad_bul(
    ocr_sonuclari,
    tc_item
):
    """
    Sadece label gerçekten bulunamazsa kullanılır.

    TC sonrası BÜYÜK HARFLİ metinleri kullanır.
    """

    if tc_item is None:

        return (
            None,
            None
        )


    try:

        tc_index = (
            ocr_sonuclari.index(
                tc_item
            )
        )

    except ValueError:

        return (
            None,
            None
        )


    adaylar = []


    for item in ocr_sonuclari[
        tc_index + 1:
        tc_index + 14
    ]:

        if not isim_value_adayi_mi(
            item
        ):

            continue


        # Çok sağ tarafı alma
        if item["x1"] > 700:

            continue


        adaylar.append(
            item
        )


    if len(adaylar) < 2:

        return (
            None,
            None
        )


    # Yukarıdan aşağı
    adaylar = sorted(
        adaylar,
        key=lambda x: (
            x["y1"],
            x["x1"]
        )
    )


    soyad_item = (
        adaylar[0]
    )

    ad_item = (
        adaylar[1]
    )


    return (
        ad_item,
        soyad_item
    )


# =========================================================
# AYNI KUTU
# =========================================================

def ayni_item_mi(a, b):

    if (
        a is None
        or
        b is None
    ):

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
# TÜRKÇE HARF İÇİN SADECE BİRKAÇ PİKSEL GENİŞLET
# =========================================================

def isim_kutusunu_hafif_genislet(
    kart,
    item
):
    """
    SEÇİMİ DEĞİŞTİRMEZ.

    Sadece seçilmiş kutudan alınan crop:

    üst  +5 px
    alt  +3 px
    sol  +2 px
    sağ  +2 px
    """

    if item is None:

        return None


    h, w = (
        kart.shape[:2]
    )


    ust = 5
    alt = 3

    sol = 2
    sag = 2


    x1 = max(
        0,
        item["x1"] - sol
    )

    y1 = max(
        0,
        item["y1"] - ust
    )

    x2 = min(
        w,
        item["x2"] + sag
    )

    y2 = min(
        h,
        item["y2"] + alt
    )


    roi = kart[
        y1:y2,
        x1:x2
    ]


    if roi.size == 0:

        return None


    return roi


# =========================================================
# TÜRKÇE KARAKTER SECOND PASS
# =========================================================

def turkce_harf_iyilestir(
    kart,
    item
):
    """
    Burada HANGİ KUTUNUN seçildiği değişmez.

    Sadece:

       GULCEMAL
          ↓
       GÜLCEMAL

    gibi düzeltme denenir.
    """

    if item is None:

        return (
            None,
            0.0
        )


    ilk_text = isim_temizle(
        item["text"]
    )

    ilk_conf = float(
        item["conf"]
    )


    if not ilk_text:

        return (
            None,
            ilk_conf
        )


    roi = isim_kutusunu_hafif_genislet(
        kart,
        item
    )


    if roi is None:

        return (
            ilk_text,
            ilk_conf
        )


    roi = cv2.resize(
        roi,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )


    try:

        sonuclar = reader.readtext(
            roi,
            detail=1,
            paragraph=False,
            decoder="greedy"
        )


    except Exception:

        return (
            ilk_text,
            ilk_conf
        )


    if not sonuclar:

        return (
            ilk_text,
            ilk_conf
        )


    uygunlar = []


    for _, ikinci_text, ikinci_conf in sonuclar:

        ikinci_text = isim_temizle(
            ikinci_text
        )


        if not ikinci_text:

            continue


        # =================================================
        # İKİNCİ OCR BAMBAŞKA KELİME ÜRETEMEZ
        # =================================================

        if (
            normalize_text(
                ikinci_text
            )
            !=
            normalize_text(
                ilk_text
            )
        ):

            continue


        uygunlar.append(
            (
                float(
                    ikinci_conf
                ),

                ikinci_text
            )
        )


    if not uygunlar:

        return (
            ilk_text,
            ilk_conf
        )


    ikinci_conf, ikinci_text = max(
        uygunlar,
        key=lambda x: x[0]
    )


    ilk_turkce = len(
        re.findall(
            r"[ÇĞİÖŞÜ]",
            ilk_text
        )
    )


    ikinci_turkce = len(
        re.findall(
            r"[ÇĞİÖŞÜ]",
            ikinci_text
        )
    )


    if (
        ikinci_turkce
        >
        ilk_turkce
    ):

        return (
            ikinci_text,
            ikinci_conf
        )


    return (
        ilk_text,
        ilk_conf
    )


# =========================================================
# DEBUG
# =========================================================

def debug_resmi_olustur(
    kart,

    tc_item,

    soyad_label,
    soyad_item,

    ad_label,
    ad_item
):

    debug = (
        kart.copy()
    )


    # =====================================================
    # TC
    # =====================================================

    if tc_item is not None:

        cv2.rectangle(
            debug,

            (
                tc_item["x1"],
                tc_item["y1"]
            ),

            (
                tc_item["x2"],
                tc_item["y2"]
            ),

            (0, 255, 0),
            4
        )


        cv2.putText(
            debug,
            "TC",

            (
                tc_item["x1"],
                max(
                    25,
                    tc_item["y1"] - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # =====================================================
    # SOYAD LABEL
    # =====================================================

    if soyad_label is not None:

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

            (0, 180, 255),
            2
        )


        cv2.putText(
            debug,
            "SOYAD LABEL",

            (
                soyad_label["x1"],
                max(
                    25,
                    soyad_label["y1"] - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1
        )


    # =====================================================
    # SOYAD VALUE
    # =====================================================

    if soyad_item is not None:

        cv2.rectangle(
            debug,

            (
                soyad_item["x1"],
                soyad_item["y1"]
            ),

            (
                soyad_item["x2"],
                soyad_item["y2"]
            ),

            (0, 255, 255),
            4
        )


        cv2.putText(
            debug,
            "SOYAD",

            (
                soyad_item["x1"],
                max(
                    25,
                    soyad_item["y1"] - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # =====================================================
    # AD LABEL
    # =====================================================

    if ad_label is not None:

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

            (255, 180, 0),
            2
        )


        cv2.putText(
            debug,
            "AD LABEL",

            (
                ad_label["x1"],
                max(
                    25,
                    ad_label["y1"] - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 0),
            1
        )


    # =====================================================
    # AD VALUE
    # =====================================================

    if ad_item is not None:

        cv2.rectangle(
            debug,

            (
                ad_item["x1"],
                ad_item["y1"]
            ),

            (
                ad_item["x2"],
                ad_item["y2"]
            ),

            (255, 255, 0),
            4
        )


        cv2.putText(
            debug,
            "AD",

            (
                ad_item["x1"],
                max(
                    25,
                    ad_item["y1"] - 10
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def bilgileri_cimbizla(
    orijinal_kart,
    sayfa_no=None,
    debug=False
):

    if orijinal_kart is None:

        return {
            "sayfa_no":
                sayfa_no,

            "tc_no":
                "Bulunamadi",

            "ad":
                "Bulunamadi",

            "soyad":
                "Bulunamadi",

            "guven":
                "dusuk",

            "ad_conf":
                0.0,

            "soyad_conf":
                0.0,

            "ocr_suresi":
                0.0,

            "debug_resmi":
                None,

            "tum_ocr":
                []
        }


    # =====================================================
    # 1. OCR BÖLGESİ
    # =====================================================

    h, w = (
        orijinal_kart.shape[:2]
    )


    x1 = int(
        w * 0.02
    )

    x2 = int(
        w * 0.70
    )

    y1 = int(
        h * 0.15
    )

    y2 = int(
        h * 0.62
    )


    bilgi_bolgesi = (
        orijinal_kart[
            y1:y2,
            x1:x2
        ]
    )


    # =====================================================
    # 2. EASYOCR
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

    (
        tc_no,
        tc_item
    ) = tc_bul(
        ocr_sonuclari
    )


    # =====================================================
    # 4. LABEL'LARI BUL
    # =====================================================

    soyad_label = (
        fuzzy_label_bul(
            ocr_sonuclari,

            SOYAD_LABEL_HEDEFLERI,

            esik=0.40
        )
    )


    ad_label = (
        fuzzy_label_bul(
            ocr_sonuclari,

            AD_LABEL_HEDEFLERI,

            esik=0.40
        )
    )


    # =====================================================
    # 5. ESKİ MANTIĞA DÖN:
    #
    # LABEL
    #   ↓
    # HEMEN ALTINDAKİ BÜYÜK HARFLİ VALUE
    # =====================================================

    soyad_item = (
        label_degerini_bul(
            ocr_sonuclari,

            soyad_label,

            sonraki_label=(
                ad_label
            ),

            maksimum_dikey=130,
            maksimum_yatay=220
        )
    )


    ad_item = (
        label_degerini_bul(
            ocr_sonuclari,

            ad_label,

            sonraki_label=None,

            maksimum_dikey=140,
            maksimum_yatay=220
        )
    )


    # =====================================================
    # 6. FALLBACK
    #
    # LABEL / VALUE gerçekten bulunamadığında.
    # =====================================================

    if (
        soyad_item is None
        or
        ad_item is None
    ):

        (
            fallback_ad,
            fallback_soyad
        ) = fallback_ad_soyad_bul(
            ocr_sonuclari,
            tc_item
        )


        if soyad_item is None:

            soyad_item = (
                fallback_soyad
            )


        if ad_item is None:

            ad_item = (
                fallback_ad
            )


    # =====================================================
    # 7. AYNI KUTU OLMASIN
    # =====================================================

    if ayni_item_mi(
        ad_item,
        soyad_item
    ):

        ad_item = None
        soyad_item = None


    # =====================================================
    # 8. TÜRKÇE HARF İYİLEŞTİR
    #
    # BURADAN SONRA KUTU SEÇİMİ DEĞİŞMEZ.
    # =====================================================

    (
        ad,
        ad_conf
    ) = turkce_harf_iyilestir(
        orijinal_kart,
        ad_item
    )


    (
        soyad,
        soyad_conf
    ) = turkce_harf_iyilestir(
        orijinal_kart,
        soyad_item
    )


    if not ad:

        ad = "Bulunamadi"


    if not soyad:

        soyad = "Bulunamadi"


    # =====================================================
    # 9. GÜVEN
    # =====================================================

    bulunan = sum([
        tc_no != "Bulunamadi",
        ad != "Bulunamadi",
        soyad != "Bulunamadi"
    ])


    if bulunan == 3:

        if (
            ad_conf < 0.50
            or
            soyad_conf < 0.50
        ):

            guven = "orta"

        else:

            guven = "yuksek"


    elif bulunan > 0:

        guven = "orta"


    else:

        guven = "dusuk"


    # =====================================================
    # 10. DEBUG
    # =====================================================

    debug_resmi = None
    tum_ocr = []


    if debug:

        debug_resmi = (
            debug_resmi_olustur(
                orijinal_kart,

                tc_item,

                soyad_label,
                soyad_item,

                ad_label,
                ad_item
            )
        )


        tum_ocr = [
            {
                "no":
                    index + 1,

                "text":
                    item["text"],

                "conf":
                    round(
                        item["conf"],
                        3
                    ),

                "buyuk_harf":
                    round(
                        buyuk_harf_orani(
                            item["text"]
                        ),
                        2
                    ),

                "x1":
                    item["x1"],

                "y1":
                    item["y1"],

                "x2":
                    item["x2"],

                "y2":
                    item["y2"]
            }

            for index, item
            in enumerate(
                ocr_sonuclari
            )
        ]


    # =====================================================
    # RETURN
    # =====================================================

    return {
        "sayfa_no":
            sayfa_no,

        "tc_no":
            tc_no,

        "ad":
            ad,

        "soyad":
            soyad,

        "guven":
            guven,

        "ad_conf":
            ad_conf,

        "soyad_conf":
            soyad_conf,

        "ocr_suresi":
            ocr_suresi,

        "debug_resmi":
            debug_resmi,

        "tum_ocr":
            tum_ocr
    }