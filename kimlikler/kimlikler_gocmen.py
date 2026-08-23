import re

from datetime import datetime, timezone
from difflib import SequenceMatcher


# =========================================================
# NORMALIZE
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
            "Ç": "C",
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
# KUTU
# =========================================================

def merkez_y(item):

    return (
        item["y1"]
        +
        item["y2"]
    ) / 2


def merkez_x(item):

    return (
        item["x1"]
        +
        item["x2"]
    ) / 2


def yukseklik(item):

    return max(
        1,
        item["y2"] - item["y1"]
    )


# =========================================================
# LABEL TİPİ
# =========================================================

def label_tipini_bul(text):

    norm = normalize_text(
        text
    )

    if not norm:
        return None


    if (
        "YABANCI" in norm
        and
        "KIMLIK" in norm
    ):
        return "kimlik_no"


    if "SOYAD" in norm:
        return "soyad"


    if (
        "BABA" in norm
        and
        "AD" in norm
    ):
        return "baba"


    if (
        "ANNE" in norm
        and
        "AD" in norm
    ):
        return "anne"


    if (
        "BELGENIN" in norm
        and
        "GECERLILIK" in norm
    ):
        return "gecerlilik"


    if (
        "BASLANGIC" in norm
        and
        "BITIS" in norm
    ):
        return "gecerlilik"


    if (
        norm == "ADI"
        or
        norm == "ADI "
        or
        norm == "AD"
    ):
        return "ad"


    hedefler = {
        "kimlik_no":
            "YABANCI KIMLIK NO",

        "soyad":
            "SOYADI",

        "ad":
            "ADI",

        "baba":
            "BABA ADI",

        "anne":
            "ANNE ADI",

        "gecerlilik":
            "BELGENIN GECERLILIK TARIHI",
    }


    skorlar = {
        tip:
            benzerlik(
                norm,
                hedef
            )

        for tip, hedef
        in hedefler.items()
    }


    tip = max(
        skorlar,
        key=skorlar.get
    )

    skor = skorlar[
        tip
    ]


    if tip == "ad":

        if any(
            x in norm
            for x in [
                "SOY",
                "BABA",
                "ANNE"
            ]
        ):
            return None

        if len(norm) > 6:
            return None

        if skor < 0.65:
            return None

        return "ad"


    if skor < 0.58:
        return None


    return tip


# =========================================================
# SATIR
# =========================================================

def satirlara_grupla(
    ocr_sonuclari
):

    sirali = sorted(
        ocr_sonuclari,
        key=lambda x: (
            merkez_y(x),
            x["x1"]
        )
    )


    satirlar = []


    for item in sirali:

        bulundu = False

        item_y = merkez_y(
            item
        )


        for satir in satirlar:

            satir_y = sum(
                merkez_y(x)
                for x in satir
            ) / len(satir)

            ort_h = sum(
                yukseklik(x)
                for x in satir
            ) / len(satir)

            tolerans = max(
                22,
                ort_h * 0.85
            )


            if abs(
                item_y - satir_y
            ) <= tolerans:

                satir.append(
                    item
                )

                bulundu = True
                break


        if not bulundu:

            satirlar.append(
                [item]
            )


    for satir in satirlar:

        satir.sort(
            key=lambda x: x["x1"]
        )


    return satirlar


# =========================================================
# LABEL BUL
# =========================================================

def satir_label_bul(
    satir,
    tip
):

    adaylar = []


    for item in satir:

        bulunan_tip = label_tipini_bul(
            item.get(
                "text",
                ""
            )
        )


        if bulunan_tip != tip:
            continue


        skor = float(
            item.get(
                "conf",
                0.0
            )
        )

        skor -= (
            item["x1"]
            *
            0.0005
        )


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
# İSİM VALUE MI
# =========================================================

def isim_value_mi(item):

    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()


    if not text:
        return False


    if label_tipini_bul(
        text
    ) is not None:
        return False


    if any(
        c.isdigit()
        for c in text
    ):
        return False


    temiz = isim_temizle(
        text
    )


    harfler = re.sub(
        r"[^A-ZÇĞİÖŞÜ]",
        "",
        temiz
    )


    return len(
        harfler
    ) >= 2


# =========================================================
# SATIRDA İSİMLERİ BİRLEŞTİR
# =========================================================

def satir_isim_bul(
    satir,
    label_item
):

    if label_item is None:
        return None


    adaylar = []


    for item in satir:

        if item is label_item:
            continue


        if not isim_value_mi(
            item
        ):
            continue


        # Label'ın sağında
        if (
            merkez_x(item)
            <=
            merkez_x(label_item) + 25
        ):
            continue


        adaylar.append(
            item
        )


    if not adaylar:
        return None


    adaylar.sort(
        key=lambda x: x["x1"]
    )


    parcalar = [
        isim_temizle(
            item["text"]
        )
        for item in adaylar
        if isim_temizle(
            item["text"]
        )
    ]


    if not parcalar:
        return None


    text = " ".join(
        parcalar
    )


    conf = sum(
        float(
            x.get(
                "conf",
                0.0
            )
        )
        for x in adaylar
    ) / len(adaylar)


    return {
        "deger":
            text,

        "item":
            {
                "text":
                    text,

                "conf":
                    conf,

                "x1":
                    min(
                        x["x1"]
                        for x in adaylar
                    ),

                "y1":
                    min(
                        x["y1"]
                        for x in adaylar
                    ),

                "x2":
                    max(
                        x["x2"]
                        for x in adaylar
                    ),

                "y2":
                    max(
                        x["y2"]
                        for x in adaylar
                    )
            },

        "items":
            adaylar
    }


# =========================================================
# YABANCI NO
# =========================================================

def yabanci_no_metni_duzelt(text):

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
        "B": "8",
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


def yabanci_no_gecerli_mi(no):

    return (
        no is not None
        and
        no.isdigit()
        and
        len(no) == 11
        and
        no.startswith("99")
    )


def yabanci_no_bul(
    ocr_sonuclari
):

    # Direkt
    for item in ocr_sonuclari:

        rakamlar = re.sub(
            r"\D",
            "",
            str(
                item.get(
                    "text",
                    ""
                )
            )
        )


        if yabanci_no_gecerli_mi(
            rakamlar
        ):

            return {
                "deger":
                    rakamlar,

                "item":
                    item
            }


    # OCR correction
    for item in ocr_sonuclari:

        aday = yabanci_no_metni_duzelt(
            item.get(
                "text",
                ""
            )
        )


        if yabanci_no_gecerli_mi(
            aday
        ):

            return {
                "deger":
                    aday,

                "item":
                    item
            }


    return None


# =========================================================
# TARİH
# =========================================================

def tarihleri_bul(text):

    if not text:
        return []


    bulunanlar = re.findall(
        r"(?<!\d)"
        r"(\d{1,2})\s*[./\-]\s*"
        r"(\d{1,2})\s*[./\-]\s*"
        r"(\d{4})"
        r"(?!\d)",
        str(text)
    )


    tarihler = []


    for gun, ay, yil in bulunanlar:

        try:

            tarihler.append(
                datetime(
                    int(yil),
                    int(ay),
                    int(gun)
                ).date()
            )

        except ValueError:
            pass


    return tarihler


def tarih_metne_cevir(tarih):

    if tarih is None:
        return "Bulunamadi"

    return tarih.strftime(
        "%d.%m.%Y"
    )


def utc_bugun():

    return datetime.now(
        timezone.utc
    ).date()


# =========================================================
# TARİH DEBUG KUTULARINI AYIR
# =========================================================

def tarih_debug_itemlari_olustur(
    satir,
    tarihler
):

    # OCR zaten iki ayrı kutu verdiyse
    tarihli_itemlar = []


    for item in satir:

        bulunan = tarihleri_bul(
            item.get(
                "text",
                ""
            )
        )

        if bulunan:

            tarihli_itemlar.append(
                (
                    item,
                    bulunan
                )
            )


    # İki fiziksel kutu
    if len(tarihli_itemlar) >= 2:

        return [
            tarihli_itemlar[0][0],
            tarihli_itemlar[1][0]
        ]


    # Tek OCR kutusunda iki tarih varsa
    # kutuyu yaklaşık iki parçaya böl
    if (
        len(tarihli_itemlar) == 1
        and
        len(
            tarihli_itemlar[0][1]
        ) >= 2
    ):

        item = tarihli_itemlar[0][0]

        orta = int(
            (
                item["x1"]
                +
                item["x2"]
            )
            /
            2
        )


        ilk = {
            **item,
            "x2":
                orta
        }


        ikinci = {
            **item,
            "x1":
                orta
        }


        return [
            ilk,
            ikinci
        ]


    return []


# =========================================================
# GEÇERLİLİK
# =========================================================

def gecerlilik_tarihi_bul(
    ocr_sonuclari
):

    satirlar = satirlara_grupla(
        ocr_sonuclari
    )


    adaylar = []


    for satir in satirlar:

        tarihler = []


        for item in satir:

            tarihler.extend(
                tarihleri_bul(
                    item.get(
                        "text",
                        ""
                    )
                )
            )


        # Geçerlilik satırında iki tarih olmak zorunda
        if len(tarihler) < 2:
            continue


        satir_y = sum(
            merkez_y(x)
            for x in satir
        ) / len(satir)


        # Aynı satırda label var mı?
        label_skor = 0.0


        for item in satir:

            tip = label_tipini_bul(
                item.get(
                    "text",
                    ""
                )
            )


            if tip == "gecerlilik":

                label_skor = max(
                    label_skor,
                    200
                )


        # Bazen label iki satıra bölünüyor.
        # Yakındaki üst/alt satırda da ara.
        if label_skor == 0:

            for diger in satirlar:

                diger_y = sum(
                    merkez_y(x)
                    for x in diger
                ) / len(diger)


                if abs(
                    diger_y - satir_y
                ) > 90:
                    continue


                for item in diger:

                    if (
                        label_tipini_bul(
                            item.get(
                                "text",
                                ""
                            )
                        )
                        ==
                        "gecerlilik"
                    ):

                        label_skor = 150
                        break


        skor = (
            300
            +
            label_skor
            +
            satir_y * 0.03
        )


        adaylar.append(
            (
                skor,
                satir,
                tarihler
            )
        )


    if not adaylar:

        return {
            "baslangic":
                None,

            "bitis":
                None,

            "gecerli":
                None,

            "durum":
                "kontrol_edilemedi",

            "items":
                []
        }


    _, satir, tarihler = max(
        adaylar,
        key=lambda x: x[0]
    )


    # soldan sağa
    baslangic = tarihler[0]
    bitis = tarihler[1]


    bugun = utc_bugun()


    if bugun < baslangic:

        gecerli = False
        durum = "henuz_baslamadi"

    elif bugun > bitis:

        gecerli = False
        durum = "suresi_gecmis"

    else:

        gecerli = True
        durum = "gecerli"


    debug_items = tarih_debug_itemlari_olustur(
        satir,
        tarihler
    )


    return {
        "baslangic":
            baslangic,

        "bitis":
            bitis,

        "gecerli":
            gecerli,

        "durum":
            durum,

        "items":
            debug_items
    }


# =========================================================
# ANA
# =========================================================

def gocmen_bilgilerini_bul(
    ocr_sonuclari
):

    satirlar = satirlara_grupla(
        ocr_sonuclari
    )


    # =====================================================
    # YKN
    # =====================================================

    no_sonuc = yabanci_no_bul(
        ocr_sonuclari
    )


    # =====================================================
    # AD
    # =====================================================

    ad_sonuc = None
    ad_label = None


    for satir in satirlar:

        label = satir_label_bul(
            satir,
            "ad"
        )

        if label is None:
            continue


        sonuc = satir_isim_bul(
            satir,
            label
        )


        if sonuc is not None:

            ad_label = label
            ad_sonuc = sonuc
            break


    # =====================================================
    # SOYAD
    # =====================================================

    soyad_sonuc = None
    soyad_label = None


    for satir in satirlar:

        label = satir_label_bul(
            satir,
            "soyad"
        )

        if label is None:
            continue


        sonuc = satir_isim_bul(
            satir,
            label
        )


        if sonuc is not None:

            soyad_label = label
            soyad_sonuc = sonuc
            break


    # =====================================================
    # GEÇERLİLİK
    # =====================================================

    gecerlilik = gecerlilik_tarihi_bul(
        ocr_sonuclari
    )


    kimlik_no = (
        no_sonuc["deger"]
        if no_sonuc
        else "Bulunamadi"
    )


    ad = (
        ad_sonuc["deger"]
        if ad_sonuc
        else "Bulunamadi"
    )


    soyad = (
        soyad_sonuc["deger"]
        if soyad_sonuc
        else "Bulunamadi"
    )


    kimlik_no_conf = (
        float(
            no_sonuc[
                "item"
            ].get(
                "conf",
                0.0
            )
        )
        if no_sonuc
        else 0.0
    )


    ad_conf = (
        float(
            ad_sonuc[
                "item"
            ].get(
                "conf",
                0.0
            )
        )
        if ad_sonuc
        else 0.0
    )


    soyad_conf = (
        float(
            soyad_sonuc[
                "item"
            ].get(
                "conf",
                0.0
            )
        )
        if soyad_sonuc
        else 0.0
    )


    bulunan = sum([
        kimlik_no != "Bulunamadi",
        ad != "Bulunamadi",
        soyad != "Bulunamadi"
    ])


    if bulunan == 3:

        if min(
            kimlik_no_conf,
            ad_conf,
            soyad_conf
        ) >= 0.50:

            guven = "yuksek"

        else:

            guven = "orta"


    elif bulunan > 0:

        guven = "orta"


    else:

        guven = "dusuk"


    return {
        "kimlik_no":
            kimlik_no,

        "ad":
            ad,

        "soyad":
            soyad,

        "guven":
            guven,

        "kimlik_no_conf":
            kimlik_no_conf,

        "ad_conf":
            ad_conf,

        "soyad_conf":
            soyad_conf,

        "baslangic_tarihi":
            tarih_metne_cevir(
                gecerlilik[
                    "baslangic"
                ]
            ),

        "bitis_tarihi":
            tarih_metne_cevir(
                gecerlilik[
                    "bitis"
                ]
            ),

        "belge_gecerli":
            gecerlilik[
                "gecerli"
            ],

        "gecerlilik_durumu":
            gecerlilik[
                "durum"
            ],

        "kimlik_no_item":
            (
                no_sonuc[
                    "item"
                ]
                if no_sonuc
                else None
            ),

        "ad_item":
            (
                ad_sonuc[
                    "item"
                ]
                if ad_sonuc
                else None
            ),

        "soyad_item":
            (
                soyad_sonuc[
                    "item"
                ]
                if soyad_sonuc
                else None
            ),

        "ad_label":
            ad_label,

        "soyad_label":
            soyad_label,

        "gecerlilik_items":
            gecerlilik[
                "items"
            ],
    }