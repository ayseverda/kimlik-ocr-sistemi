import re
from difflib import SequenceMatcher

from kimlikler.kimlikler_tc import tc_bul


def normalize_text(text):
    text = str(text or "").upper().strip()

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
    text = str(text or "").upper().strip()

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


def merkez_y(item):
    return (
        item["y1"]
        +
        item["y2"]
    ) / 2.0


def merkez_x(item):
    return (
        item["x1"]
        +
        item["x2"]
    ) / 2.0


def yukseklik(item):
    return max(
        1,
        item["y2"] - item["y1"]
    )


def tc_label_mi(text):
    norm = normalize_text(text)

    if not norm:
        return False

    if (
        "KIMLIK" in norm
        and
        (
            "NO" in norm
            or
            "N0" in norm
        )
    ):
        return True

    hedefler = [
        "TC KIMLIK NO",
        "T C KIMLIK NO",
        "TC KIMLIK NUMARASI",
    ]

    return max(
        benzerlik(norm, hedef)
        for hedef in hedefler
    ) >= 0.58


def soyad_label_mi(text):
    norm = normalize_text(text)

    if not norm:
        return False

    if (
        "SOYAD" in norm
        or
        "SOYADI" in norm
    ):
        return True

    return max(
        benzerlik(norm, "SOYADI"),
        benzerlik(norm, "SOYAD")
    ) >= 0.64


def ad_label_mi(text):
    norm = normalize_text(text)

    if not norm:
        return False

    yasaklar = [
        "SOY",
        "BABA",
        "ANA",
        "DOGUM",
        "KIMLIK",
        "SERI",
    ]

    if any(
        x in norm
        for x in yasaklar
    ):
        return False

    if norm in {
        "ADI",
        "AD",
        "AD1",
        "A DI",
        "A0I",
        "A01",
        "AOI",
    }:
        return True

    if len(norm) <= 5:
        return (
            benzerlik(
                norm,
                "ADI"
            )
            >= 0.68
        )

    return False


def herhangi_bir_label_mi(text):
    norm = normalize_text(text)

    if not norm:
        return False

    if tc_label_mi(text):
        return True

    if soyad_label_mi(text):
        return True

    if ad_label_mi(text):
        return True

    sabitler = [
        "BABA",
        "ANA",
        "DOGUM",
        "TARIHI",
        "YERI",
        "SERI",
        "MEDENI",
        "CINSIYET",
        "KAN",
        "NUFUS",
        "CUZDANI",
        "TURKIYE",
        "CUMHURIYETI",
    ]

    return any(
        x in norm
        for x in sabitler
    )


def en_iyi_label_bul(
    ocr,
    tip
):
    adaylar = []

    for item in ocr:
        text = item.get("text", "")

        if tip == "soyad":
            uygun = soyad_label_mi(text)
            hedef = "SOYADI"
        elif tip == "ad":
            uygun = ad_label_mi(text)
            hedef = "ADI"
        elif tip == "tc":
            uygun = tc_label_mi(text)
            hedef = "TC KIMLIK NO"
        else:
            uygun = False
            hedef = ""

        if not uygun:
            continue

        skor = benzerlik(
            text,
            hedef
        )

        skor += (
            float(
                item.get(
                    "conf",
                    0.0
                )
            )
            *
            0.12
        )

        skor -= (
            item["x1"]
            *
            0.00015
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


def isim_value_adayi_mi(item):
    if item is None:
        return False

    text = str(
        item.get(
            "text",
            ""
        )
    ).strip()

    if not text:
        return False

    if herhangi_bir_label_mi(text):
        return False

    if any(
        c.isdigit()
        for c in text
    ):
        return False

    temiz = isim_temizle(text)

    harfler = re.sub(
        r"[^A-ZÇĞİÖŞÜ]",
        "",
        temiz
    )

    return len(harfler) >= 2


def ayni_satir_mi(
    item1,
    item2,
    minimum_tolerans=20
):
    if (
        item1 is None
        or
        item2 is None
    ):
        return False

    ort_h = (
        yukseklik(item1)
        +
        yukseklik(item2)
    ) / 2.0

    tolerans = max(
        minimum_tolerans,
        ort_h * 0.65
    )

    return (
        abs(
            merkez_y(item1)
            -
            merkez_y(item2)
        )
        <=
        tolerans
    )


def label_satirindaki_deger_bul(
    ocr,
    label
):
    if label is None:
        return None

    label_y = merkez_y(label)
    label_h = yukseklik(label)

    adaylar = []

    for item in ocr:
        if item is label:
            continue

        if not isim_value_adayi_mi(item):
            continue

        item_y = merkez_y(item)

        tolerans_y = max(
            28,
            label_h * 1.30
        )

        if abs(
            item_y - label_y
        ) > tolerans_y:
            continue

        if (
            merkez_x(item)
            <=
            merkez_x(label) + 35
        ):
            continue

        yatay_mesafe = (
            item["x1"]
            -
            label["x2"]
        )

        if yatay_mesafe > 900:
            continue

        dikey_fark = abs(
            item_y - label_y
        )

        skor = 0.0

        skor -= (
            dikey_fark
            *
            7.0
        )

        skor -= (
            max(
                0,
                yatay_mesafe
            )
            *
            0.025
        )

        skor += (
            yukseklik(item)
            *
            0.65
        )

        skor += (
            float(
                item.get(
                    "conf",
                    0.0
                )
            )
            *
            15
        )

        adaylar.append(
            (
                skor,
                item
            )
        )

    if not adaylar:
        return None

    _, anchor = max(
        adaylar,
        key=lambda x: x[0]
    )

    anchor_y = merkez_y(anchor)
    anchor_h = yukseklik(anchor)

    ayni_deger_parcalari = []

    for _, item in adaylar:
        if abs(
            merkez_y(item)
            -
            anchor_y
        ) <= max(
            18,
            anchor_h * 0.55
        ):
            ayni_deger_parcalari.append(item)

    ayni_deger_parcalari.sort(
        key=lambda x: x["x1"]
    )

    parcalar = []

    for item in ayni_deger_parcalari:
        temiz = isim_temizle(
            item["text"]
        )

        if temiz:
            parcalar.append(temiz)

    if not parcalar:
        return None

    deger = " ".join(parcalar)

    conf = sum(
        float(
            item.get(
                "conf",
                0.0
            )
        )
        for item in ayni_deger_parcalari
    ) / len(ayni_deger_parcalari)

    birlesik = {
        "text": deger,
        "conf": conf,
        "x1": min(
            x["x1"]
            for x in ayni_deger_parcalari
        ),
        "y1": min(
            x["y1"]
            for x in ayni_deger_parcalari
        ),
        "x2": max(
            x["x2"]
            for x in ayni_deger_parcalari
        ),
        "y2": max(
            x["y2"]
            for x in ayni_deger_parcalari
        ),
    }

    return {
        "deger": deger,
        "item": birlesik,
        "items": ayni_deger_parcalari
    }


def eski_tc_no_bul(
    ocr
):
    tc_no, tc_item = tc_bul(
        ocr
    )

    if (
        tc_no
        and
        tc_no != "Bulunamadi"
    ):
        return (
            tc_no,
            tc_item
        )

    tc_label = en_iyi_label_bul(
        ocr,
        "tc"
    )

    adaylar = []

    for item in ocr:
        text = str(
            item.get(
                "text",
                ""
            )
        )

        rakam = re.sub(
            r"\D",
            "",
            text
        )

        if len(rakam) != 11:
            continue

        if rakam[0] == "0":
            continue

        skor = float(
            item.get(
                "conf",
                0.0
            )
        )

        if tc_label is not None:
            fark = abs(
                merkez_y(item)
                -
                merkez_y(tc_label)
            )

            if fark < 60:
                skor += 1.5

        adaylar.append(
            (
                skor,
                rakam,
                item
            )
        )

    if not adaylar:
        return (
            "Bulunamadi",
            None
        )

    _, rakam, item = max(
        adaylar,
        key=lambda x: x[0]
    )

    return (
        rakam,
        item
    )


def tc_altindaki_deger_satirlari(
    ocr,
    tc_item
):
    """
    Label okunmasa bile:
      TC'den sonraki ilk gerçek value satırı = soyad
      sonraki farklı value satırı = ad

    Aynı fiziksel satır tek bir kez döndürülür.
    """

    if tc_item is None:
        return []

    tc_y = merkez_y(tc_item)

    adaylar = []

    for item in ocr:
        if not isim_value_adayi_mi(item):
            continue

        if merkez_y(item) <= tc_y:
            continue

        if yukseklik(item) < 16:
            continue

        adaylar.append(item)

    adaylar.sort(
        key=lambda x: (
            merkez_y(x),
            x["x1"]
        )
    )

    satirlar = []

    for item in adaylar:
        uygun_satir = None

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
                20,
                ort_h * 0.60
            )

            if abs(
                merkez_y(item)
                -
                satir_y
            ) <= tolerans:
                uygun_satir = satir
                break

        if uygun_satir is None:
            satirlar.append([item])
        else:
            uygun_satir.append(item)

    sonuclar = []

    for satir in satirlar:
        satir.sort(
            key=lambda x: x["x1"]
        )

        max_h = max(
            yukseklik(x)
            for x in satir
        )

        value_parcalari = [
            x
            for x in satir
            if yukseklik(x) >= max_h * 0.60
        ]

        if not value_parcalari:
            continue

        textler = []

        for item in value_parcalari:
            temiz = isim_temizle(
                item.get(
                    "text",
                    ""
                )
            )

            if temiz:
                textler.append(temiz)

        if not textler:
            continue

        text = " ".join(textler)

        conf = sum(
            float(
                x.get(
                    "conf",
                    0.0
                )
            )
            for x in value_parcalari
        ) / len(value_parcalari)

        birlesik = {
            "text": text,
            "conf": conf,
            "x1": min(
                x["x1"]
                for x in value_parcalari
            ),
            "y1": min(
                x["y1"]
                for x in value_parcalari
            ),
            "x2": max(
                x["x2"]
                for x in value_parcalari
            ),
            "y2": max(
                x["y2"]
                for x in value_parcalari
            ),
        }

        sonuclar.append({
            "deger": text,
            "item": birlesik,
            "items": value_parcalari
        })

    sonuclar.sort(
        key=lambda x: merkez_y(
            x["item"]
        )
    )

    return sonuclar


def kullanilan_satiri_cikar(
    satirlar,
    kullanilan_sonuc
):
    if kullanilan_sonuc is None:
        return list(satirlar)

    kullanilan_item = kullanilan_sonuc.get(
        "item"
    )

    if kullanilan_item is None:
        return list(satirlar)

    kalan = []

    for satir in satirlar:
        if ayni_satir_mi(
            satir.get("item"),
            kullanilan_item
        ):
            continue

        kalan.append(satir)

    return kalan


def eski_tc_bilgilerini_bul(
    ocr_sonuclari
):
    tc_no, tc_item = eski_tc_no_bul(
        ocr_sonuclari
    )

    soyad_label = en_iyi_label_bul(
        ocr_sonuclari,
        "soyad"
    )

    ad_label = en_iyi_label_bul(
        ocr_sonuclari,
        "ad"
    )

    soyad_sonuc = label_satirindaki_deger_bul(
        ocr_sonuclari,
        soyad_label
    )

    ad_sonuc = label_satirindaki_deger_bul(
        ocr_sonuclari,
        ad_label
    )

    # Aynı satır asla iki alana verilemez.
    if (
        soyad_sonuc is not None
        and
        ad_sonuc is not None
        and
        ayni_satir_mi(
            soyad_sonuc["item"],
            ad_sonuc["item"]
        )
    ):
        ad_sonuc = None

    # Soyad mutlaka adın üstünde olmalı.
    if (
        soyad_sonuc is not None
        and
        ad_sonuc is not None
        and
        merkez_y(
            soyad_sonuc["item"]
        )
        >=
        merkez_y(
            ad_sonuc["item"]
        )
    ):
        ad_sonuc = None

    tum_satirlar = tc_altindaki_deger_satirlari(
        ocr_sonuclari,
        tc_item
    )

    kalan_satirlar = kullanilan_satiri_cikar(
        tum_satirlar,
        soyad_sonuc
    )

    kalan_satirlar = kullanilan_satiri_cikar(
        kalan_satirlar,
        ad_sonuc
    )

    if soyad_sonuc is None:
        if tum_satirlar:
            soyad_sonuc = tum_satirlar[0]

        kalan_satirlar = kullanilan_satiri_cikar(
            tum_satirlar,
            soyad_sonuc
        )

        kalan_satirlar = kullanilan_satiri_cikar(
            kalan_satirlar,
            ad_sonuc
        )

    if ad_sonuc is None:
        soy_y = (
            merkez_y(
                soyad_sonuc["item"]
            )
            if soyad_sonuc is not None
            else
            merkez_y(tc_item)
            if tc_item is not None
            else -1
        )

        ad_adaylari = [
            satir
            for satir in kalan_satirlar
            if merkez_y(
                satir["item"]
            ) > soy_y + 12
        ]

        if ad_adaylari:
            ad_sonuc = ad_adaylari[0]

    # Son güvenlik
    if (
        soyad_sonuc is not None
        and
        ad_sonuc is not None
    ):
        # Aynı fiziksel satır asla iki alana verilemez.
        if ayni_satir_mi(
            soyad_sonuc["item"],
            ad_sonuc["item"]
        ):
            ad_sonuc = None

        # Soyad satırı mutlaka ad satırının üstünde olmalı.
        elif (
            merkez_y(
                soyad_sonuc["item"]
            )
            >=
            merkez_y(
                ad_sonuc["item"]
            )
        ):
            ad_sonuc = None

        # OCR iki farklı kutuya aynı metni üretmiş olsa bile
        # aynı kişi alanını iki kez kullanma.
        elif (
            normalize_text(
                soyad_sonuc.get(
                    "deger",
                    ""
                )
            )
            ==
            normalize_text(
                ad_sonuc.get(
                    "deger",
                    ""
                )
            )
        ):
            ad_sonuc = None

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

    if (
        ad != "Bulunamadi"
        and herhangi_bir_label_mi(ad)
    ):
        ad = "Bulunamadi"
        ad_sonuc = None

    if (
        soyad != "Bulunamadi"
        and herhangi_bir_label_mi(soyad)
    ):
        soyad = "Bulunamadi"
        soyad_sonuc = None

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
        tc_no != "Bulunamadi",
        ad != "Bulunamadi",
        soyad != "Bulunamadi"
    ])

    if bulunan == 3:
        if (
            ad_conf >= 0.50
            and soyad_conf >= 0.50
        ):
            guven = "yuksek"
        else:
            guven = "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    return {
        "tc_no": tc_no,
        "ad": ad,
        "soyad": soyad,
        "guven": guven,
        "ad_conf": ad_conf,
        "soyad_conf": soyad_conf,
        "tc_item": tc_item,
        "tc_label": en_iyi_label_bul(
            ocr_sonuclari,
            "tc"
        ),
        "soyad_label": soyad_label,
        "ad_label": ad_label,
        "soyad_item": (
            soyad_sonuc["item"]
            if soyad_sonuc
            else None
        ),
        "ad_item": (
            ad_sonuc["item"]
            if ad_sonuc
            else None
        ),
    }