import re
from difflib import SequenceMatcher

from kimlikler.kimlikler_tc import tc_bul


def normalize_text(text):
    text = str(text or "").upper().strip()
    text = text.translate(str.maketrans({
        "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
    }))
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def isim_temizle(text):
    text = str(text or "").upper().strip()
    text = re.sub(r"[^A-ZÇĞİIÖŞÜ\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def benzerlik(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def kelimeler(text):
    """Metni, etiket kontrolünde kullanılacak bağımsız token'lara ayırır."""
    norm = normalize_text(text)
    return norm.split() if norm else []


def _kisa_label_adayi_mi(norm, max_kelime=2, max_uzunluk=18):
    """Fuzzy eşleşmenin kişi adları üzerinde çalışmasını engeller."""
    tokens = norm.split()
    return bool(tokens) and len(tokens) <= max_kelime and len(norm) <= max_uzunluk


def merkez_y(item):
    return (item["y1"] + item["y2"]) / 2.0


def merkez_x(item):
    return (item["x1"] + item["x2"]) / 2.0


def yukseklik(item):
    return max(1, item["y2"] - item["y1"])


def _deger_sutunu_sol_siniri(tc_item):
    """Eski nufus cuzdaninin sagdaki deger sutunu icin guvenli sol sinir.

    Perspektif duzeltmeden sonra TC numarasi, soyad ve ad ayni deger sutununda
    hizalanir. Sol sutundaki etiket OCR tarafindan ``SOYADI -> VIU`` gibi
    anlamsiz bir metne donusse bile TC numarasinin belirledigi bu sinirin
    solunda kalir. Boylece bozuk etiket kisi adi olarak kullanilmaz.
    """
    if tc_item is None:
        return None

    tc_genislik = max(1.0, float(tc_item["x2"] - tc_item["x1"]))
    # Isimler TC numarasindan biraz daha soldan baslayabilir. Ancak 100 px'den
    # fazla gevseme, 750 px'lik standart kartta etiket sutununa tasar.
    tolerans = max(35.0, min(100.0, tc_genislik * 0.45))
    return max(0.0, float(tc_item["x1"]) - tolerans)


def _deger_sutununda_mi(item, sol_sinir):
    if item is None or sol_sinir is None:
        return True

    # x1 kullanmak, etiketi ve degeri tek genis kutuda birlestiren bozuk OCR
    # sonucunu da konservatif olarak reddeder. Yanlis isim yerine Bulunamadi daha
    # guvenlidir; ikinci OCR gecisi gercek degeri yeniden bulabilir.
    return float(item["x1"]) >= sol_sinir


# =========================================================
# LABEL TESPİTİ
# =========================================================

def tc_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False

    tokens = norm.split()
    if "KIMLIK" in tokens and ({"NO", "N0", "NUMARASI"} & set(tokens)):
        return True

    hedefler = ["TC KIMLIK NO", "T C KIMLIK NO", "TC KIMLIK NUMARASI"]
    return (
        _kisa_label_adayi_mi(norm, max_kelime=4, max_uzunluk=24)
        and max(benzerlik(norm, hedef) for hedef in hedefler) >= 0.58
    )


def soyad_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False

    tokens = norm.split()
    if norm == "SOY ADI" or (tokens and tokens[0] in {"SOYAD", "SOYADI"}):
        return True

    return (
        _kisa_label_adayi_mi(norm)
        and max(benzerlik(norm, "SOYADI"), benzerlik(norm, "SOYAD")) >= 0.72
    )


def ad_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False

    tokens = set(kelimeler(norm))
    yasaklar = {
        "SOY", "SOYAD", "SOYADI", "BABA", "ANA", "DOGUM", "KIMLIK",
        "SERI", "MEDENI", "CINSIYET", "KAN", "NUFUS",
    }
    if tokens & yasaklar:
        return False

    ad_varyantlari = {"ADI", "AD", "AD1", "A0I", "A01", "AOI"}
    if norm == "A DI" or (norm.split() and norm.split()[0] in ad_varyantlari):
        return True

    if len(tokens) == 1 and len(norm) <= 3:
        return benzerlik(norm, "ADI") >= 0.68

    return False


def diger_alan_label_mi(text):
    """Ad/soyad dışındaki eski kimlik etiketlerini token/ifade bazında tanır."""
    norm = normalize_text(text)
    if not norm:
        return False

    tokens = norm.split()
    tek_kelime_etiketleri = {
        "BABA", "ANA", "DOGUM", "TARIHI", "YERI", "SERI", "MEDENI",
        "CINSIYET", "KAN", "NUFUS", "CUZDANI", "TURKIYE", "CUMHURIYETI",
    }
    if tokens[0] in tek_kelime_etiketleri:
        return True

    # Etiket ve değerin aynı OCR kutusunda birleştiği durumlarda yalnızca bilinen
    # etiket ifadelerinin başta olmasına izin verilir. Böylece "HAKAN", "CANAN"
    # veya "SERIF" içindeki harf dizileri etiket sayılmaz.
    etiket_baslangiclari = (
        ("BABA", "ADI"), ("ANA", "ADI"),
        ("DOGUM", "TARIHI"), ("DOGUM", "YERI"),
        ("SERI", "NO"), ("SERI", "NUMARASI"),
        ("MEDENI", "HALI"), ("KAN", "GRUBU"),
        ("NUFUS", "CUZDANI"), ("TURKIYE", "CUMHURIYETI"),
    )
    return any(tuple(tokens[:len(baslangic)]) == baslangic for baslangic in etiket_baslangiclari)


def herhangi_bir_label_mi(text):
    return (
        tc_label_mi(text)
        or soyad_label_mi(text)
        or ad_label_mi(text)
        or diger_alan_label_mi(text)
    )


def en_iyi_label_bul(ocr, tip):
    adaylar = []

    for item in ocr:
        text = item.get("text", "")

        if tip == "soyad":
            uygun, hedef = soyad_label_mi(text), "SOYADI"
        elif tip == "ad":
            uygun, hedef = ad_label_mi(text), "ADI"
        elif tip == "tc":
            uygun, hedef = tc_label_mi(text), "TC KIMLIK NO"
        else:
            uygun, hedef = False, ""

        if not uygun:
            continue

        skor = benzerlik(text, hedef)
        skor += float(item.get("conf", 0.0)) * 0.12
        skor -= item["x1"] * 0.00015
        adaylar.append((skor, item))

    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


# =========================================================
# DEĞER ADAYLARI
# =========================================================

def isim_value_adayi_mi(item):
    if item is None:
        return False

    text = str(item.get("text", "")).strip()
    if not text:
        return False
    if herhangi_bir_label_mi(text):
        return False
    if any(c.isdigit() for c in text):
        return False

    temiz = isim_temizle(text)
    harfler = re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)
    return len(harfler) >= 2


def ayni_satir_mi(item1, item2, minimum_tolerans=12):
    if item1 is None or item2 is None:
        return False

    ort_h = (yukseklik(item1) + yukseklik(item2)) / 2.0
    # Sabit 20px tolerans, küçük OCR kutularında komşu iki fiziksel satırı tek
    # satır sayabiliyordu. Ölçeğe bağlı ama üst sınırı olan daha dar tolerans.
    tolerans = max(minimum_tolerans, min(24, ort_h * 0.55))
    return abs(merkez_y(item1) - merkez_y(item2)) <= tolerans


def label_satirindaki_deger_bul(ocr, label, deger_sutunu_sol_siniri=None):
    if label is None:
        return None

    label_y = merkez_y(label)
    label_h = yukseklik(label)
    adaylar = []

    for item in ocr:
        if item is label or not isim_value_adayi_mi(item):
            continue
        if not _deger_sutununda_mi(item, deger_sutunu_sol_siniri):
            continue

        item_y = merkez_y(item)
        item_h = yukseklik(item)
        tolerans_y = max(10, min(28, max(label_h, item_h) * 0.70))
        if abs(item_y - label_y) > tolerans_y:
            continue
        yatay_bindirme_payi = max(8, min(16, label_h * 0.35))
        if item["x1"] < label["x2"] - yatay_bindirme_payi:
            continue

        yatay_mesafe = item["x1"] - label["x2"]
        if yatay_mesafe > 900:
            continue

        dikey_fark = abs(item_y - label_y)
        skor = (
            -dikey_fark * 7.0
            - max(0, yatay_mesafe) * 0.025
            + yukseklik(item) * 0.65
            + float(item.get("conf", 0.0)) * 15
        )
        adaylar.append((skor, item))

    if not adaylar:
        return None

    _, anchor = max(adaylar, key=lambda x: x[0])

    ayni_deger_parcalari = [item for _, item in adaylar if ayni_satir_mi(item, anchor)]
    ayni_deger_parcalari.sort(key=lambda x: x["x1"])

    parcalar = [isim_temizle(item["text"]) for item in ayni_deger_parcalari]
    parcalar = [p for p in parcalar if p]
    if not parcalar:
        return None

    deger = " ".join(parcalar)
    conf = sum(float(item.get("conf", 0.0)) for item in ayni_deger_parcalari) / len(ayni_deger_parcalari)

    birlesik = {
        "text": deger, "conf": conf,
        "x1": min(x["x1"] for x in ayni_deger_parcalari),
        "y1": min(x["y1"] for x in ayni_deger_parcalari),
        "x2": max(x["x2"] for x in ayni_deger_parcalari),
        "y2": max(x["y2"] for x in ayni_deger_parcalari),
    }
    return {"deger": deger, "item": birlesik, "items": ayni_deger_parcalari}


# =========================================================
# TC NO
# =========================================================

def eski_tc_no_bul(ocr):
    # Ortak tc_bul hem doğrudan rakamları hem OCR harf/rakam düzeltmesini resmi
    # checksum ile doğrular. Checksum başarısızsa 11 hane olması yeterli değildir.
    return tc_bul(ocr)


# =========================================================
# TC ALTINDAKİ DEĞER SATIRLARI (label okunamazsa fallback için)
# =========================================================

def tc_altindaki_deger_satirlari(ocr, tc_item, deger_sutunu_sol_siniri=None):
    """Label okunmasa bile: TC'den sonraki ilk gerçek value satırı = soyad,
    sonraki farklı value satırı = ad. Aynı fiziksel satır tek kez döndürülür."""
    if tc_item is None:
        return []

    tc_y = merkez_y(tc_item)

    # Ad/soyad bloğundan sonraki ilk alan etiketi bir sınırdır. Bu sınır olmazsa
    # eksik bir ad/soyad, baba/ana adı gibi daha aşağıdaki bir değerle dolabiliyor.
    sonraki_alan_etiketleri = [
        merkez_y(item) for item in ocr
        if merkez_y(item) > tc_y and diger_alan_label_mi(item.get("text", ""))
    ]
    ad_soyad_blok_siniri = min(sonraki_alan_etiketleri) if sonraki_alan_etiketleri else None

    adaylar = [
        item for item in ocr
        if (
            isim_value_adayi_mi(item)
            and _deger_sutununda_mi(item, deger_sutunu_sol_siniri)
            and merkez_y(item) > tc_y
            and yukseklik(item) >= 16
            and (ad_soyad_blok_siniri is None or merkez_y(item) < ad_soyad_blok_siniri)
        )
    ]
    adaylar.sort(key=lambda x: (merkez_y(x), x["x1"]))

    satirlar = []
    for item in adaylar:
        uygun_satir = None
        for satir in satirlar:
            satir_y = sum(merkez_y(x) for x in satir) / len(satir)
            ort_h = sum(yukseklik(x) for x in satir) / len(satir)
            tolerans = max(20, ort_h * 0.60)
            if abs(merkez_y(item) - satir_y) <= tolerans:
                uygun_satir = satir
                break
        if uygun_satir is None:
            satirlar.append([item])
        else:
            uygun_satir.append(item)

    sonuclar = []
    for satir in satirlar:
        satir.sort(key=lambda x: x["x1"])
        max_h = max(yukseklik(x) for x in satir)
        value_parcalari = [x for x in satir if yukseklik(x) >= max_h * 0.60]
        if not value_parcalari:
            continue

        textler = [isim_temizle(item.get("text", "")) for item in value_parcalari]
        textler = [t for t in textler if t]
        if not textler:
            continue

        text = " ".join(textler)
        conf = sum(float(x.get("conf", 0.0)) for x in value_parcalari) / len(value_parcalari)
        birlesik = {
            "text": text, "conf": conf,
            "x1": min(x["x1"] for x in value_parcalari),
            "y1": min(x["y1"] for x in value_parcalari),
            "x2": max(x["x2"] for x in value_parcalari),
            "y2": max(x["y2"] for x in value_parcalari),
        }
        sonuclar.append({"deger": text, "item": birlesik, "items": value_parcalari})

    sonuclar.sort(key=lambda x: merkez_y(x["item"]))
    return sonuclar


def kullanilan_satiri_cikar(satirlar, kullanilan_sonuc):
    if kullanilan_sonuc is None:
        return list(satirlar)

    kullanilan_item = kullanilan_sonuc.get("item")
    if kullanilan_item is None:
        return list(satirlar)

    return [satir for satir in satirlar if not ayni_satir_mi(satir.get("item"), kullanilan_item)]


def _label_deger_dikey_farki(label, sonuc):
    if label is None or sonuc is None or sonuc.get("item") is None:
        return float("inf")
    return abs(merkez_y(label) - merkez_y(sonuc["item"]))


# =========================================================
# ANA
# =========================================================

def eski_tc_bilgilerini_bul(ocr_sonuclari):
    tc_no, tc_item = eski_tc_no_bul(ocr_sonuclari)
    deger_sutunu_sol_siniri = _deger_sutunu_sol_siniri(tc_item)

    soyad_label = en_iyi_label_bul(ocr_sonuclari, "soyad")
    ad_label = en_iyi_label_bul(ocr_sonuclari, "ad")

    soyad_sonuc = label_satirindaki_deger_bul(
        ocr_sonuclari, soyad_label, deger_sutunu_sol_siniri
    )
    ad_sonuc = label_satirindaki_deger_bul(
        ocr_sonuclari, ad_label, deger_sutunu_sol_siniri
    )

    # Aynı satır asla iki alana verilemez. Komşu etiketlerden ikisi de aynı
    # değeri yakaladıysa değere dikey olarak daha yakın etiketi koru.
    if soyad_sonuc is not None and ad_sonuc is not None and ayni_satir_mi(soyad_sonuc["item"], ad_sonuc["item"]):
        soyad_fark = _label_deger_dikey_farki(soyad_label, soyad_sonuc)
        ad_fark = _label_deger_dikey_farki(ad_label, ad_sonuc)
        if ad_fark < soyad_fark:
            soyad_sonuc = None
        else:
            ad_sonuc = None

    # Soyad mutlaka adın üstünde olmalı. Ters sıralama alan karışmasına işaret
    # eder; aşağıdaki fiziksel satır fallback'i iki alanı yeniden kurar.
    if soyad_sonuc is not None and ad_sonuc is not None and merkez_y(soyad_sonuc["item"]) >= merkez_y(ad_sonuc["item"]):
        soyad_sonuc = None
        ad_sonuc = None

    tum_satirlar = tc_altindaki_deger_satirlari(
        ocr_sonuclari, tc_item, deger_sutunu_sol_siniri
    )
    kalan_satirlar = kullanilan_satiri_cikar(tum_satirlar, soyad_sonuc)
    kalan_satirlar = kullanilan_satiri_cikar(kalan_satirlar, ad_sonuc)

    if soyad_sonuc is None:
        # Doğru bulunmuş ad satırını hiçbir zaman soyad fallback'i olarak kullanma.
        soyad_adaylari = kullanilan_satiri_cikar(tum_satirlar, ad_sonuc)
        if ad_sonuc is not None:
            ad_y = merkez_y(ad_sonuc["item"])
            soyad_adaylari = [
                satir for satir in soyad_adaylari
                if merkez_y(satir["item"]) < ad_y - 12
            ]
            if soyad_adaylari:
                # Ada en yakın üst satır, eski kimlik düzeninde soyad satırıdır.
                soyad_sonuc = soyad_adaylari[-1]
        elif soyad_adaylari:
            soyad_sonuc = soyad_adaylari[0]
        kalan_satirlar = kullanilan_satiri_cikar(tum_satirlar, soyad_sonuc)
        kalan_satirlar = kullanilan_satiri_cikar(kalan_satirlar, ad_sonuc)

    if ad_sonuc is None:
        soy_y = (
            merkez_y(soyad_sonuc["item"]) if soyad_sonuc is not None
            else merkez_y(tc_item) if tc_item is not None
            else -1
        )
        ad_adaylari = [satir for satir in kalan_satirlar if merkez_y(satir["item"]) > soy_y + 12]
        if ad_adaylari:
            ad_sonuc = ad_adaylari[0]

    # Son güvenlik
    if soyad_sonuc is not None and ad_sonuc is not None:
        if ayni_satir_mi(soyad_sonuc["item"], ad_sonuc["item"]):
            ad_sonuc = None
        elif merkez_y(soyad_sonuc["item"]) >= merkez_y(ad_sonuc["item"]):
            ad_sonuc = None
        elif normalize_text(soyad_sonuc.get("deger", "")) == normalize_text(ad_sonuc.get("deger", "")):
            ad_sonuc = None

    ad = ad_sonuc["deger"] if ad_sonuc else "Bulunamadi"
    soyad = soyad_sonuc["deger"] if soyad_sonuc else "Bulunamadi"

    if ad != "Bulunamadi" and herhangi_bir_label_mi(ad):
        ad, ad_sonuc = "Bulunamadi", None
    if soyad != "Bulunamadi" and herhangi_bir_label_mi(soyad):
        soyad, soyad_sonuc = "Bulunamadi", None

    ad_conf = float(ad_sonuc["item"].get("conf", 0.0)) if ad_sonuc else 0.0
    soyad_conf = float(soyad_sonuc["item"].get("conf", 0.0)) if soyad_sonuc else 0.0

    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if (ad_conf >= 0.50 and soyad_conf >= 0.50) else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    return {
        "tc_no": tc_no, "ad": ad, "soyad": soyad, "guven": guven,
        "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "tc_item": tc_item, "tc_label": en_iyi_label_bul(ocr_sonuclari, "tc"),
        "soyad_label": soyad_label, "ad_label": ad_label,
        "soyad_item": soyad_sonuc["item"] if soyad_sonuc else None,
        "ad_item": ad_sonuc["item"] if ad_sonuc else None,
    }
