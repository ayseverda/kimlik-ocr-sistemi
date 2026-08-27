import cv2
import re
import time
import threading
import easyocr
import numpy as np

from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher


# =========================================================
# EASYOCR — LAZY SINGLETON (bu kısım sorunlu değildi, korunuyor)
# =========================================================

_READER = None
_READER_LOCK = threading.Lock()


def reader_getir():
    global _READER
    if _READER is None:
        with _READER_LOCK:
            if _READER is None:
                _READER = easyocr.Reader(["tr", "en"], gpu=False)
    return _READER


# =========================================================
# METİN YARDIMCILARI
# =========================================================

def normalize_text(text):
    text = str(text).upper().strip()
    text = text.translate(str.maketrans({
        "İ": "I", "Ş": "S", "Ğ": "G", "Ü": "U", "Ö": "O", "Ç": "C",
    }))
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def isim_temizle(text):
    if not text:
        return ""
    text = str(text).upper().strip()
    text = re.sub(r"[^A-ZÇĞİIÖŞÜ\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def benzerlik(a, b):
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def buyuk_harf_orani(text):
    harfler = [c for c in str(text) if c.isalpha()]
    if not harfler:
        return 0.0
    return sum(1 for c in harfler if c.isupper()) / len(harfler)


def merkez_y(item):
    return (item["y1"] + item["y2"]) / 2.0


def merkez_x(item):
    return (item["x1"] + item["x2"]) / 2.0


def yukseklik(item):
    return max(1, item["y2"] - item["y1"])


# =========================================================
# TC KİMLİK NUMARASI DOĞRULAMA (verdiğiniz kod, birebir)
# =========================================================

def tc_kimlik_gecerli_mi(no):
    if not no or len(no) != 11 or not no.isdigit() or no[0] == "0":
        return False

    d = [int(x) for x in no]
    d10 = (sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11


def tc_metni_duzelt(text):
    donusum = {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "İ": "1", "L": "1",
        "Z": "2", "S": "5", "G": "6", "B": "8",
    }
    sonuc = ""
    for karakter in str(text).upper():
        if karakter.isdigit():
            sonuc += karakter
        elif karakter in donusum:
            sonuc += donusum[karakter]
    return sonuc


def tc_bul(ocr_sonuclari):
    # 1. Direkt TC
    for item in ocr_sonuclari:
        rakamlar = re.sub(r"\D", "", item["text"])
        if len(rakamlar) == 11 and tc_kimlik_gecerli_mi(rakamlar):
            return rakamlar, item

    # 2. OCR hatalarını düzeltip tekrar dene
    for item in ocr_sonuclari:
        aday = tc_metni_duzelt(item["text"])
        if len(aday) == 11 and tc_kimlik_gecerli_mi(aday):
            return aday, item

    return "Bulunamadi", None


def yabanci_no_gecerli_mi(no):
    return bool(no) and no.isdigit() and len(no) == 11 and no.startswith("99") and tc_kimlik_gecerli_mi(no)


# =========================================================
# OCR
# =========================================================

def easyocr_oku(resim, offset_x=0, offset_y=0):
    baslangic = time.perf_counter()
    if resim is None or resim.size == 0:
        return [], 0.0

    try:
        sonuclar = reader_getir().readtext(resim, detail=1, paragraph=False, decoder="greedy")
    except Exception as e:
        print("EasyOCR hatası:", repr(e))
        return [], time.perf_counter() - baslangic

    bulunanlar = []
    for sonuc in sonuclar:
        try:
            box, text, conf = sonuc
        except Exception:
            continue
        text = str(text).strip()
        if not text:
            continue
        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text, "norm": normalize_text(text), "conf": float(conf),
            "x1": int(box[:, 0].min()) + offset_x, "y1": int(box[:, 1].min()) + offset_y,
            "x2": int(box[:, 0].max()) + offset_x, "y2": int(box[:, 1].max()) + offset_y,
        })
    return bulunanlar, time.perf_counter() - baslangic


# =========================================================
# TÜRKÇE İKİNCİ GEÇİŞ
# =========================================================

def turkce_second_pass_gerekli_mi(text):
    if not text:
        return False
    norm = normalize_text(text)
    return any(harf in norm for harf in ("S", "I", "U", "O", "C", "G"))


def turkce_harf_iyilestir(kart, item):
    if item is None:
        return None, 0.0, 0.0

    ilk_text = isim_temizle(item.get("text", ""))
    ilk_conf = float(item.get("conf", 0.0))

    if not ilk_text or not turkce_second_pass_gerekli_mi(ilk_text):
        return (ilk_text or None), ilk_conf, 0.0

    baslangic = time.perf_counter()
    h, w = kart.shape[:2]
    x1, y1 = max(0, item["x1"] - 3), max(0, item["y1"] - 6)
    x2, y2 = min(w, item["x2"] + 3), min(h, item["y2"] + 4)
    roi = kart[y1:y2, x1:x2]
    if roi.size == 0:
        return ilk_text, ilk_conf, 0.0

    roi = cv2.resize(roi, None, fx=1.65, fy=1.65, interpolation=cv2.INTER_LINEAR)

    try:
        roi_h, roi_w = roi.shape[:2]
        sonuclar = reader_getir().recognize(
            roi, horizontal_list=[[0, roi_w, 0, roi_h]], free_list=[],
            decoder="greedy", beamWidth=5, batch_size=1, workers=0,
            detail=1, paragraph=False, contrast_ths=0.1, adjust_contrast=0.5,
        )
    except Exception:
        return ilk_text, ilk_conf, time.perf_counter() - baslangic

    sure = time.perf_counter() - baslangic
    parcalar = []
    for sonuc in sonuclar or []:
        try:
            _, ikinci_text, ikinci_conf = sonuc
        except Exception:
            continue
        ikinci_text = isim_temizle(ikinci_text)
        if ikinci_text:
            parcalar.append((ikinci_text, float(ikinci_conf)))

    if not parcalar:
        return ilk_text, ilk_conf, sure

    ikinci_text = " ".join(x[0] for x in parcalar)
    ikinci_conf = sum(x[1] for x in parcalar) / len(parcalar)

    if normalize_text(ikinci_text) == normalize_text(ilk_text):
        ilk_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ilk_text))
        ikinci_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ikinci_text))
        if ikinci_tr > ilk_tr or ikinci_conf > ilk_conf + 0.12:
            return ikinci_text, ikinci_conf, sure

    return ilk_text, ilk_conf, sure


# =========================================================
# DEBUG
# =========================================================

def debug_resmi_olustur(kart, alanlar):
    debug = kart.copy()
    for item, etiket, renk in alanlar:
        if item is None:
            continue
        cv2.rectangle(debug, (item["x1"], item["y1"]), (item["x2"], item["y2"]), renk, 4)
        cv2.putText(
            debug, etiket, (item["x1"], max(25, item["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2,
        )
    return debug


# =========================================================
# YENİ T.C. KİMLİK (bu yol zaten sorunsuzdu, dokunulmadı)
# =========================================================

SOYAD_LABEL_HEDEFLERI = ["SOYADI SURNAME", "SURNAME", "SOYADI"]
AD_LABEL_HEDEFLERI = [
    "ADI GIVEN NAME S", "ADI GIVEN NAMES", "ADI GIVEN NAME",
    "GIVEN NAME S", "GIVEN NAMES", "GIVEN NAME",
]


def fuzzy_label_bul(ocr, hedefler, esik=0.40):
    en_iyi_item, en_iyi_skor = None, 0.0
    for item in ocr:
        for hedef in hedefler:
            skor = benzerlik(item["text"], hedef)
            if skor > en_iyi_skor:
                en_iyi_skor, en_iyi_item = skor, item
    return en_iyi_item if en_iyi_skor >= esik else None


def tc_isim_adayi_mi(item):
    text = str(item.get("text", ""))
    if any(c.isdigit() for c in text):
        return False
    temiz = isim_temizle(text)
    if len(re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)) < 2:
        return False
    norm = normalize_text(text)
    yasaklar = [
        "SOYADI", "SURNAME", "GIVEN", "NAME", "KIMLIK", "IDENTITY",
        "BIRTH", "GENDER", "NATIONALITY", "DOCUMENT",
    ]
    if any(x in norm for x in yasaklar):
        return False
    return buyuk_harf_orani(text) >= 0.75


def label_altindaki_degeri_bul(ocr, label, sonraki_label=None):
    if label is None:
        return None
    adaylar = []
    for item in ocr:
        if item is label or not tc_isim_adayi_mi(item):
            continue
        dy = item["y1"] - label["y2"]
        if dy < -12 or dy > 150:
            continue
        if sonraki_label is not None and item["y1"] >= sonraki_label["y1"]:
            continue
        dx = abs(item["x1"] - label["x1"])
        if dx > 240:
            continue
        skor = -dy * 2 - dx * 0.2 + float(item.get("conf", 0)) * 20
        adaylar.append((skor, item))
    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


def tc_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    h, w = kart.shape[:2]
    x1, x2 = int(w * 0.02), int(w * 0.70)
    y1, y2 = int(h * 0.15), int(h * 0.70)
    roi = kart[y1:y2, x1:x2]

    ocr, sure = easyocr_oku(roi, x1, y1)
    tc_no, tc_item = tc_bul(ocr)

    soyad_label = fuzzy_label_bul(ocr, SOYAD_LABEL_HEDEFLERI)
    ad_label = fuzzy_label_bul(ocr, AD_LABEL_HEDEFLERI)
    soyad_item = label_altindaki_degeri_bul(ocr, soyad_label, ad_label)
    ad_item = label_altindaki_degeri_bul(ocr, ad_label)

    ad = isim_temizle(ad_item["text"]) if ad_item else "Bulunamadi"
    soyad = isim_temizle(soyad_item["text"]) if soyad_item else "Bulunamadi"
    kimlik_no_conf = float(tc_item.get("conf", 0)) if tc_item else 0
    ad_conf = float(ad_item.get("conf", 0)) if ad_item else 0
    soyad_conf = float(soyad_item.get("conf", 0)) if soyad_item else 0

    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if min(kimlik_no_conf, ad_conf, soyad_conf) >= 0.50 else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    debug_resmi = None
    if debug:
        debug_resmi = debug_resmi_olustur(kart, [
            (tc_item, "TC", (0, 255, 0)),
            (soyad_item, "SOYAD", (0, 255, 255)),
            (ad_item, "AD", (255, 255, 0)),
        ])

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "tc", "tc_no": tc_no, "ad": ad, "soyad": soyad,
        "guven": guven, "kimlik_no_conf": kimlik_no_conf, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None, "gecerlilik_durumu": None,
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [], "ocr_suresi": sure,
    }


# =========================================================
# ESKİ T.C. KİMLİK — eski (kanıtlanmış) satır/kutu mantığına dönüldü
# =========================================================

def _eski_soyad_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False
    if "SOYAD" in norm:
        return True
    return max(benzerlik(norm, "SOYADI"), benzerlik(norm, "SOYAD")) >= 0.64


def _icerir_etiket_kelimesi(norm, hedef):
    """Ham 'hedef in norm' alt-dize kontrolü, gerçek OCR gürültüsünde
    tehlikeli tesadüfler üretebiliyordu: 'AĞANAK' soyadı OCR'da 'atanak'
    diye okununca, içinde harfiyen 'ANA' geçtiği için (A-T-ANA-K) 'ANA ADI'
    etiketi sanılıp adaylıktan düşürülüyordu. Bu yüzden kısa/genel kelimeler
    (ANA, BABA, DOGUM, KIMLIK, SERI...) artık sadece TAM KELİME olarak
    eşleşiyor. 'SOY' istisna: SOYADI/SOYAD tek bir OCR token'ı olduğu için
    kelimenin BAŞINDA olması yeterli sayılıyor."""
    kelimeler = norm.split()
    if hedef == "SOY":
        return any(k.startswith("SOY") for k in kelimeler)
    return hedef in kelimeler


def _eski_ad_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False
    yasaklar = ["SOY", "BABA", "ANA", "DOGUM", "KIMLIK", "SERI"]
    if any(_icerir_etiket_kelimesi(norm, x) for x in yasaklar):
        return False
    if norm in {"ADI", "AD", "AD1", "A DI", "A0I", "A01", "AOI"}:
        return True
    if len(norm) <= 5:
        return benzerlik(norm, "ADI") >= 0.68
    return False


def _eski_tc_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False
    if "KIMLIK" in norm and ("NO" in norm or "N0" in norm):
        return True
    hedefler = ["TC KIMLIK NO", "T C KIMLIK NO", "TC KIMLIK NUMARASI"]
    return max(benzerlik(norm, hedef) for hedef in hedefler) >= 0.58


def _eski_herhangi_bir_label_mi(text):
    norm = normalize_text(text)
    if not norm:
        return False
    if _eski_tc_label_mi(text) or _eski_soyad_label_mi(text) or _eski_ad_label_mi(text):
        return True
    sabitler = [
        "BABA", "ANA", "DOGUM", "TARIHI", "YERI", "SERI", "MEDENI",
        "CINSIYET", "KAN", "NUFUS", "CUZDANI", "TURKIYE", "CUMHURIYETI",
    ]
    return any(_icerir_etiket_kelimesi(norm, x) for x in sabitler)


def _eski_en_iyi_label_bul(ocr, tip):
    adaylar = []
    for item in ocr:
        text = item.get("text", "")
        if tip == "soyad":
            uygun, hedef = _eski_soyad_label_mi(text), "SOYADI"
        elif tip == "ad":
            uygun, hedef = _eski_ad_label_mi(text), "ADI"
        elif tip == "tc":
            uygun, hedef = _eski_tc_label_mi(text), "TC KIMLIK NO"
        else:
            uygun, hedef = False, ""
        if not uygun:
            continue
        skor = benzerlik(text, hedef) + float(item.get("conf", 0.0)) * 0.12 - item["x1"] * 0.00015
        adaylar.append((skor, item))
    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


def _eski_isim_value_adayi_mi(item):
    text = str(item.get("text", "")).strip()
    if not text:
        return False
    if _eski_herhangi_bir_label_mi(text):
        return False
    if any(c.isdigit() for c in text):
        return False
    temiz = isim_temizle(text)
    harfler = re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)
    # 2 harf çok düşük bir eşikti: TC no'nun hemen altında sık görülen 'ld',
    # 'No', 'So' gibi okunamayan etiket kırıntıları da bu eşiği geçip gerçek
    # satırdan ÖNCE sahte bir "değer satırı" oluşturabiliyordu (gerçek bir
    # görselde tam bu yüzden soyad='LD' gibi anlamsız bir sonuç çıktı).
    # Türkçe ad/soyadların neredeyse tamamı 3+ harf olduğu için bu güvenli.
    return len(harfler) >= 3


def _eski_ayni_satir_mi(item1, item2, minimum_tolerans=20):
    if item1 is None or item2 is None:
        return False
    ort_h = (yukseklik(item1) + yukseklik(item2)) / 2.0
    tolerans = max(minimum_tolerans, ort_h * 0.65)
    return abs(merkez_y(item1) - merkez_y(item2)) <= tolerans


def _parcalari_birlestir(parcalar):
    """Birden çok OCR kutusunu tek bir 'değer' sonucuna birleştirir (metni
    boşlukla birleştirir, kutuyu dış sınırlarına genişletir, confidence'ı
    ortalar). Bu birleştirme eskiden üç ayrı yerde (eski TC'nin hem etiket-
    bazlı hem TC-altı-satır bazlı yolları, göçmenin satır-içi isim bulması)
    birebir aynı şekilde tekrarlanıyordu — davranış aynı, tek yerden çağrılıyor."""
    textler = [isim_temizle(item.get("text", "")) for item in parcalar]
    textler = [t for t in textler if t]
    if not textler:
        return None
    deger = " ".join(textler)
    conf = sum(float(item.get("conf", 0.0)) for item in parcalar) / len(parcalar)
    return {
        "deger": deger,
        "item": {
            "text": deger, "conf": conf,
            "x1": min(x["x1"] for x in parcalar), "y1": min(x["y1"] for x in parcalar),
            "x2": max(x["x2"] for x in parcalar), "y2": max(x["y2"] for x in parcalar),
        },
        "items": parcalar,
    }


def _eski_label_satirindaki_deger_bul(ocr, label):
    if label is None:
        return None
    label_y = merkez_y(label)
    label_h = yukseklik(label)
    adaylar = []
    for item in ocr:
        if item is label or not _eski_isim_value_adayi_mi(item):
            continue
        item_y = merkez_y(item)
        tolerans_y = max(28, label_h * 1.30)
        if abs(item_y - label_y) > tolerans_y:
            continue
        if merkez_x(item) <= merkez_x(label) + 35:
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
    anchor_y, anchor_h = merkez_y(anchor), yukseklik(anchor)
    ayni_deger_parcalari = [
        item for _, item in adaylar
        if abs(merkez_y(item) - anchor_y) <= max(18, anchor_h * 0.55)
    ]
    ayni_deger_parcalari.sort(key=lambda x: x["x1"])
    return _parcalari_birlestir(ayni_deger_parcalari)


def _eski_tc_no_bul(ocr):
    tc_no, tc_item = tc_bul(ocr)
    if tc_no and tc_no != "Bulunamadi":
        return tc_no, tc_item

    tc_label = _eski_en_iyi_label_bul(ocr, "tc")
    adaylar = []
    for item in ocr:
        text = str(item.get("text", ""))
        rakam = re.sub(r"\D", "", text)
        if len(rakam) != 11 or rakam[0] == "0":
            continue
        skor = float(item.get("conf", 0.0))
        if tc_label is not None and abs(merkez_y(item) - merkez_y(tc_label)) < 60:
            skor += 1.5
        adaylar.append((skor, rakam, item))
    if not adaylar:
        return "Bulunamadi", None
    _, rakam, item = max(adaylar, key=lambda x: x[0])
    return rakam, item


def _eski_tc_altindaki_deger_satirlari(ocr, tc_item):
    """Label okunmasa bile: TC'den sonraki ilk gerçek value satırı = soyad,
    sonraki farklı value satırı = ad."""
    if tc_item is None:
        return []
    tc_y = merkez_y(tc_item)
    adaylar = [
        item for item in ocr
        if _eski_isim_value_adayi_mi(item) and merkez_y(item) > tc_y and yukseklik(item) >= 16
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
        # Etiketler değerden daha küçük punto olur; belirgin ufak kalan
        # kutuları (muhtemelen okunamayan bir etiket kırıntısı) satıra katma.
        value_parcalari = [x for x in satir if yukseklik(x) >= max_h * 0.60]
        if not value_parcalari:
            continue
        sonuc = _parcalari_birlestir(value_parcalari)
        if sonuc is not None:
            sonuclar.append(sonuc)

    sonuclar.sort(key=lambda x: merkez_y(x["item"]))
    return sonuclar


def _eski_kullanilan_satiri_cikar(satirlar, kullanilan_sonuc):
    if kullanilan_sonuc is None:
        return list(satirlar)
    kullanilan_item = kullanilan_sonuc.get("item")
    if kullanilan_item is None:
        return list(satirlar)
    return [s for s in satirlar if not _eski_ayni_satir_mi(s.get("item"), kullanilan_item)]


def eski_tc_bilgilerini_bul(ocr):
    tc_no, tc_item = _eski_tc_no_bul(ocr)

    soyad_label = _eski_en_iyi_label_bul(ocr, "soyad")
    ad_label = _eski_en_iyi_label_bul(ocr, "ad")

    soyad_sonuc = _eski_label_satirindaki_deger_bul(ocr, soyad_label)
    ad_sonuc = _eski_label_satirindaki_deger_bul(ocr, ad_label)

    if soyad_sonuc is not None and ad_sonuc is not None and _eski_ayni_satir_mi(soyad_sonuc["item"], ad_sonuc["item"]):
        ad_sonuc = None
    if soyad_sonuc is not None and ad_sonuc is not None and merkez_y(soyad_sonuc["item"]) >= merkez_y(ad_sonuc["item"]):
        ad_sonuc = None

    tum_satirlar = _eski_tc_altindaki_deger_satirlari(ocr, tc_item)
    kalan_satirlar = _eski_kullanilan_satiri_cikar(tum_satirlar, soyad_sonuc)
    kalan_satirlar = _eski_kullanilan_satiri_cikar(kalan_satirlar, ad_sonuc)

    if soyad_sonuc is None:
        if tum_satirlar:
            soyad_sonuc = tum_satirlar[0]
        kalan_satirlar = _eski_kullanilan_satiri_cikar(tum_satirlar, soyad_sonuc)
        kalan_satirlar = _eski_kullanilan_satiri_cikar(kalan_satirlar, ad_sonuc)

    if ad_sonuc is None:
        soy_y = (
            merkez_y(soyad_sonuc["item"]) if soyad_sonuc is not None
            else merkez_y(tc_item) if tc_item is not None
            else -1
        )
        ad_adaylari = [s for s in kalan_satirlar if merkez_y(s["item"]) > soy_y + 12]
        if ad_adaylari:
            ad_sonuc = ad_adaylari[0]

    if soyad_sonuc is not None and ad_sonuc is not None:
        if _eski_ayni_satir_mi(soyad_sonuc["item"], ad_sonuc["item"]):
            ad_sonuc = None
        elif merkez_y(soyad_sonuc["item"]) >= merkez_y(ad_sonuc["item"]):
            ad_sonuc = None
        elif normalize_text(soyad_sonuc.get("deger", "")) == normalize_text(ad_sonuc.get("deger", "")):
            ad_sonuc = None

    # ---- Son çare: hâlâ eksikse, kalan satırlardan (kesin eşleşme
    # kriterlerini geçemeyen ama en azından bir metin bulunan) en yakın
    # adayı kullan. Boş "Bulunamadi" yerine düşük güvenle de olsa bir
    # tahmin göstermek, elle kontrol eden kişiye ipucu verir. Confidence
    # zaten OCR'ın kendi düşük değeri olacağı için mevcut güven/renklendirme
    # sistemi bunu otomatik olarak "düşük güvenli" işaretler.
    if soyad_sonuc is None and kalan_satirlar:
        soyad_sonuc = kalan_satirlar[0]
        kalan_satirlar = _eski_kullanilan_satiri_cikar(kalan_satirlar, soyad_sonuc)
    if ad_sonuc is None and kalan_satirlar:
        ad_sonuc = kalan_satirlar[0]

    ad = ad_sonuc["deger"] if ad_sonuc else "Bulunamadi"
    soyad = soyad_sonuc["deger"] if soyad_sonuc else "Bulunamadi"

    if ad != "Bulunamadi" and _eski_herhangi_bir_label_mi(ad):
        ad, ad_sonuc = "Bulunamadi", None
    if soyad != "Bulunamadi" and _eski_herhangi_bir_label_mi(soyad):
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
        "tc_item": tc_item, "soyad_item": soyad_sonuc["item"] if soyad_sonuc else None,
        "ad_item": ad_sonuc["item"] if ad_sonuc else None,
    }


def eski_tc_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    h, w = kart.shape[:2]
    x1, x2 = int(w * 0.025), int(w * 0.985)
    y1, y2 = int(h * 0.40), int(h * 0.90)
    roi = kart[y1:y2, x1:x2]
    ocr, sure = easyocr_oku(roi, x1, y1)

    sonuc = eski_tc_bilgilerini_bul(ocr)

    ad_item, soyad_item = sonuc.get("ad_item"), sonuc.get("soyad_item")
    ad_ham, soyad_ham = sonuc.get("ad", "Bulunamadi"), sonuc.get("soyad", "Bulunamadi")
    ad_ham_conf = float(sonuc.get("ad_conf", 0.0))
    soyad_ham_conf = float(sonuc.get("soyad_conf", 0.0))

    if ad_item is not None and ad_ham != "Bulunamadi" and ad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(ad_ham):
        ad, ad_conf, ad_sure = turkce_harf_iyilestir(kart, ad_item)
        ad = ad or ad_ham
    else:
        ad, ad_conf, ad_sure = ad_ham, ad_ham_conf, 0.0

    if soyad_item is not None and soyad_ham != "Bulunamadi" and soyad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(soyad_ham):
        soyad, soyad_conf, soyad_sure = turkce_harf_iyilestir(kart, soyad_item)
        soyad = soyad or soyad_ham
    else:
        soyad, soyad_conf, soyad_sure = soyad_ham, soyad_ham_conf, 0.0

    tc_no = sonuc.get("tc_no", "Bulunamadi")
    kimlik_no_conf = float(sonuc.get("tc_item", {}).get("conf", 0.0)) if sonuc.get("tc_item") else 0.0

    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if (ad_conf >= 0.50 and soyad_conf >= 0.50) else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    debug_resmi = None
    if debug:
        debug_resmi = debug_resmi_olustur(kart, [
            (sonuc.get("tc_item"), "TC", (0, 255, 0)),
            (soyad_item, "SOYAD", (0, 255, 255)),
            (ad_item, "AD", (255, 255, 0)),
        ])

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "eski_tc", "tc_no": tc_no, "ad": ad, "soyad": soyad,
        "guven": guven, "kimlik_no_conf": kimlik_no_conf, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None, "gecerlilik_durumu": None,
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [],
        "ocr_suresi": sure + ad_sure + soyad_sure,
    }


# =========================================================
# GÖÇMEN — eski (kanıtlanmış) satır/kutu mantığına dönüldü
# =========================================================

def _gocmen_label_tipini_bul(text):
    norm = normalize_text(text)
    if not norm:
        return None
    if "YABANCI" in norm and "KIMLIK" in norm:
        return "kimlik_no"
    if "SOYAD" in norm:
        return "soyad"
    if "BABA" in norm and "AD" in norm:
        return "baba"
    if "ANNE" in norm and "AD" in norm:
        return "anne"
    if "BELGENIN" in norm and "GECERLILIK" in norm:
        return "gecerlilik"
    if "BASLANGIC" in norm and "BITIS" in norm:
        return "gecerlilik"
    if norm in ("ADI", "ADI ", "AD"):
        return "ad"

    hedefler = {
        "kimlik_no": "YABANCI KIMLIK NO", "soyad": "SOYADI", "ad": "ADI",
        "baba": "BABA ADI", "anne": "ANNE ADI", "gecerlilik": "BELGENIN GECERLILIK TARIHI",
    }
    skorlar = {tip: benzerlik(norm, hedef) for tip, hedef in hedefler.items()}
    tip = max(skorlar, key=skorlar.get)
    skor = skorlar[tip]
    if tip == "ad":
        if any(x in norm for x in ["SOY", "BABA", "ANNE"]):
            return None
        if len(norm) > 6 or skor < 0.65:
            return None
        return "ad"
    return tip if skor >= 0.58 else None


def satirlara_grupla(ocr):
    sirali = sorted(ocr, key=lambda x: (merkez_y(x), x["x1"]))
    satirlar = []
    for item in sirali:
        item_y = merkez_y(item)
        bulundu = False
        for satir in satirlar:
            satir_y = sum(merkez_y(x) for x in satir) / len(satir)
            ort_h = sum(yukseklik(x) for x in satir) / len(satir)
            tolerans = max(22, ort_h * 0.85)
            if abs(item_y - satir_y) <= tolerans:
                satir.append(item)
                bulundu = True
                break
        if not bulundu:
            satirlar.append([item])
    for satir in satirlar:
        satir.sort(key=lambda x: x["x1"])
    return satirlar


def _gocmen_satir_label_bul(satir, tip):
    adaylar = []
    for item in satir:
        if _gocmen_label_tipini_bul(item.get("text", "")) != tip:
            continue
        skor = float(item.get("conf", 0.0)) - item["x1"] * 0.0005
        adaylar.append((skor, item))
    if not adaylar:
        return None
    return max(adaylar, key=lambda x: x[0])[1]


def _gocmen_isim_value_mi(item):
    text = str(item.get("text", "")).strip()
    if not text:
        return False
    if _gocmen_label_tipini_bul(text) is not None:
        return False
    if any(c.isdigit() for c in text):
        return False
    temiz = isim_temizle(text)
    harfler = re.sub(r"[^A-ZÇĞİÖŞÜ]", "", temiz)
    return len(harfler) >= 2


def _gocmen_satir_isim_bul(satir, label_item):
    if label_item is None:
        return None
    adaylar = [
        item for item in satir
        if item is not label_item and _gocmen_isim_value_mi(item) and merkez_x(item) > merkez_x(label_item) + 25
    ]
    if not adaylar:
        return None
    adaylar.sort(key=lambda x: x["x1"])
    return _parcalari_birlestir(adaylar)


def _gocmen_no_bul(ocr):
    for item in ocr:
        rakamlar = re.sub(r"\D", "", str(item.get("text", "")))
        if yabanci_no_gecerli_mi(rakamlar):
            return {"deger": rakamlar, "item": item}
    for item in ocr:
        # Göçmen kimlik no'su da TC ile aynı 11 haneli checksum algoritmasını
        # kullandığı için (99 önekiyle) harf/rakam düzeltmesi de tc_metni_duzelt
        # ile birebir aynıdır — ayrı bir kopya fonksiyona gerek yok.
        aday = tc_metni_duzelt(item.get("text", ""))
        if yabanci_no_gecerli_mi(aday):
            return {"deger": aday, "item": item}
    return None


_TARIH_DESENI = re.compile(r"(?<!\d)(\d{1,2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{4})(?!\d)")


def _gocmen_tarihleri_bul(text):
    if not text:
        return []
    tarihler = []
    for gun, ay, yil in _TARIH_DESENI.findall(str(text)):
        try:
            tarihler.append(datetime(int(yil), int(ay), int(gun)).date())
        except ValueError:
            pass
    return tarihler


def _gocmen_tarih_metne_cevir(tarih):
    return tarih.strftime("%d.%m.%Y") if tarih is not None else "Bulunamadi"


def _gocmen_gecerlilik_tarihi_bul(satirlar):
    adaylar = []
    for satir in satirlar:
        tarihler = []
        for item in satir:
            tarihler.extend(_gocmen_tarihleri_bul(item.get("text", "")))
        if len(tarihler) < 2:
            continue

        satir_y = sum(merkez_y(x) for x in satir) / len(satir)
        label_skor = 0.0
        for item in satir:
            if _gocmen_label_tipini_bul(item.get("text", "")) == "gecerlilik":
                label_skor = max(label_skor, 200)
        if label_skor == 0:
            for diger in satirlar:
                diger_y = sum(merkez_y(x) for x in diger) / len(diger)
                if abs(diger_y - satir_y) > 90:
                    continue
                for item in diger:
                    if _gocmen_label_tipini_bul(item.get("text", "")) == "gecerlilik":
                        label_skor = 150
                        break

        skor = 300 + label_skor + satir_y * 0.03
        adaylar.append((skor, satir, tarihler))

    if not adaylar:
        return {"baslangic": None, "bitis": None, "gecerli": None, "durum": "kontrol_edilemedi", "items": []}

    _, satir, tarihler = max(adaylar, key=lambda x: x[0])
    baslangic, bitis = tarihler[0], tarihler[1]
    bugun = datetime.now(timezone(timedelta(hours=3))).date()

    if bugun < baslangic:
        gecerli, durum = False, "henuz_baslamadi"
    elif bugun > bitis:
        gecerli, durum = False, "suresi_gecmis"
    else:
        gecerli, durum = True, "gecerli"

    tarihli_itemlar = [(item, _gocmen_tarihleri_bul(item.get("text", ""))) for item in satir]
    tarihli_itemlar = [(it, t) for it, t in tarihli_itemlar if t]
    if len(tarihli_itemlar) >= 2:
        debug_items = [tarihli_itemlar[0][0], tarihli_itemlar[1][0]]
    elif len(tarihli_itemlar) == 1 and len(tarihli_itemlar[0][1]) >= 2:
        it = tarihli_itemlar[0][0]
        orta = int((it["x1"] + it["x2"]) / 2)
        debug_items = [{**it, "x2": orta}, {**it, "x1": orta}]
    else:
        debug_items = []

    return {"baslangic": baslangic, "bitis": bitis, "gecerli": gecerli, "durum": durum, "items": debug_items}


def gocmen_bilgilerini_bul(ocr):
    satirlar = satirlara_grupla(ocr)
    no_sonuc = _gocmen_no_bul(ocr)

    ad_sonuc, ad_label = None, None
    for satir in satirlar:
        label = _gocmen_satir_label_bul(satir, "ad")
        if label is None:
            continue
        sonuc = _gocmen_satir_isim_bul(satir, label)
        if sonuc is not None:
            ad_label, ad_sonuc = label, sonuc
            break

    soyad_sonuc, soyad_label = None, None
    for satir in satirlar:
        label = _gocmen_satir_label_bul(satir, "soyad")
        if label is None:
            continue
        sonuc = _gocmen_satir_isim_bul(satir, label)
        if sonuc is not None:
            soyad_label, soyad_sonuc = label, sonuc
            break

    gecerlilik = _gocmen_gecerlilik_tarihi_bul(satirlar)

    # ---- Son çare: etiket (ADI/SOYADI) hiç bulunamadıysa, kimlik no'nun
    # altındaki ilk isim-adayı satırlarını sırayla kullan. Eski TC'deki
    # "TC altındaki satırlar" yedeğinin göçmen karşılığı — kesin eşleşme
    # değil ama boş "Bulunamadi" yerine bir tahmin verir.
    if (ad_sonuc is None or soyad_sonuc is None) and no_sonuc is not None:
        no_y = merkez_y(no_sonuc["item"])
        satir_adaylari = []
        for satir in satirlar:
            varlar = [it for it in satir if _gocmen_isim_value_mi(it) and merkez_y(it) > no_y]
            if varlar:
                birlesik = _parcalari_birlestir(sorted(varlar, key=lambda x: x["x1"]))
                if birlesik is not None:
                    satir_adaylari.append(birlesik)
        satir_adaylari.sort(key=lambda a: merkez_y(a["item"]))

        kalan = [
            a for a in satir_adaylari
            if a.get("item") is not (ad_sonuc or {}).get("item")
            and a.get("item") is not (soyad_sonuc or {}).get("item")
        ]
        # Göçmen tablosunda fiziksel sıra: Yabancı Kimlik No -> Adı -> Soyadı.
        if ad_sonuc is None and kalan:
            ad_sonuc = kalan.pop(0)
        if soyad_sonuc is None and kalan:
            soyad_sonuc = kalan.pop(0)

    kimlik_no = no_sonuc["deger"] if no_sonuc else "Bulunamadi"
    ad = ad_sonuc["deger"] if ad_sonuc else "Bulunamadi"
    soyad = soyad_sonuc["deger"] if soyad_sonuc else "Bulunamadi"

    kimlik_no_conf = float(no_sonuc["item"].get("conf", 0.0)) if no_sonuc else 0.0
    ad_conf = float(ad_sonuc["item"].get("conf", 0.0)) if ad_sonuc else 0.0
    soyad_conf = float(soyad_sonuc["item"].get("conf", 0.0)) if soyad_sonuc else 0.0

    bulunan = sum([kimlik_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if min(kimlik_no_conf, ad_conf, soyad_conf) >= 0.50 else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    return {
        "kimlik_no": kimlik_no, "ad": ad, "soyad": soyad, "guven": guven,
        "kimlik_no_conf": kimlik_no_conf, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": _gocmen_tarih_metne_cevir(gecerlilik["baslangic"]),
        "bitis_tarihi": _gocmen_tarih_metne_cevir(gecerlilik["bitis"]),
        "belge_gecerli": gecerlilik["gecerli"], "gecerlilik_durumu": gecerlilik["durum"],
        "kimlik_no_item": no_sonuc["item"] if no_sonuc else None,
        "ad_item": ad_sonuc["item"] if ad_sonuc else None,
        "soyad_item": soyad_sonuc["item"] if soyad_sonuc else None,
        "gecerlilik_items": gecerlilik["items"],
    }


def gocmen_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    ocr, sure = easyocr_oku(kart)
    sonuc = gocmen_bilgilerini_bul(ocr)

    debug_resmi = None
    if debug:
        alanlar = [
            (sonuc.get("kimlik_no_item"), "YKN", (0, 255, 0)),
            (sonuc.get("soyad_item"), "SOYAD", (0, 255, 255)),
            (sonuc.get("ad_item"), "AD", (255, 255, 0)),
        ]
        tarih_items = sonuc.get("gecerlilik_items", [])
        if len(tarih_items) >= 1:
            alanlar.append((tarih_items[0], "BASLANGIC", (255, 0, 255)))
        if len(tarih_items) >= 2:
            alanlar.append((tarih_items[1], "BITIS", (180, 0, 255)))
        debug_resmi = debug_resmi_olustur(kart, alanlar)

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "gocmen", "tc_no": sonuc["kimlik_no"],
        "ad": sonuc["ad"], "soyad": sonuc["soyad"], "guven": sonuc["guven"],
        "kimlik_no_conf": sonuc["kimlik_no_conf"], "ad_conf": sonuc["ad_conf"], "soyad_conf": sonuc["soyad_conf"],
        "baslangic_tarihi": sonuc["baslangic_tarihi"], "bitis_tarihi": sonuc["bitis_tarihi"],
        "belge_gecerli": sonuc["belge_gecerli"], "gecerlilik_durumu": sonuc["gecerlilik_durumu"],
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [], "ocr_suresi": sure,
    }


# =========================================================
# ANA
# =========================================================

def bilgileri_cimbizla(orijinal_kart, sayfa_no=None, debug=False, belge_tipi="tc"):
    if orijinal_kart is None:
        return {
            "sayfa_no": sayfa_no, "belge_tipi": belge_tipi, "tc_no": "Bulunamadi",
            "ad": "Bulunamadi", "soyad": "Bulunamadi", "guven": "dusuk",
            "kimlik_no_conf": 0.0, "ad_conf": 0.0, "soyad_conf": 0.0,
            "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None,
            "gecerlilik_durumu": None, "debug_resmi": None, "tum_ocr": [], "ocr_suresi": 0.0,
        }
    if belge_tipi == "gocmen":
        return gocmen_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    if belge_tipi == "eski_tc":
        return eski_tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    return tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)