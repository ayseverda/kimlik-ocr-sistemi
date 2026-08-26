import cv2
import re
import time
import threading
import unicodedata
import easyocr
import numpy as np

from difflib import SequenceMatcher

from kimlikler.kimlikler_tc import tc_bul
from kimlikler.kimlikler_eski_tc import eski_tc_bilgilerini_bul
from kimlikler.kimlikler_gocmen import (
    gocmen_bilgilerini_bul,
    satir_label_bul,
    satirlara_grupla,
    yabanci_no_gecerli_mi,
)


# =========================================================
# EASYOCR — LAZY SINGLETON
# =========================================================
#
# ÖNEMLİ: 'reader = easyocr.Reader(...)' MODÜL SEVİYESİNDE çalıştırılırsa,
# bu modül import edilir edilmez (yani web.py açılır açılmaz, kullanıcı daha
# hiçbir dosya yüklemeden) model indirmeye başlar. İlk deploy'da bu indirme
# birkaç dakika sürebiliyor; Streamlit Cloud script tamamen çalışıp bir sayfa
# döndürene kadar health-check bekliyor, bu da health-check'i zaman aşımına
# uğratıp "Oh no. Error running app." hatasına yol açıyor (gerçekte yaşandı).
# Bu yüzden reader SADECE ilk gerçek OCR çağrısında (kullanıcı bir dosya
# işlettiğinde) oluşturuluyor; app hemen açılıyor, indirme yalnızca o an ve
# yalnızca bir kez gerçekleşiyor (cache'lenir).

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
# NORMALIZE
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


# =========================================================
# OCR
# =========================================================

def easyocr_oku(resim, offset_x=0, offset_y=0):
    baslangic = time.perf_counter()
    bulunanlar = []

    try:
        sonuclar = reader_getir().readtext(resim, detail=1, paragraph=False, decoder="greedy")
    except Exception as e:
        print("EasyOCR hatası:", repr(e))
        return [], time.perf_counter() - baslangic

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
# ESKİ TC — ÖZEL OCR
# =========================================================

def _roi_ocr_oku(roi, offset_x, offset_y, **kwargs):
    """easyocr_oku_eski_tc / _fallback ortak gövdesi: roi üzerinde OCR
    çalıştırır, kutuları orijinal kart koordinatına taşır."""
    try:
        sonuclar = reader_getir().readtext(roi, detail=1, paragraph=False, decoder="greedy", **kwargs)
    except Exception as e:
        print("Eski TC EasyOCR hatası:", repr(e))
        return []

    bulunanlar = []
    for sonuc in sonuclar:
        try:
            box, text_ocr, conf = sonuc
        except Exception:
            continue

        text_ocr = str(text_ocr).strip()
        if not text_ocr:
            continue

        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text_ocr, "norm": normalize_text(text_ocr), "conf": float(conf),
            "x1": int(box[:, 0].min()) + offset_x, "y1": int(box[:, 1].min()) + offset_y,
            "x2": int(box[:, 0].max()) + offset_x, "y2": int(box[:, 1].max()) + offset_y,
        })
    return bulunanlar


def easyocr_oku_eski_tc(resim):
    """HIZLI ESKİ TC OCR: ilk aşamada SADECE TC + SOYADI + ADI bölgesini okur.
    İkinci/geniş OCR yalnızca parser gerçekten bir alanı bulamazsa
    eski_tc_bilgilerini_oku() içinde çağrılır."""
    baslangic = time.perf_counter()
    if resim is None:
        return [], 0.0

    h, w = resim.shape[:2]
    x1, x2 = int(w * 0.025), int(w * 0.985)
    # Perspektif sonrası eski nüfus cüzdanında TC + SOYADI + ADI alanlarını kapsayan bölüm.
    y1, y2 = int(h * 0.47), int(h * 0.82)

    roi = resim[y1:y2, x1:x2]
    if roi.size == 0:
        return [], 0.0

    bulunanlar = _roi_ocr_oku(
        roi, x1, y1,
        text_threshold=0.48, low_text=0.28, link_threshold=0.32,
        min_size=8, width_ths=0.75, ycenter_ths=0.5, add_margin=0.05,
    )
    return bulunanlar, time.perf_counter() - baslangic


def easyocr_oku_eski_tc_fallback(resim):
    """SADECE GEREKİRSE çalışan geniş eski-TC OCR: ilk hızlı ROI sonucunda
    TC/AD/SOYAD alanlarından biri bulunamazsa çağrılır."""
    baslangic = time.perf_counter()
    if resim is None:
        return [], 0.0

    h, w = resim.shape[:2]
    x1, x2 = int(w * 0.015), int(w * 0.99)
    # Biraz daha geniş alan: farklı basım/kırpma varyasyonlarına karşı fallback.
    y1, y2 = int(h * 0.40), int(h * 0.90)

    roi = resim[y1:y2, x1:x2]
    if roi.size == 0:
        return [], 0.0

    olcek = 1.10  # sadece fallback'te hafif büyütme
    roi_ocr = cv2.resize(roi, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_LINEAR)

    try:
        sonuclar = reader_getir().readtext(
            roi_ocr, detail=1, paragraph=False, decoder="greedy",
            text_threshold=0.45, low_text=0.25, link_threshold=0.30,
            min_size=8, width_ths=0.75, ycenter_ths=0.5, add_margin=0.06,
        )
    except Exception as e:
        print("Eski TC fallback EasyOCR hatası:", repr(e))
        return [], time.perf_counter() - baslangic

    bulunanlar = []
    for sonuc in sonuclar:
        try:
            box, text_ocr, conf = sonuc
        except Exception:
            continue

        text_ocr = str(text_ocr).strip()
        if not text_ocr:
            continue

        box = np.asarray(box, dtype=np.float32)
        bulunanlar.append({
            "text": text_ocr, "norm": normalize_text(text_ocr), "conf": float(conf),
            "x1": int(box[:, 0].min() / olcek) + x1, "y1": int(box[:, 1].min() / olcek) + y1,
            "x2": int(box[:, 0].max() / olcek) + x1, "y2": int(box[:, 1].max() / olcek) + y1,
        })

    return bulunanlar, time.perf_counter() - baslangic


# =========================================================
# GÖÇMEN — YALNIZCA GEREKLİ ALANLAR
# =========================================================

def _birlesik_bolge_ocr_oku(resim, bolgeler, ayirici=24):
    """Aynı genişlikteki uzak kart bantlarını birleştirip tek OCR çalıştırır."""
    parcalar = []
    eslemeler = []
    canvas_y = 0
    beklenen_genislik = None

    for index, (x1, y1, x2, y2) in enumerate(bolgeler):
        roi = resim[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        if beklenen_genislik is None:
            beklenen_genislik = roi.shape[1]
        if roi.shape[1] != beklenen_genislik:
            raise ValueError("Birleştirilen OCR bölgeleri aynı genişlikte olmalıdır.")

        if index > 0 and parcalar:
            ayirici_shape = (ayirici, beklenen_genislik, *roi.shape[2:])
            parcalar.append(np.full(ayirici_shape, 255, dtype=resim.dtype))
            canvas_y += ayirici

        baslangic = canvas_y
        parcalar.append(roi.copy())
        canvas_y += roi.shape[0]
        eslemeler.append({
            "canvas_y1": baslangic, "canvas_y2": canvas_y,
            "kart_x": x1, "kart_y": y1,
        })

    if not parcalar:
        return [], 0.0

    canvas = np.concatenate(parcalar, axis=0)
    ocr, sure = easyocr_oku(canvas)
    bulunanlar = []

    for item in ocr:
        merkez = (item["y1"] + item["y2"]) / 2.0
        esleme = next(
            (x for x in eslemeler if x["canvas_y1"] <= merkez < x["canvas_y2"]),
            None,
        )
        if esleme is None:
            continue

        mapped = dict(item)
        mapped["x1"] = item["x1"] + esleme["kart_x"]
        mapped["x2"] = item["x2"] + esleme["kart_x"]
        mapped["y1"] = item["y1"] - esleme["canvas_y1"] + esleme["kart_y"]
        mapped["y2"] = item["y2"] - esleme["canvas_y1"] + esleme["kart_y"]
        bulunanlar.append(mapped)

    return bulunanlar, sure


def easyocr_oku_gocmen(resim):
    """Normalize gocmen kartini detector calistirmadan deger hucrelerinden okur.

    Kart perspektif duzeltmeden sonra sabit sablondadir. Bu nedenle pahali
    ``readtext`` detector gecisi yerine yalniz YKN, ad, soyad ve gecerlilik
    hucreleri dogrudan recognizer'a verilir. Uygulama asagida, hucre geometri
    yardimcilari tanimlandiktan sonra bulunan hizli-yol fonksiyonundadir.
    """
    if resim is None:
        return [], 0.0
    return _gocmen_dogrudan_hucre_ocr_oku(resim)


def _gocmen_detector_ocr_oku(resim):
    """Zor/uyumsuz kartta eski guvenli bant detector yolunu calistirir."""
    if resim is None:
        return [], 0.0
    h, w = resim.shape[:2]
    return _birlesik_bolge_ocr_oku(
        resim,
        [
            (0, int(h * 0.39), w, int(h * 0.530)),
            (0, int(h * 0.74), w, int(h * 0.88)),
        ],
        ayirici=24,
    )


# Perspektif duzeltme gocmen kartini 720x1100 sabit sablona getiriyor. Ilk
# detector gecisi bir value metnini kacirirsa tum bandi yeniden detect etmek
# yerine yalnizca bu sabit hucreler recognizer'a verilir. Koordinatlar mevcut
# hizli OCR bantlarinin (0.39-.530 ve 0.74-.88) icinde kalir.
_GOCMEN_FALLBACK_HUCRELERI = {
    "kimlik_no": (0.455, 0.414, 0.985, 0.452),
    "ad": (0.455, 0.448, 0.985, 0.490),
    "soyad": (0.455, 0.484, 0.985, 0.528),
    "gecerlilik": (0.455, 0.785, 0.985, 0.850),
}

_GOCMEN_NO_ALLOWLIST = "0123456789"
_GOCMEN_TARIH_ALLOWLIST = "0123456789./- "
_GOCMEN_ISIM_BLOCKLIST = "0123456789"


def _gocmen_label_haritasi(ocr):
    """Ana detector'un gordugu alan etiketlerini fiziksel satir ankraji yapar."""
    harita = {}
    for satir in satirlara_grupla(ocr):
        for tip, alan in (
            ("kimlik_no", "kimlik_no"),
            ("ad", "ad"),
            ("soyad", "soyad"),
            ("baba", "baba"),
            ("anne", "anne"),
            ("gecerlilik", "gecerlilik"),
        ):
            if alan in harita:
                continue
            label = satir_label_bul(satir, tip)
            if label is not None:
                harita[alan] = label
    return harita


def _gocmen_gri(kart):
    if kart.ndim == 3:
        renk = kart[..., :3].astype(np.float32, copy=False)
        gri = renk[..., 0] * 0.114 + renk[..., 1] * 0.587 + renk[..., 2] * 0.299
    else:
        gri = kart.astype(np.float32, copy=False)
    return np.nan_to_num(gri, nan=255.0, posinf=255.0, neginf=0.0)


def _gocmen_sablon_dikey_kaymasi(kart, label_haritasi):
    """Eksik hedef label icin gorulen tablo labellarindan ortak y kaymasi bulur."""
    h = kart.shape[0]
    kaymalar = []
    for alan in ("kimlik_no", "ad", "soyad"):
        label = (label_haritasi or {}).get(alan)
        if label is None:
            continue
        _, ry1, _, ry2 = _GOCMEN_FALLBACK_HUCRELERI[alan]
        sablon_merkezi = h * ((ry1 + ry2) / 2.0)
        label_merkezi = (float(label["y1"]) + float(label["y2"])) / 2.0
        kaymalar.append(label_merkezi - sablon_merkezi)

    if not kaymalar:
        return 0.0
    # Tek label gercek bir sablon kaymasini gosterebilir ama yanlis bir OCR
    # kutusunun birkac satir suruklemesine izin verilmez. Birden cok label'in
    # median uzlasmasinda daha genis, tek kanitta daha dar limit kullanilir.
    limit = h * (0.10 if len(kaymalar) >= 2 else 0.055)
    return float(np.clip(np.median(kaymalar), -limit, limit))


def _gocmen_projeksiyon_esigi(gri):
    """Tarama arka planina gore koyu-piksel esigi; duz/bozuk ROI'de None."""
    if gri.size == 0:
        return None
    alt, ust = np.percentile(gri, (5.0, 90.0))
    if ust - alt < 18.0:
        return None
    return float(alt + (ust - alt) * 0.38)


def _gocmen_deger_sutunu_projeksiyondan_bul(kart, geometri_gri=None):
    """Kesintisiz dikey tablo ayiricisini bulur; kanit yoksa None dondurur."""
    h, w = kart.shape[:2]
    gri = geometri_gri if geometri_gri is not None else _gocmen_gri(kart)
    # Sablonun tamaminda dikey kalan ayiriciyi genis bir bantta ara. Onceki
    # dar .39-.55 araligi tablo dikey kaydiginda veya farkli olcekte
    # normalize edildiginde cizginin kisa bir parcasini gorup reddedebiliyordu.
    # Bu yalnizca bir NumPy projeksiyonudur; OCR/detector maliyeti eklemez.
    by1, by2 = int(h * 0.30), int(h * 0.91)
    bx1, bx2 = int(w * 0.32), int(w * 0.64)
    bant = gri[by1:by2, bx1:bx2]
    esik = _gocmen_projeksiyon_esigi(bant)
    if esik is None and bant.size:
        # Genis aralikta ince ayirici toplam piksellerin %5'inden az olabilir.
        # Alt yuzdeligi yalniz bu cizgi aramasinda dusurerek temiz taramadaki
        # gercek kontrasti kaybetmeyiz.
        alt, ust = np.percentile(bant, (1.0, 90.0))
        if ust - alt >= 18.0:
            esik = float(alt + (ust - alt) * 0.38)
    if esik is None or bant.size == 0:
        return None

    skorlar = np.mean(bant < esik, axis=0)
    tepe_index = int(np.argmax(skorlar))
    tepe = float(skorlar[tepe_index])
    taban = float(np.percentile(skorlar, 75.0))
    if tepe < 0.42 or tepe - taban < 0.12:
        return None

    # ``tepe_index`` kalin ayirici cizgisinin genellikle en soldaki pikselidir.
    # Buraya sabit/genislige oranli bir bosluk eklemek, deger cizgiye bitisik
    # basildiginda YKN'nin ilk 9 hanesini de kirpabiliyor. Cizginin gercek sag
    # kenarini projeksiyondan bulup ilk kullanilabilir pikselden basla. Bu yalniz
    # NumPy uzerinde birkac sutun gezer; yeni OCR/model cagrisi eklemez.
    cizgi_esigi = max(0.36, taban + 0.10, tepe * 0.70)
    sag_kenar = tepe_index
    en_fazla_cizgi_genisligi = max(3, int(round(w * 0.012)))
    while (
        sag_kenar + 1 < len(skorlar)
        and sag_kenar - tepe_index + 1 < en_fazla_cizgi_genisligi
        and float(skorlar[sag_kenar + 1]) >= cizgi_esigi
    ):
        sag_kenar += 1
    return int(np.clip(
        bx1 + sag_kenar + 1,
        w * 0.40,
        w * 0.62,
    ))


# Normalize referansta hedef value hucrelerini cevreleyen yatay tablo
# cizgileri. Direct yol bu oranlari dogrudan crop olarak kullanmak yerine,
# goruntudeki gercek cizgilere tek bir affine (olcek+kayma) modelle esler.
_GOCMEN_GRID_REFERANS_SINIRLARI = {
    "kimlik_no": (0.416, 0.453),
    "ad": (0.453, 0.489),
    "soyad": (0.489, 0.525),
    "gecerlilik": (0.785, 0.846),
}

# Referans taramada YKN satiri ad/soyad satirlarindan daha kisa. Bazi eski
# baskilarda ise ilk uc satir esit yukseklikte. Iki bilinen topolojiyi de ayni
# ucuz affine esleme icinde sinamak, yanlis satiri YKN diye kesmekten guvenlidir.
_GOCMEN_GRID_REFERANS_PROFILLERI = (
    ((0.419, 0.445, 0.483, 0.520), (0.784, 0.846)),
    ((0.416, 0.453, 0.489, 0.525), (0.785, 0.846)),
)


def _gocmen_yatay_cizgi_adaylari(kart, deger_x1, geometri_gri=None):
    """Value tablosundaki uzun yatay cizgileri tek projeksiyonda bulur."""
    h, w = kart.shape[:2]
    gri = geometri_gri if geometri_gri is not None else _gocmen_gri(kart)
    y1, y2 = int(h * 0.28), int(h * 0.94)
    x1, x2 = max(int(deger_x1), int(w * 0.39)), int(w * 0.985)
    bant = gri[y1:y2, x1:x2]
    if bant.shape[0] < 3 or bant.shape[1] < 20:
        return []

    # Koyu-piksel kaplamasi gercek basili cizgiyi bulur. Cok temiz sentetik
    # veya soluk taramada koyu pikseller tum bandin %5'inden az kalabildigi
    # icin ikinci, bagimsiz kanit olarak satirlar arasi gradyan da kullanilir.
    alt, ust = np.percentile(bant, (1.0, 90.0))
    koyu_skorlari = np.zeros(bant.shape[0], dtype=np.float32)
    if ust - alt >= 18.0:
        esik = float(alt + (ust - alt) * 0.38)
        koyu_skorlari = np.mean(bant < esik, axis=1).astype(np.float32)
    koyu_taban = float(np.percentile(koyu_skorlari, 75.0))

    kenar = np.mean(
        np.abs(np.diff(bant.astype(np.float32, copy=False), axis=0)),
        axis=1,
    )
    kenar_skorlari = np.zeros(bant.shape[0], dtype=np.float32)
    kenar_skorlari[:-1] = kenar
    kenar_taban = float(np.percentile(kenar, 75.0)) if kenar.size else 0.0
    piksel_kontrasti = float(
        np.percentile(bant, 95.0) - np.percentile(bant, 5.0)
    )

    kanit_maskesi = np.zeros(bant.shape[0], dtype=bool)
    kalite_skorlari = np.zeros(bant.shape[0], dtype=np.float32)
    for indeks in range(bant.shape[0]):
        koyu = float(koyu_skorlari[indeks])
        kenar_degeri = float(kenar_skorlari[indeks])
        koyu_kanit = koyu >= 0.42 and koyu - koyu_taban >= 0.11
        kenar_kanit = (
            kenar_degeri >= max(10.0, piksel_kontrasti * 0.075)
            and kenar_degeri - kenar_taban >= max(
                4.5, piksel_kontrasti * 0.030,
            )
        )
        if not (koyu_kanit or kenar_kanit):
            continue
        kanit_maskesi[indeks] = True
        kalite_skorlari[indeks] = max(
            koyu * 100.0 + max(0.0, koyu - koyu_taban) * 80.0,
            kenar_degeri - kenar_taban,
        )

    # Uzun koyu fotograf/metin bolgeleri onlarca ardışık satiri aday yapabilir.
    # Yalniz yerel tepeleri tutmak, bu bolgelerin ince gercek grid cizgisini
    # zincirleme bicimde yutmasini engeller.
    adaylar = []
    for indeks in np.flatnonzero(kanit_maskesi):
        sol, sag = max(0, indeks - 2), min(len(kalite_skorlari), indeks + 3)
        if kalite_skorlari[indeks] + 1e-6 < np.max(kalite_skorlari[sol:sag]):
            continue
        adaylar.append((y1 + int(indeks), float(kalite_skorlari[indeks])))

    if not adaylar:
        return []

    # Kalin cizginin iki kenari ayri aday olmasin. Birbirine en fazla bes
    # piksel uzaktaki tepeleri tek cizgi kabul edip en guclu satiri sakla.
    gruplar = [[adaylar[0]]]
    for aday in adaylar[1:]:
        if aday[0] - gruplar[-1][-1][0] <= 5:
            gruplar[-1].append(aday)
        else:
            gruplar.append([aday])
    return [
        (
            int(round(float(np.median([aday[0] for aday in grup])))),
            max(aday[1] for aday in grup),
        )
        for grup in gruplar
    ]


def _gocmen_dinamik_hucre_sinirlari(
    kart, deger_x1, geometri_gri=None,
):
    """Kaymis/olceklenmis tabloyu hedef hucrelere guvenli bicimde esler.

    Ilk uc hedef satirin dort ardisik siniri affine modeli kurar. Gecerlilik
    satirinin iki siniri de ayni modeli desteklemiyorsa direct yol reddedilir;
    boylece dogum tarihi gibi baska bir satira sessizce kayilmaz.
    """
    h = kart.shape[0]
    adaylar = _gocmen_yatay_cizgi_adaylari(
        kart, deger_x1, geometri_gri,
    )
    if len(adaylar) < 6:
        return None

    ys = np.asarray([x[0] for x in adaylar], dtype=np.float64)
    kaliteler = np.asarray([x[1] for x in adaylar], dtype=np.float64)
    en_iyi = None
    # Birinci ve dorduncu cizgi adayindan olcek+kayma tahmin edilir. Aradaki
    # iki satir ile uzaktaki gecerlilik satiri ayni modeli dogrulamak zorunda.
    for profil_ust, profil_gecerlilik in _GOCMEN_GRID_REFERANS_PROFILLERI:
        referans_ust = np.asarray(profil_ust, dtype=np.float64) * h
        referans_gecerlilik = np.asarray(
            profil_gecerlilik, dtype=np.float64,
        ) * h
        for ilk_indeks in range(len(ys) - 3):
            for son_indeks in range(ilk_indeks + 3, len(ys)):
                olcek = (ys[son_indeks] - ys[ilk_indeks]) / (
                    referans_ust[-1] - referans_ust[0]
                )
                if not 0.70 <= olcek <= 1.38:
                    continue
                kayma = ys[ilk_indeks] - olcek * referans_ust[0]
                if abs(kayma) > h * 0.18:
                    continue

                tahmin_ust = referans_ust * olcek + kayma
                eslesen_indeksler = [ilk_indeks]
                hata = 0.0
                onceki = ilk_indeks
                gecerli = True
                for tahmin in tahmin_ust[1:-1]:
                    aralik = np.arange(onceki + 1, son_indeks)
                    if aralik.size == 0:
                        gecerli = False
                        break
                    secilen = int(
                        aralik[np.argmin(np.abs(ys[aralik] - tahmin))]
                    )
                    fark = abs(ys[secilen] - tahmin)
                    if fark > max(6.0, h * 0.008):
                        gecerli = False
                        break
                    eslesen_indeksler.append(secilen)
                    onceki = secilen
                    hata += fark
                if not gecerli:
                    continue
                eslesen_indeksler.append(son_indeks)
                if len(set(eslesen_indeksler)) != 4:
                    continue

                tahmin_gecerlilik = referans_gecerlilik * olcek + kayma
                gec_indeksleri = []
                onceki = eslesen_indeksler[-1]
                for tahmin in tahmin_gecerlilik:
                    aralik = np.arange(onceki + 1, len(ys))
                    if aralik.size == 0:
                        gecerli = False
                        break
                    secilen = int(
                        aralik[np.argmin(np.abs(ys[aralik] - tahmin))]
                    )
                    fark = abs(ys[secilen] - tahmin)
                    if fark > max(8.0, h * 0.014):
                        gecerli = False
                        break
                    gec_indeksleri.append(secilen)
                    onceki = secilen
                    hata += fark
                if not gecerli or len(set(gec_indeksleri)) != 2:
                    continue

                gec_yuksekligi = (
                    ys[gec_indeksleri[1]] - ys[gec_indeksleri[0]]
                )
                if not h * 0.035 <= gec_yuksekligi <= h * 0.10:
                    continue

                tum_indeksler = [*eslesen_indeksler, *gec_indeksleri]
                # Benzer birden cok duzen varsa referansa en az zorlamayla
                # uyan ve cizgi kaniti daha guclu olan model secilir.
                puan = (
                    hata / h
                    + abs(olcek - 1.0) * 0.045
                    + abs(kayma) / h * 0.20
                    - min(
                        float(np.mean(kaliteler[tum_indeksler])), 180.0,
                    ) * 0.00001
                )
                if en_iyi is None or puan < en_iyi[0]:
                    en_iyi = (puan, eslesen_indeksler, gec_indeksleri)

    if en_iyi is None:
        return None

    _, ilk_satirlar, gecerlilik_satirlari = en_iyi
    sinir_yleri = [int(ys[x]) for x in ilk_satirlar]
    gec_yleri = [int(ys[x]) for x in gecerlilik_satirlari]
    return {
        "kimlik_no": (sinir_yleri[0], sinir_yleri[1]),
        "ad": (sinir_yleri[1], sinir_yleri[2]),
        "soyad": (sinir_yleri[2], sinir_yleri[3]),
        "gecerlilik": (gec_yleri[0], gec_yleri[1]),
    }


def _gocmen_deger_sutunu_sol_siniri(
    kart, ocr, sonuc, label_haritasi, geometri_gri=None,
):
    """Deger sutununu mevcut value kutulari veya kesintisiz tablo cizgisiyle bulur."""
    h, w = kart.shape[:2]
    aday_xler = []

    # Parser bir alani bulduysa onun kutusu ayni tablodaki diger value hucreleri
    # icin en guvenli x ankrajidir. Parserin reddettigi ama dogru satirda duran
    # detector parcalari da (ornegin checksum gecmeyen YKN) sutunu gosterebilir.
    for alan in ("kimlik_no", "ad", "soyad"):
        item = sonuc.get(f"{alan}_item")
        if item is not None and w * 0.38 <= float(item.get("x1", 0)) <= w * 0.68:
            aday_xler.append(float(item["x1"]))

        label = label_haritasi.get(alan)
        if label is None:
            continue
        merkez = (float(label["y1"]) + float(label["y2"])) / 2.0
        tolerans = max(h * 0.018, (float(label["y2"]) - float(label["y1"])) * 0.72)
        for ocr_item in ocr:
            item_merkez = (float(ocr_item["y1"]) + float(ocr_item["y2"])) / 2.0
            item_x1 = float(ocr_item.get("x1", 0))
            if (
                ocr_item is not label
                and abs(item_merkez - merkez) <= tolerans
                and w * 0.38 <= item_x1 <= w * 0.68
            ):
                aday_xler.append(item_x1)

    if aday_xler:
        # Uzun ad EasyOCR tarafindan ILHAM / MUSTAFA / YOUNUS gibi birden cok
        # kutuya bolunebilir. Median bu durumda sutun baslangicini ikinci kelimeye
        # kaydirip ilk kelimeyi kirpiyordu. Ayni alan satirlarindaki en soldaki
        # value parcasi, gercek deger sutunu icin guvenli ankrajdir.
        return int(np.clip(min(aday_xler) - w * 0.012, w * 0.40, w * 0.62))

    # Label/value metni bulunamasa bile tablo ayirici cizgisi tum uc satir boyunca
    # devam eder. Dikey projeksiyon, sabit .455 varsayimindan kayan basimlarda da
    # cizgiyi yakalar. Yeterli belirginlik yoksa konservatif sabit koordinata doner.
    projeksiyon_x = _gocmen_deger_sutunu_projeksiyondan_bul(
        kart, geometri_gri,
    )
    if projeksiyon_x is not None:
        return projeksiyon_x

    return int(w * 0.462)


def _gocmen_yatay_cizgi_siniri(
    kart, merkez, yon, deger_x1, geometri_gri=None,
):
    """Satir merkezinin ust/altindaki uzun tablo cizgisini bulur."""
    h, w = kart.shape[:2]
    gri = geometri_gri if geometri_gri is not None else _gocmen_gri(kart)
    uzak_min = max(4, int(h * 0.006))
    uzak_max = max(uzak_min + 2, int(h * 0.048))
    if yon < 0:
        sy1, sy2 = int(merkez) - uzak_max, int(merkez) - uzak_min
    else:
        sy1, sy2 = int(merkez) + uzak_min, int(merkez) + uzak_max
    sy1, sy2 = max(0, sy1), min(h, sy2)
    sx1, sx2 = max(0, int(deger_x1)), min(w, int(w * 0.985))
    bant = gri[sy1:sy2, sx1:sx2]
    if bant.size == 0:
        return None

    esik = _gocmen_projeksiyon_esigi(bant)
    if esik is not None:
        skorlar = np.mean(bant < esik, axis=1)
        tepe_index = int(np.argmax(skorlar))
        tepe = float(skorlar[tepe_index])
        taban = float(np.percentile(skorlar, 70.0))
        if tepe >= 0.45 and tepe - taban >= 0.12:
            return sy1 + tepe_index

    # Dokulu/eski baskida satir ici metin koyu-piksel projeksiyonunun tabanini
    # yukseltebilir. Tablo cizgisinin iki kenari yine genislik boyunca keskin bir
    # gri-seviye degisimi olusturur; ikinci kanit olarak yatay gradyani kullan.
    if bant.shape[0] < 2:
        return None
    kenar_skorlari = np.mean(
        np.abs(np.diff(bant.astype(np.float32, copy=False), axis=0)),
        axis=1,
    )
    tepe_index = int(np.argmax(kenar_skorlari))
    tepe = float(kenar_skorlari[tepe_index])
    taban = float(np.percentile(kenar_skorlari, 75.0))
    kontrast = float(np.percentile(bant, 95.0) - np.percentile(bant, 5.0))
    if tepe < max(10.0, kontrast * 0.08):
        return None
    if tepe - taban < max(5.0, kontrast * 0.035):
        return None
    return sy1 + tepe_index + 1


def _gocmen_hucre_kutusu(
    kart, alan, label_item=None, label_haritasi=None, deger_x1=None,
    geometri_gri=None, dinamik_sinirlar=None,
):
    h, w = kart.shape[:2]
    rx1, ry1, rx2, ry2 = _GOCMEN_FALLBACK_HUCRELERI[alan]
    x1, y1 = max(0, int(w * rx1)), max(0, int(h * ry1))
    x2, y2 = min(w, int(w * rx2)), min(h, int(h * ry2))

    if deger_x1 is not None:
        x1 = max(0, min(x2 - 4, int(deger_x1)))

    if dinamik_sinirlar and alan in dinamik_sinirlar:
        ust, alt = dinamik_sinirlar[alan]
        if 12 <= alt - ust <= h * 0.10:
            return [x1, x2, min(h, int(ust) + 2), max(0, int(alt) - 2)]

    if label_item is None:
        # Hedefin kucuk sol etiketi ana OCR bandinda kacabilir. Diger gorulen
        # tablo labellari sablonun ortak dikey kaymasini verir; sabit hucreyi bu
        # kaymayla tasiyarak bir ust/alt alanin degerini okumayi engelleriz.
        kayma = _gocmen_sablon_dikey_kaymasi(kart, label_haritasi)
        y1 = max(0, min(h - 4, int(round(y1 + kayma))))
        y2 = min(h, max(y1 + 4, int(round(y2 + kayma))))
    else:
        # Farkli basimlarda tablo birkac satir yukari/asagi kayabiliyor. Sabit
        # y koordinati yerine ana OCR'in buldugu sol etiketi ankraj al; deger
        # sutununun x siniri sablon boyunca sabit kalir.
        label_y1 = float(label_item["y1"])
        label_y2 = float(label_item["y2"])
        label_h = max(1.0, label_y2 - label_y1)
        merkez = (label_y1 + label_y2) / 2.0
        if alan == "gecerlilik":
            yari_yukseklik = max(h * 0.028, min(h * 0.042, label_h * 0.72))
        else:
            yari_yukseklik = max(h * 0.014, min(h * 0.023, label_h * 0.62))
            # En yakin alan etiketi komsu satirin nerede basladigini gosterir.
            # Crop yaricapini iki satir merkezinin yarisindan kucuk tutarak adin
            # soyada, soyadin baba adina tasmasini engelleriz.
            komsu_merkezler = []
            for komsu in (label_haritasi or {}).values():
                if komsu is label_item:
                    continue
                komsu_merkez = (float(komsu["y1"]) + float(komsu["y2"])) / 2.0
                if abs(komsu_merkez - merkez) <= h * 0.09:
                    komsu_merkezler.append(komsu_merkez)
            if komsu_merkezler:
                en_yakin = min(abs(x - merkez) for x in komsu_merkezler)
                yari_yukseklik = min(yari_yukseklik, en_yakin * 0.44)
        y1 = max(0, int(merkez - yari_yukseklik))
        y2 = min(h, int(merkez + yari_yukseklik))

    merkez = (y1 + y2) / 2.0
    ust_cizgi = _gocmen_yatay_cizgi_siniri(
        kart, merkez, -1, x1, geometri_gri,
    )
    alt_cizgi = _gocmen_yatay_cizgi_siniri(
        kart, merkez, 1, x1, geometri_gri,
    )
    # Sinir cizgilerini hucreye dahil etme; ikisi birden bulununca projeksiyon
    # geometrisi guvenilirdir. Tek cizgiyle sabit kutuyu gereksiz daraltmayiz.
    if ust_cizgi is not None and alt_cizgi is not None and alt_cizgi - ust_cizgi >= 12:
        y1 = min(h, ust_cizgi + 2)
        y2 = max(y1 + 4, alt_cizgi - 2)
    return [x1, x2, y1, y2]


def _gocmen_hucreleri_tani(
    kart, alanlar, allowlist=None, label_haritasi=None, deger_x1=None,
    blocklist=None, geometri_gri=None, dinamik_sinirlar=None,
):
    """Verilen hucreleri detector calistirmadan tek recognize cagrisinda tanir."""
    if not alanlar:
        return {}, 0.0

    baslangic = time.perf_counter()
    label_haritasi = label_haritasi or {}
    kutular = [
        _gocmen_hucre_kutusu(
            kart, alan, label_haritasi.get(alan), label_haritasi, deger_x1,
            geometri_gri, dinamik_sinirlar,
        )
        for alan in alanlar
    ]
    try:
        sonuclar = reader_getir().recognize(
            kart, horizontal_list=kutular, free_list=[],
            decoder="greedy", beamWidth=5,
            # Ayni karakter kumesine sahip hucreleri gercek bir tensor batch'i
            # olarak tanit. Ozellikle CPU'da iki ayri ileri gecis yerine tek
            # ileri gecis kullanmak ad+soyad ve YKN+tarih maliyetini dusurur.
            batch_size=max(1, min(4, len(kutular))), workers=0,
            allowlist=allowlist, blocklist=blocklist, detail=1, rotation_info=None,
            # Net hucrede EasyOCR'in dahili dusuk-kontrast ikinci turunu kapat.
            # Gecersiz sonuc icin kontrollu onislem fallback'i zaten var.
            paragraph=False, contrast_ths=0.0, adjust_contrast=0.5,
            filter_ths=0.003,
        )
    except Exception as e:
        print("Gocmen hucre OCR hatasi:", repr(e))
        return {}, time.perf_counter() - baslangic

    bulunanlar = {}
    # EasyOCR, horizontal_list sonucunu verilen kutu sirasinda dondurur. Bos bir
    # hucre de bu sirada bos text olarak yer aldigi icin alan eslemesi kaymaz.
    for alan, kutu, sonuc in zip(alanlar, kutular, sonuclar):
        try:
            _, text_ocr, conf = sonuc
            conf = float(conf)
        except (TypeError, ValueError):
            continue

        text_ocr = str(text_ocr).strip()
        if not text_ocr:
            continue

        x1, x2, y1, y2 = kutu
        bulunanlar[alan] = {
            "text": text_ocr, "norm": normalize_text(text_ocr), "conf": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "_fallback_alan": alan,
        }

    return bulunanlar, time.perf_counter() - baslangic


def _gocmen_sag_bitis_hucresini_tani(
    kart, deger_x1, geometri_gri, dinamik_sinirlar,
):
    """Dogrulanmis gecerlilik satirinin sag bolumunu tek bbox ile okur.

    Bu kosullu gecis yalniz tam gecerlilik hucresi iki tarih uretemediginde
    calisir. Dolayisiyla normal temiz yolun 2 API / 4 bbox maliyeti degismez;
    tek tarih cikarsa bu yalnizca ucuncu ve son bir API gecisidir.
    """
    baslangic = time.perf_counter()
    tam_kutu = _gocmen_hucre_kutusu(
        kart, "gecerlilik", None, {}, deger_x1, geometri_gri,
        dinamik_sinirlar,
    )
    x1, x2, y1, y2 = tam_kutu
    # Baslangic tarihini disarida birakacak kadar sagdan basla; ayirac ve bitis
    # tarihinin ilk rakamlarini kirpmamak icin tam ortanin biraz solunda kal.
    sag_x1 = int(round(x1 + (x2 - x1) * 0.46))
    kutu = [max(x1, min(x2 - 8, sag_x1)), x2, y1, y2]
    try:
        sonuclar = reader_getir().recognize(
            kart, horizontal_list=[kutu], free_list=[],
            decoder="greedy", beamWidth=5, batch_size=1, workers=0,
            allowlist=_GOCMEN_TARIH_ALLOWLIST, blocklist=None, detail=1,
            rotation_info=None, paragraph=False, contrast_ths=0.0,
            adjust_contrast=0.5, filter_ths=0.003,
        )
    except Exception as e:
        print("Gocmen bitis hucresi OCR hatasi:", repr(e))
        return None, time.perf_counter() - baslangic

    if not sonuclar:
        return None, time.perf_counter() - baslangic
    try:
        _, text_ocr, conf = sonuclar[0]
        conf = float(conf)
    except (TypeError, ValueError):
        return None, time.perf_counter() - baslangic
    text_ocr = str(text_ocr).strip()
    if not text_ocr:
        return None, time.perf_counter() - baslangic
    return {
        "text": text_ocr,
        "norm": normalize_text(text_ocr),
        "conf": conf,
        "x1": kutu[0], "x2": kutu[1], "y1": kutu[2], "y2": kutu[3],
        "_fallback_alan": "gecerlilik",
        "_bitis_hucresi": True,
        "_geometri_dogrulandi": True,
        "_dogrudan_hucre_ocr": True,
    }, time.perf_counter() - baslangic


def _gocmen_dogrudan_geometri_guvenli_mi(
    kart, deger_x1, geometri_gri=None,
):
    """Direct yol icin her hedef hucrenin iki grid sinirini zorunlu tutar."""
    h = kart.shape[0]
    for alan in ("kimlik_no", "ad", "soyad", "gecerlilik"):
        _, ry1, _, ry2 = _GOCMEN_FALLBACK_HUCRELERI[alan]
        merkez = h * ((ry1 + ry2) / 2.0)
        ust = _gocmen_yatay_cizgi_siniri(
            kart, merkez, -1, deger_x1, geometri_gri,
        )
        alt = _gocmen_yatay_cizgi_siniri(
            kart, merkez, 1, deger_x1, geometri_gri,
        )
        if ust is None or alt is None or not (12 <= alt - ust <= h * 0.09):
            return False
    return True


_GOCMEN_SENTETIK_LABEL_METINLERI = {
    "kimlik_no": "YABANCI KIMLIK NO",
    "ad": "ADI",
    "soyad": "SOYADI",
    "gecerlilik": "BELGENIN GECERLILIK TARIHI",
}


def _gocmen_sentetik_label_itemi(kart, alan, kutu, deger_x1):
    """Detector'siz value sonucunu mevcut guvenli parser sozlesmesine baglar."""
    _, w = kart.shape[:2]
    _, _, y1, y2 = kutu
    text = _GOCMEN_SENTETIK_LABEL_METINLERI[alan]
    return {
        "text": text,
        "norm": normalize_text(text),
        "conf": 1.0,
        "x1": int(w * 0.01),
        "y1": y1,
        "x2": max(int(w * 0.08), int(deger_x1 - w * 0.015)),
        "y2": y2,
        # Eksik alan fallback'i ayni ham hucreyi ikinci kez okumasin; bu isaret
        # ilk recognize denemesinin detector'siz hizli yolda yapildigini bildirir.
        "_dogrudan_hucre_ocr": True,
        "_sentetik_label": True,
    }



def _gocmen_sablon_kaydirma_kutusu(
    kart,
    alan,
    kayma_y,
    deger_x1
):
    """
    Normalize gocmen sablonundaki value hucrelerinden birini,
    tum tabloya uygulanan ortak dikey kaymayla uretir.

    Bu yol, strict grid projeksiyonu yeterli cizgi bulamadiginda
    kullanilir. Alanlar birbirinden bagimsiz kaydirilmaz; butun
    sablon tek parca halinde yukari/asagi tasinir. Boylece AD ve
    SOYAD'in ayni satira dusmesi gibi hatalar azalir.
    """
    h, w = kart.shape[:2]
    _, ry1, rx2, ry2 = _GOCMEN_FALLBACK_HUCRELERI[alan]

    x1 = int(np.clip(
        deger_x1,
        w * 0.39,
        w * 0.64,
    ))
    x2 = int(np.clip(
        w * rx2,
        x1 + 10,
        w,
    ))

    y1 = int(round(h * ry1 + kayma_y))
    y2 = int(round(h * ry2 + kayma_y))

    # Hucre cizgilerini ve komsu satirin harflerini bir miktar disarida tut.
    ic_marj = max(1, int(round(h * 0.002)))
    y1 += ic_marj
    y2 -= ic_marj

    y1 = max(0, min(h - 6, y1))
    y2 = min(h, max(y1 + 6, y2))

    return [x1, x2, y1, y2]


def _gocmen_toplu_kutulari_tani(
    kart,
    eslemeler,
    allowlist=None,
    blocklist=None
):
    """
    Birden cok global sablon kaymasinin ayni tur hucrelerini
    TEK EasyOCR recognize cagrisi ile okur.

    eslemeler:
        [(kayma_indeksi, alan, kutu), ...]

    Detector calismaz. Bu yuzden 8-9 saniyelik full-band
    readtext fallback'inden cok daha ucuzdur.
    """
    if not eslemeler:
        return {}, 0.0

    baslangic = time.perf_counter()
    kutular = [x[2] for x in eslemeler]

    try:
        sonuclar = reader_getir().recognize(
            kart,
            horizontal_list=kutular,
            free_list=[],
            decoder="greedy",
            beamWidth=5,
            batch_size=max(1, min(8, len(kutular))),
            workers=0,
            allowlist=allowlist,
            blocklist=blocklist,
            detail=1,
            rotation_info=None,
            paragraph=False,
            contrast_ths=0.0,
            adjust_contrast=0.5,
            filter_ths=0.003,
        )
    except Exception as e:
        print("Gocmen kaydirilmis sablon OCR hatasi:", repr(e))
        return {}, time.perf_counter() - baslangic

    bulunan = {}
    for esleme, sonuc in zip(eslemeler, sonuclar):
        kayma_indeksi, alan, kutu = esleme

        try:
            _, text_ocr, conf = sonuc
            text_ocr = str(text_ocr).strip()
            conf = float(conf)
        except (TypeError, ValueError):
            continue

        if not text_ocr:
            continue

        x1, x2, y1, y2 = kutu
        bulunan.setdefault(kayma_indeksi, {})[alan] = {
            "text": text_ocr,
            "norm": normalize_text(text_ocr),
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "_fallback_alan": alan,
            "_dogrudan_hucre_ocr": True,
            "_kaydirilmis_sablon": True,
        }

    return bulunan, time.perf_counter() - baslangic


def _gocmen_kaydirilmis_sablon_skoru(sonuc):
    """
    Bir global sablon kaymasinin ne kadar mantikli veri verdigini puanlar.

    Sadece confidence'a guvenilmez:
      - YKN parser/checksum'dan gecmeli
      - AD ve SOYAD farkli fiziksel satirlardan gelmeli
      - tarih parser tarafindan gercek takvim tarihi olmali
    """
    skor = 0.0

    kimlik_no = sonuc.get("kimlik_no")
    if (
        kimlik_no not in {None, "", "Bulunamadi"}
        and yabanci_no_gecerli_mi(str(kimlik_no))
    ):
        skor += 9.0
        skor += min(
            1.0,
            float(sonuc.get("kimlik_no_conf", 0.0))
        )

    ad = sonuc.get("ad")
    if ad not in {None, "", "Bulunamadi"}:
        skor += 3.0
        skor += min(
            0.8,
            float(sonuc.get("ad_conf", 0.0)) * 0.8
        )

    soyad = sonuc.get("soyad")
    if soyad not in {None, "", "Bulunamadi"}:
        skor += 3.0
        skor += min(
            0.8,
            float(sonuc.get("soyad_conf", 0.0)) * 0.8
        )

    if (
        ad not in {None, "", "Bulunamadi"}
        and soyad not in {None, "", "Bulunamadi"}
        and normalize_text(ad) == normalize_text(soyad)
    ):
        skor -= 5.0

    bitis = sonuc.get("bitis_tarihi")
    if bitis not in {None, "", "Bulunamadi"}:
        skor += 5.0

    baslangic = sonuc.get("baslangic_tarihi")
    if baslangic not in {None, "", "Bulunamadi"}:
        skor += 1.0

    return skor


def _gocmen_kaydirilmis_sablon_ocr_oku(
    kart,
    deger_x1=None
):
    """
    STRICT GRID BASARISIZSA KULLANILAN HIZLI KURTARMA YOLU.

    Sorun:
        Kart dogru sekilde "gocmen" olarak taninmis olabilir fakat
        tarama/crop/warp farki yuzunden tablo 20-60 px yukari veya
        asagi kayabilir. Sabit hucreler bu durumda bos alan okur.

    Cozum:
        Hucreleri birbirinden bagimsiz aramak yerine tum sablonu ortak
        bir dikey kaymayla 7 konumda deneriz. Tum sayisal hucreler tek
        recognize batch'inde, tum isim hucreleri ikinci batch'te okunur.

    Bu yol:
        - full EasyOCR detector CALISTIRMAZ
        - label OCR'ina bagimli DEGILDIR
        - AD/SOYAD satir sirasini korur
        - en iyi kaymayi parser + YKN checksum + tarih dogrulamasi ile secer
    """
    baslangic = time.perf_counter()

    if kart is None:
        return [], 0.0

    h, w = kart.shape[:2]

    if deger_x1 is None:
        deger_x1 = int(w * 0.462)

    # Normalize warp'taki pratik kaymalar icin ±%7.
    # 1100 px kartta yaklasik ±77 px.
    kayma_oranlari = (
        -0.070,
        -0.046,
        -0.023,
        0.000,
        0.023,
        0.046,
        0.070,
    )
    kaymalar = [
        int(round(h * oran))
        for oran in kayma_oranlari
    ]

    sayisal_eslemeler = []
    isim_eslemeler = []

    for indeks, kayma in enumerate(kaymalar):
        for alan in ("kimlik_no", "gecerlilik"):
            sayisal_eslemeler.append((
                indeks,
                alan,
                _gocmen_sablon_kaydirma_kutusu(
                    kart,
                    alan,
                    kayma,
                    deger_x1,
                )
            ))

        for alan in ("ad", "soyad"):
            isim_eslemeler.append((
                indeks,
                alan,
                _gocmen_sablon_kaydirma_kutusu(
                    kart,
                    alan,
                    kayma,
                    deger_x1,
                )
            ))

    sayisal_sonuclar, _ = _gocmen_toplu_kutulari_tani(
        kart,
        sayisal_eslemeler,
        allowlist=_GOCMEN_TARIH_ALLOWLIST,
        blocklist=None,
    )

    isim_sonuclar, _ = _gocmen_toplu_kutulari_tani(
        kart,
        isim_eslemeler,
        allowlist=None,
        blocklist=_GOCMEN_ISIM_BLOCKLIST,
    )

    en_iyi = None

    for indeks, kayma in enumerate(kaymalar):
        hucreler = {}
        hucreler.update(
            sayisal_sonuclar.get(indeks, {})
        )
        hucreler.update(
            isim_sonuclar.get(indeks, {})
        )

        ocr = []

        for alan in (
            "kimlik_no",
            "ad",
            "soyad",
            "gecerlilik",
        ):
            item = hucreler.get(alan)
            kutu = (
                [
                    item["x1"],
                    item["x2"],
                    item["y1"],
                    item["y2"],
                ]
                if item is not None
                else
                _gocmen_sablon_kaydirma_kutusu(
                    kart,
                    alan,
                    kayma,
                    deger_x1,
                )
            )

            label = _gocmen_sentetik_label_itemi(
                kart,
                alan,
                kutu,
                deger_x1,
            )
            label = {
                **label,
                "_kaydirilmis_sablon": True,
                "_sablon_kayma_y": kayma,
            }
            ocr.append(label)

            if item is not None:
                ocr.append({
                    **item,
                    "_sablon_kayma_y": kayma,
                })

        sonuc = gocmen_bilgilerini_bul(
            ocr
        )
        skor = _gocmen_kaydirilmis_sablon_skoru(
            sonuc
        )

        # Esit skorda sifira daha yakin kaymayi tercih et.
        siralama = (
            skor,
            -abs(kayma),
        )

        if (
            en_iyi is None
            or siralama > en_iyi[0]
        ):
            en_iyi = (
                siralama,
                ocr,
                sonuc,
                kayma,
            )

    if en_iyi is None:
        return [], time.perf_counter() - baslangic

    _, en_iyi_ocr, en_iyi_sonuc, en_iyi_kayma = en_iyi
    en_iyi_skor = _gocmen_kaydirilmis_sablon_skoru(
        en_iyi_sonuc
    )

    # Yalnizca tek guvensiz isim cikti diye detector'u kapatma.
    # En az:
    #   - dogrulanmis YKN + bir isim
    # veya
    #   - YKN + gecerlilik
    # veya
    #   - iki isim + gecerlilik
    # gibi anlamli yapisal kanit ariyoruz.
    if en_iyi_skor < 11.0:
        return [], time.perf_counter() - baslangic

    return (
        en_iyi_ocr,
        time.perf_counter() - baslangic,
    )



def _gocmen_dogrudan_hucre_ocr_oku(kart):
    """
    HIZLI GOCMEN VALUE-HUCRE OCR'I

    Yalnizca strict tablo geometrisi guvenilir oldugunda
    detector kullanmadan YKN / AD / SOYAD / GECERLILIK
    hucrelerini recognize eder.

    ONEMLI:
    Onceki surumde strict grid kurulamazsa 7 farkli global
    kaydirma deneniyordu. CPU'da bu yol 8-11 saniyeye
    cikabiliyordu.

    Artik:
        strict grid varsa -> hizli recognize
        strict grid yoksa -> burada pahali tarama yapma

    Eksik/yanlis alanlari gocmen_bilgilerini_oku() icindeki
    DAR DEGER BANDI kurtarmasi tamamlar.
    """

    baslangic = time.perf_counter()

    if kart is None:
        return [], 0.0

    geometri_gri = _gocmen_gri(
        kart
    )

    label_haritasi = {}

    deger_x1 = _gocmen_deger_sutunu_projeksiyondan_bul(
        kart,
        geometri_gri,
    )

    if deger_x1 is None:
        return (
            [],
            time.perf_counter() - baslangic,
        )

    dinamik_sinirlar = _gocmen_dinamik_hucre_sinirlari(
        kart,
        deger_x1,
        geometri_gri,
    )

    if dinamik_sinirlar is None:
        return (
            [],
            time.perf_counter() - baslangic,
        )

    bulunanlar = {}

    # YKN + tarih tek batch.
    grup, _ = _gocmen_hucreleri_tani(
        kart,
        [
            "kimlik_no",
            "gecerlilik",
        ],
        _GOCMEN_TARIH_ALLOWLIST,
        label_haritasi,
        deger_x1,
        None,
        geometri_gri,
        dinamik_sinirlar,
    )

    bulunanlar.update(
        grup
    )

    # AD + SOYAD tek batch.
    grup, _ = _gocmen_hucreleri_tani(
        kart,
        [
            "ad",
            "soyad",
        ],
        None,
        label_haritasi,
        deger_x1,
        _GOCMEN_ISIM_BLOCKLIST,
        geometri_gri,
        dinamik_sinirlar,
    )

    bulunanlar.update(
        grup
    )

    ocr = []

    for alan in (
        "kimlik_no",
        "ad",
        "soyad",
        "gecerlilik",
    ):
        item = bulunanlar.get(
            alan
        )

        kutu = (
            [
                item["x1"],
                item["x2"],
                item["y1"],
                item["y2"],
            ]
            if item is not None
            else
            _gocmen_hucre_kutusu(
                kart,
                alan,
                None,
                label_haritasi,
                deger_x1,
                geometri_gri,
                dinamik_sinirlar,
            )
        )

        ocr.append(
            _gocmen_sentetik_label_itemi(
                kart,
                alan,
                kutu,
                deger_x1,
            )
        )

        if item is not None:
            ocr.append(
                {
                    **item,
                    "_dogrudan_hucre_ocr":
                        True,
                }
            )

    # Tarih iki tarihi birlikte vermediyse yalnız bitis
    # hücresini bir kez daha dar şekilde dene.
    tarih_sonucu = gocmen_bilgilerini_bul(
        ocr
    )

    if tarih_sonucu.get(
        "bitis_tarihi"
    ) in {
        None,
        "",
        "Bulunamadi",
    }:
        bitis_item, _ = _gocmen_sag_bitis_hucresini_tani(
            kart,
            deger_x1,
            geometri_gri,
            dinamik_sinirlar,
        )

        if bitis_item is not None:
            ocr.append(
                bitis_item
            )

    return (
        ocr,
        time.perf_counter() - baslangic,
    )



def _gocmen_deger_bandi_detector_oku(kart):
    """
    GOCMEN ICIN DAR-BANT KURTARMA OCR'I

    Belge tipi zaten dogru bulunmusken YKN/AD/SOYAD hucreleri
    geometri nedeniyle bir satir kayarsa tum karti tekrar detector
    ile taramak yerine sadece TABLONUN SAG DEGER SUTUNUNU okur.

    Bu bant:
        - Yabanci Kimlik No
        - Adi
        - Soyadi

    satirlarini kapsar.

    Avantaj:
        - label okumaya ihtiyac yok
        - full detector fallback'ten cok daha ucuz
        - satir sirasi fiziksel olarak YKN -> AD -> SOYAD
    """

    baslangic = time.perf_counter()

    if kart is None:
        return [], 0.0

    h, w = kart.shape[:2]

    # Deger sutununu soldan biraz genis tutuyoruz.
    # Ozellikle YKN'nin ilk 99 hanesinin dikey tablo cizgisine
    # yakin olmasi nedeniyle ilk rakamlarin kirpilmasini engeller.
    x1 = int(w * 0.405)
    x2 = int(w * 0.995)

    # YKN + AD + SOYAD satirlarini genis toleransla kapsa.
    y1 = int(h * 0.385)
    y2 = int(h * 0.555)

    roi = kart[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return [], 0.0

    # Tablo cizgileri bazen detector'u bozuyor.
    # Fakat goruntuyu agresif threshold etmiyoruz; okunabilir net metni koruyoruz.
    try:
        ocr, _ = easyocr_oku(
            roi,
            offset_x=x1,
            offset_y=y1,
        )
    except Exception as e:
        print(
            "Gocmen dar bant detector OCR hatasi:",
            repr(e),
        )
        return [], time.perf_counter() - baslangic

    # Yalniz sag deger sutununda kalan kutulari tut.
    # Detector bazen dikey cizginin solundaki label'in bir parcasini
    # de banda dahil edebilir.
    min_deger_x = int(w * 0.425)

    filtreli = [
        item
        for item in ocr
        if item.get("x2", 0) > min_deger_x
    ]

    return (
        filtreli,
        time.perf_counter() - baslangic,
    )


def _gocmen_no_metni_hafif_duzelt(text):
    """
    YKN hucrelerinde sik gorulen harf/rakam OCR hatalarini
    konservatif bicimde rakama cevirir.
    """

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
            sonuc += donusum[karakter]

    return sonuc


def _gocmen_dar_bant_no_bul(ocr):
    """
    Dar sag sutun OCR'indan dogrulanmis YKN bulur.
    Prefix 99 + 11 hane + mevcut yabanci_no_gecerli_mi kontrolu zorunludur.
    """

    adaylar = []

    for item in ocr:
        ham = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if not ham:
            continue

        # 1) Direkt rakam
        direkt = re.sub(
            r"\D",
            "",
            ham,
        )

        # 2) Harf/rakam karisik OCR duzeltmesi
        duzeltilmis = _gocmen_no_metni_hafif_duzelt(
            ham
        )

        for aday, duzeltme_cezasi in (
            (
                direkt,
                0.0,
            ),
            (
                duzeltilmis,
                0.12,
            ),
        ):
            if (
                len(aday) != 11
                or not aday.startswith("99")
                or not yabanci_no_gecerli_mi(
                    aday
                )
            ):
                continue

            skor = float(
                item.get(
                    "conf",
                    0.0,
                )
            ) - duzeltme_cezasi

            adaylar.append(
                (
                    skor,
                    aday,
                    item,
                )
            )

    if not adaylar:
        return None

    _, no, item = max(
        adaylar,
        key=lambda x: x[0],
    )

    return {
        "deger":
            no,

        "item":
            {
                **item,
                "text":
                    no,
                "norm":
                    no,
                "_dar_bant_kurtarma":
                    True,
                "_fallback_alan":
                    "kimlik_no",
            },
    }


def _gocmen_dar_bant_isim_satirlari(
    ocr,
    ykn_item
):
    """
    YKN'nin ALTINDAKI ilk iki fiziksel metin satirini dondurur.

    Gocmen formunda sira:
        YKN
        AD
        SOYAD

    Label'lar sag deger sutununda olmadigi icin bu yontem,
    label OCR'i bozulsa bile calisir.
    """

    if ykn_item is None:
        return []

    ykn_y = (
        float(
            ykn_item["y1"]
        )
        +
        float(
            ykn_item["y2"]
        )
    ) / 2.0

    uygun = []

    for item in ocr:
        merkez_y = (
            float(
                item["y1"]
            )
            +
            float(
                item["y2"]
            )
        ) / 2.0

        # YKN satirinin altinda olmali.
        if merkez_y <= ykn_y + 6:
            continue

        ham = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        if not ham:
            continue

        # Ad/soyad satirinda rakam istemiyoruz.
        if any(
            c.isdigit()
            for c in ham
        ):
            continue

        temiz = isim_temizle(
            ham
        )

        harfler = re.sub(
            r"[^A-ZÇĞİÖŞÜ]",
            "",
            temiz,
        )

        if len(
            harfler
        ) < 2:
            continue

        norm = normalize_text(
            temiz
        )

        # Sag banda tasan bilinen label/noise kelimeleri.
        if any(
            kelime in norm
            for kelime in (
                "YABANCI",
                "KIMLIK",
                "SOYAD",
                "ADI",
                "BABA",
                "ANNE",
                "DOGUM",
                "MEDENI",
                "UYRUK",
                "GECERLILIK",
            )
        ):
            continue

        uygun.append(
            {
                **item,
                "text":
                    temiz,
                "norm":
                    normalize_text(
                        temiz
                    ),
            }
        )

    # OCR kutularini fiziksel satirlara grupla.
    satirlar = satirlara_grupla(
        uygun
    )

    sonuclar = []

    for satir in satirlar:
        if not satir:
            continue

        satir_y = sum(
            (
                float(
                    x["y1"]
                )
                +
                float(
                    x["y2"]
                )
            ) / 2.0
            for x in satir
        ) / len(
            satir
        )

        if satir_y <= ykn_y + 6:
            continue

        satir = sorted(
            satir,
            key=lambda x: x["x1"],
        )

        parcalar = []

        for item in satir:
            temiz = isim_temizle(
                item.get(
                    "text",
                    "",
                )
            )

            if temiz:
                parcalar.append(
                    temiz
                )

        if not parcalar:
            continue

        text_satir = " ".join(
            parcalar
        )

        conf = sum(
            float(
                x.get(
                    "conf",
                    0.0,
                )
            )
            for x in satir
        ) / len(
            satir
        )

        sonuclar.append(
            {
                "deger":
                    text_satir,

                "item":
                    {
                        "text":
                            text_satir,
                        "norm":
                            normalize_text(
                                text_satir
                            ),
                        "conf":
                            conf,
                        "x1":
                            min(
                                x["x1"]
                                for x in satir
                            ),
                        "y1":
                            min(
                                x["y1"]
                                for x in satir
                            ),
                        "x2":
                            max(
                                x["x2"]
                                for x in satir
                            ),
                        "y2":
                            max(
                                x["y2"]
                                for x in satir
                            ),
                        "_dar_bant_kurtarma":
                            True,
                    },
            }
        )

    sonuclar.sort(
        key=lambda x: (
            x["item"]["y1"]
            +
            x["item"]["y2"]
        ) / 2.0
    )

    # YKN'den sonraki ilk iki FARKLI satir:
    # AD ve SOYAD
    return sonuclar[:2]


def _gocmen_dar_bant_kurtarma(
    kart,
    mevcut_sonuc
):
    """
    Net kartta:
        - belge tipi dogru
        - isimlerden biri okunmus
        - YKN veya diger isim kayip

    durumunu kurtarir.

    Tum belgeyi yeniden okumaz. Yalniz sagdaki YKN+AD+SOYAD
    bandini detector ile bir kez okur ve fiziksel satir sirasini
    kullanir.

    Basariliysa mevcut sonucu guvenli alanlarla gunceller.
    """

    baslangic = time.perf_counter()

    ocr, detector_suresi = _gocmen_deger_bandi_detector_oku(
        kart
    )

    if not ocr:
        return (
            mevcut_sonuc,
            [],
            time.perf_counter() - baslangic,
            [],
        )

    no_sonuc = _gocmen_dar_bant_no_bul(
        ocr
    )

    if no_sonuc is None:
        return (
            mevcut_sonuc,
            ocr,
            time.perf_counter() - baslangic,
            [],
        )

    isim_satirlari = _gocmen_dar_bant_isim_satirlari(
        ocr,
        no_sonuc["item"],
    )

    sonuc = dict(
        mevcut_sonuc
    )

    kullanilan = []

    # YKN kesin dogrulanmissa mevcut eksik/yanlis alani düzelt.
    sonuc["kimlik_no"] = no_sonuc["deger"]
    sonuc["kimlik_no_item"] = no_sonuc["item"]
    sonuc["kimlik_no_conf"] = float(
        no_sonuc["item"].get(
            "conf",
            0.0,
        )
    )
    kullanilan.append(
        "kimlik_no"
    )

    # YKN'nin hemen altindaki 1. satir AD,
    # 2. satir SOYAD.
    if len(
        isim_satirlari
    ) >= 1:
        ad = isim_satirlari[0]

        sonuc["ad"] = ad["deger"]
        sonuc["ad_item"] = {
            **ad["item"],
            "_fallback_alan":
                "ad",
        }
        sonuc["ad_conf"] = float(
            ad["item"].get(
                "conf",
                0.0,
            )
        )
        kullanilan.append(
            "ad"
        )

    if len(
        isim_satirlari
    ) >= 2:
        soyad = isim_satirlari[1]

        # AD ve SOYAD ayni fiziksel satir / ayni metin olamaz.
        if (
            normalize_text(
                soyad["deger"]
            )
            !=
            normalize_text(
                sonuc.get(
                    "ad",
                    "",
                )
            )
        ):
            sonuc["soyad"] = soyad["deger"]
            sonuc["soyad_item"] = {
                **soyad["item"],
                "_fallback_alan":
                    "soyad",
            }
            sonuc["soyad_conf"] = float(
                soyad["item"].get(
                    "conf",
                    0.0,
                )
            )
            kullanilan.append(
                "soyad"
            )

    _gocmen_guveni_guncelle(
        sonuc
    )

    return (
        sonuc,
        ocr,
        time.perf_counter() - baslangic,
        kullanilan,
    )



def _gocmen_no_fallback_itemini_dogrula(item):
    if item is None or float(item.get("conf", 0.0)) < 0.30:
        return None
    rakamlar = re.sub(r"\s+", "", str(item.get("text", "")))
    if re.fullmatch(r"99\d{9}", rakamlar) is None or not yabanci_no_gecerli_mi(rakamlar):
        return None
    return {**item, "text": rakamlar, "norm": rakamlar}


def _gocmen_isim_fallback_itemini_dogrula(item):
    """Sabit hucre guvenini kullanir ama label/noise metnini isim yapmaz."""
    if item is None or float(item.get("conf", 0.0)) < 0.35:
        return None

    ham = str(item.get("text", "")).strip()
    if not ham or any(c.isdigit() for c in ham):
        return None

    gorunen = [c for c in ham if not c.isspace()]
    izinli = [c for c in gorunen if c.isalpha() or c in "-'’"]
    if not gorunen or len(izinli) / len(gorunen) < 0.90:
        return None

    temiz = "".join(
        c if c.isalpha() or c.isspace() or c in "-'’" else " "
        for c in ham.upper()
    )
    temiz = re.sub(r"\s+", " ", temiz).strip()
    harfler = [c for c in temiz if c.isalpha()]
    if len(harfler) < 2 or len(set(harfler)) < 2:
        return None
    if len(temiz) > 80 or len(temiz.split()) > 8:
        return None

    norm = normalize_text(temiz)
    yasaklar = {
        "AD", "ADI", "SOYAD", "SOYADI", "BABA", "BABA ADI", "ANNE",
        "ANNE ADI", "YABANCI KIMLIK NO", "KIMLIK NO", "BASLANGIC",
        "BITIS", "TARIH", "TARIHI", "BELGENIN GECERLILIK TARIHI",
    }
    if norm in yasaklar or any(norm.startswith(f"{x} ") for x in yasaklar):
        return None

    return {**item, "text": temiz, "norm": normalize_text(temiz)}


def _gocmen_tarih_fallback_itemini_dogrula(item):
    if item is None or float(item.get("conf", 0.0)) < 0.25:
        return None, None

    # Parser tek labelsiz tarih satirini kabul eder; ayni zamanda takvim
    # gecerliligini ve baslangic <= bitis sirasini burada yeniden dogrular.
    aday_sonuc = gocmen_bilgilerini_bul([item])
    if aday_sonuc.get("bitis_tarihi") in {None, "", "Bulunamadi"}:
        # Cizgi/kase gurultusu EasyOCR'in ayiraclari atmasina yol acabiliyor:
        # 11/07/2025 - 27/06/2026 -> 1107202527062026. Yalnizca sabit
        # gecerlilik hucresinde ve tam 16 rakam varsa iki DDMMYYYY parcasi
        # varsayilir; takvim ve sira yine parser tarafindan dogrulanir.
        rakamlar = re.sub(r"\D", "", str(item.get("text", "")))
        if len(rakamlar) == 16:
            ilk, ikinci = rakamlar[:8], rakamlar[8:]
            standart = (
                f"{ilk[:2]}/{ilk[2:4]}/{ilk[4:]} - "
                f"{ikinci[:2]}/{ikinci[2:4]}/{ikinci[4:]}"
            )
            item = {**item, "text": standart, "norm": normalize_text(standart)}
            aday_sonuc = gocmen_bilgilerini_bul([item])
    if aday_sonuc.get("bitis_tarihi") in {None, "", "Bulunamadi"}:
        return None, None
    return item, aday_sonuc


def _gocmen_hucre_onislem_varyantlari(roi):
    """Ayni kucuk hucrenin kontrast ve ikili iki hizli varyantini uretir."""
    gri = _gocmen_gri(roi)
    alt, ust = np.percentile(gri, (3.0, 97.0))
    if ust - alt >= 8.0:
        gri = np.clip((gri - alt) * (255.0 / (ust - alt)), 0.0, 255.0)
    else:
        gri = np.clip(gri, 0.0, 255.0)
    kontrast = gri.astype(np.uint8)

    # Hucre siniri veya damga cizgisi rakamlardan/harflerden cok daha uzun olur.
    # Yalniz yuksek kaplamali satir/sutunlari silmek karakter govdelerini korur.
    koyu = kontrast < 112
    satir_cizgisi = np.mean(koyu, axis=1) >= 0.56
    # I/l/1 govdesi de hucre yuksekliginin %70-80'ini kaplayabilir. Yalniz
    # neredeyse kesintisiz (%88+) dikey izleri tablo/damga cizgisi say.
    sutun_cizgisi = np.mean(koyu, axis=0) >= 0.88
    if np.any(satir_cizgisi):
        kontrast[satir_cizgisi, :] = 255
    if np.any(sutun_cizgisi):
        kontrast[:, sutun_cizgisi] = 255

    # Kenarda kalan bir iki cizgi pikselini recognizer'a karakter diye verme.
    kenar = max(1, min(kontrast.shape[:2]) // 30)
    kontrast[:kenar, :] = 255
    kontrast[-kenar:, :] = 255
    kontrast[:, :kenar] = 255
    kontrast[:, -kenar:] = 255

    kontrast_2x = np.repeat(np.repeat(kontrast, 2, axis=0), 2, axis=1)
    # Renkli/eski taramalarda bir varyant gri tonunu, digeri arka plan dokusunu
    # tamamen atan ikili goruntuyu temsil eder. Ikisi tek recognize batch'indedir.
    esik = float(np.clip(np.percentile(kontrast, 38.0), 105.0, 205.0))
    ikili = np.where(kontrast <= esik, 0, 255).astype(np.uint8)
    ikili_2x = np.repeat(np.repeat(ikili, 2, axis=0), 2, axis=1)
    return [("kontrast", kontrast_2x), ("ikili", ikili_2x)]


def _gocmen_onislemli_hucre_adaylari_tani(
    kart, alanlar, allowlist, label_haritasi, deger_x1, blocklist=None,
    geometri_gri=None,
):
    """Yalniz ilk hucre okumasi gecersiz kalan alanlari iki varyantta tanir."""
    if not alanlar:
        return {}, 0.0

    baslangic = time.perf_counter()
    parcalar = []
    kutular = []
    eslemeler = []
    canvas_y = 0
    canvas_w = 0
    ayirici = 12

    for alan in alanlar:
        kutu = _gocmen_hucre_kutusu(
            kart, alan, label_haritasi.get(alan), label_haritasi, deger_x1,
            geometri_gri,
        )
        x1, x2, y1, y2 = kutu
        roi = kart[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        for varyant_adi, varyant in _gocmen_hucre_onislem_varyantlari(roi):
            vh, vw = varyant.shape[:2]
            parcalar.append((canvas_y, varyant))
            kutular.append([0, vw, canvas_y, canvas_y + vh])
            eslemeler.append((alan, varyant_adi, kutu))
            canvas_y += vh + ayirici
            canvas_w = max(canvas_w, vw)

    if not parcalar:
        return {}, time.perf_counter() - baslangic

    canvas_h = max(1, canvas_y - ayirici)
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    for py, parca in parcalar:
        ph, pw = parca.shape[:2]
        canvas[py:py + ph, :pw] = parca

    try:
        sonuclar = reader_getir().recognize(
            canvas, horizontal_list=kutular, free_list=[],
            decoder="greedy", beamWidth=5,
            batch_size=max(1, min(4, len(kutular))), workers=0,
            allowlist=allowlist, blocklist=blocklist, detail=1, rotation_info=None,
            # Bu goruntuler zaten kontrast/ikili varyantlardir; EasyOCR'in
            # dusuk guvende ayni kutuyu dahili olarak ikinci kez calistirmasi
            # gereksizdir.
            paragraph=False, contrast_ths=0.0, adjust_contrast=0.5,
            filter_ths=0.003,
        )
    except Exception as e:
        print("Gocmen onislemli hucre OCR hatasi:", repr(e))
        return {}, time.perf_counter() - baslangic

    adaylar = {alan: [] for alan in alanlar}
    for (alan, varyant_adi, kutu), sonuc in zip(eslemeler, sonuclar):
        try:
            _, text_ocr, conf = sonuc
            text_ocr = str(text_ocr).strip()
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if not text_ocr:
            continue
        x1, x2, y1, y2 = kutu
        adaylar[alan].append({
            "text": text_ocr,
            "norm": normalize_text(text_ocr),
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "_fallback_alan": alan,
            "_onislemli": True,
            "_varyant": varyant_adi,
        })
    return adaylar, time.perf_counter() - baslangic


def _gocmen_konsensus_anahtari(text):
    """Aksan farkini yok sayan, yalniz aday gruplamada kullanilan anahtar."""
    ayrismis = unicodedata.normalize("NFKD", str(text or "").upper())
    return "".join(
        c for c in ayrismis
        if not unicodedata.combining(c) and c.isalnum()
    )


def _gocmen_dogrulanmis_aday_sec(
    adaylar, dogrulayici, celiskili_tekil_adaylari_reddet=False,
):
    """Yalniz alan dogrulamasini gecen adayi, varyant konsensusuyle secer.

    YKN gibi yanlis pozitif maliyeti yuksek alanlarda iki goruntu varyanti iki
    farkli ama checksum-gecerli deger uretebilir. Bu durumda tekil tahminlerin
    confidence farkina guvenilmez; ancak bir deger birden cok varyantta tekrar
    ediyorsa konsensusla secilir.
    """
    gruplar = {}
    for ham in adaylar or []:
        dogru = dogrulayici(ham)
        if dogru is None:
            continue
        anahtar = _gocmen_konsensus_anahtari(dogru.get("text", ""))
        if not anahtar:
            continue
        gruplar.setdefault(anahtar, []).append(dogru)
    if not gruplar:
        return None

    def grup_skoru(grup):
        varyant_sayisi = len({x.get("_varyant", id(x)) for x in grup})
        en_yuksek_guven = max(float(x.get("conf", 0.0)) for x in grup)
        # Konsensus, tek bir varyantin yuksek confidence tahmininden daima once
        # gelir. Checksum/geometri dogrulamasi yine zorunludur; numara uretilmez.
        return varyant_sayisi, en_yuksek_guven

    sirali_gruplar = sorted(gruplar.values(), key=grup_skoru, reverse=True)
    en_iyi_grup = sirali_gruplar[0]
    if celiskili_tekil_adaylari_reddet and len(sirali_gruplar) > 1:
        en_iyi_konsensus = grup_skoru(en_iyi_grup)[0]
        ikinci_konsensus = grup_skoru(sirali_gruplar[1])[0]
        if en_iyi_konsensus <= 1 or en_iyi_konsensus == ikinci_konsensus:
            return None
    en_yuksek_guven = max(float(x.get("conf", 0.0)) for x in en_iyi_grup)
    yakinlar = [
        x for x in en_iyi_grup
        if float(x.get("conf", 0.0)) >= en_yuksek_guven - 0.08
    ]

    def unicode_ayrinti_sayisi(item):
        text = str(item.get("text", ""))
        ayrisma = unicodedata.normalize("NFD", text)
        aksan = sum(unicodedata.combining(c) != 0 for c in ayrisma)
        noktalama = sum(c in "-'’" for c in text)
        return aksan + noktalama

    # Ayni adin aksanli ve aksansiz iki okumasinda guvenler yakinsa belgede
    # gorulen zengin Unicode yazimini koru (ÉLODIE/GARCÍA gibi).
    return max(
        yakinlar,
        key=lambda x: (unicode_ayrinti_sayisi(x), float(x.get("conf", 0.0))),
    )


def _gocmen_onislemli_tarih_hucresi_tani(
    kart, label_item=None, label_haritasi=None, deger_x1=None,
    geometri_gri=None,
):
    """Geriye uyumlu tarih yardimcisi; iki ortak varyanttan gecerli olani secer."""
    label_haritasi = dict(label_haritasi or {})
    if label_item is not None:
        label_haritasi["gecerlilik"] = label_item
    adaylar, sure = _gocmen_onislemli_hucre_adaylari_tani(
        kart, ["gecerlilik"], _GOCMEN_TARIH_ALLOWLIST,
        label_haritasi, deger_x1, None, geometri_gri,
    )
    item = _gocmen_dogrulanmis_aday_sec(
        adaylar.get("gecerlilik", []),
        lambda x: _gocmen_tarih_fallback_itemini_dogrula(x)[0],
        celiskili_tekil_adaylari_reddet=True,
    )
    return item, sure


def _gocmen_guveni_guncelle(sonuc):
    bulunan = sum(
        sonuc.get(alan) not in {None, "", "Bulunamadi"}
        for alan in ("kimlik_no", "ad", "soyad")
    )
    if bulunan == 3:
        sonuc["guven"] = (
            "yuksek"
            if min(
                float(sonuc.get("kimlik_no_conf", 0.0)),
                float(sonuc.get("ad_conf", 0.0)),
                float(sonuc.get("soyad_conf", 0.0)),
            ) >= 0.50
            else "orta"
        )
    elif bulunan > 0:
        sonuc["guven"] = "orta"
    else:
        sonuc["guven"] = "dusuk"


def _gocmen_isim_fallbacki_karsi_alanla_celisiyor(yeniden, alan, item):
    """Yanlis satirdan okunan adi soyad (veya tersini) diye zorla yazmayi onler."""
    diger_alan = "soyad" if alan == "ad" else "ad"
    aday_norm = normalize_text(item.get("text", ""))
    if not aday_norm:
        return True

    diger_item = yeniden.get(f"{diger_alan}_item")
    if diger_item is not None:
        aday_merkez = (float(item["y1"]) + float(item["y2"])) / 2.0
        diger_merkez = (float(diger_item["y1"]) + float(diger_item["y2"])) / 2.0
        aday_h = max(1.0, float(item["y2"]) - float(item["y1"]))
        diger_h = max(1.0, float(diger_item["y2"]) - float(diger_item["y1"]))
        ayni_satir_toleransi = max(12.0, (aday_h + diger_h) * 0.35)
        # Parser fallback metnini hedef alana da eslemis olsa bile fiziksel kutu
        # zaten diger alanin satirindaysa bu bir ad<->soyad kopyasidir.
        if abs(aday_merkez - diger_merkez) <= ayni_satir_toleransi:
            return True

    if normalize_text(yeniden.get(alan, "")) == aday_norm:
        return False
    if normalize_text(yeniden.get(diger_alan, "")) != aday_norm:
        return False
    return diger_item is None


def _gocmen_eksik_hucre_fallback(
    kart, ocr, ilk_sonuc, atlanacak_alanlar=None,
):
    """Yalniz eksik oncelikli alanlari fixed-cell recognizer ile tamamlar."""
    atlanacak_alanlar = set(atlanacak_alanlar or ())
    eksik_no = (
        "kimlik_no" not in atlanacak_alanlar
        and ilk_sonuc.get("kimlik_no") in {None, "", "Bulunamadi"}
    )
    eksik_isimler = [
        alan for alan in ("ad", "soyad")
        if alan not in atlanacak_alanlar
        and ilk_sonuc.get(alan) in {None, "", "Bulunamadi"}
    ]
    eksik_tarih = (
        "gecerlilik" not in atlanacak_alanlar
        and ilk_sonuc.get("bitis_tarihi") in {None, "", "Bulunamadi"}
    )
    denenler = (["kimlik_no"] if eksik_no else []) + eksik_isimler + (["gecerlilik"] if eksik_tarih else [])
    if not denenler:
        return ilk_sonuc, ocr, 0.0, [], []

    dogrudan_ilk_gecis = any(
        bool(item.get("_dogrudan_hucre_ocr")) for item in ocr
    )
    label_haritasi = _gocmen_label_haritasi(ocr)
    # Tum fallback boyunca ayni gri geometriyi yeniden kullan; projeksiyon icin
    # karti her hucrede tekrar float diziye donusturme.
    geometri_gri = _gocmen_gri(kart)
    deger_x1 = _gocmen_deger_sutunu_sol_siniri(
        kart, ocr, ilk_sonuc, label_haritasi, geometri_gri,
    )
    ham_itemlar = {}
    toplam_sure = 0.0
    if eksik_no and not dogrudan_ilk_gecis:
        bulunan, sure = _gocmen_hucreleri_tani(
            kart, ["kimlik_no"], _GOCMEN_NO_ALLOWLIST, label_haritasi, deger_x1,
            None, geometri_gri,
        )
        ham_itemlar.update(bulunan)
        toplam_sure += sure
    if eksik_isimler and not dogrudan_ilk_gecis:
        # Ad ve soyad ayni karakter kumesini kullanir; ikisi eksikse tek
        # recognize API cagrisi yeterlidir (CPU'da kutular yine sirali islenir).
        bulunan, sure = _gocmen_hucreleri_tani(
            kart, eksik_isimler, None, label_haritasi, deger_x1,
            _GOCMEN_ISIM_BLOCKLIST, geometri_gri,
        )
        ham_itemlar.update(bulunan)
        toplam_sure += sure
    if eksik_tarih and not dogrudan_ilk_gecis:
        bulunan, sure = _gocmen_hucreleri_tani(
            kart, ["gecerlilik"], _GOCMEN_TARIH_ALLOWLIST, label_haritasi, deger_x1,
            None, geometri_gri,
        )
        ham_itemlar.update(bulunan)
        toplam_sure += sure

    kabul_edilen = {}
    if eksik_no:
        item = _gocmen_no_fallback_itemini_dogrula(ham_itemlar.get("kimlik_no"))
        if item is not None:
            kabul_edilen["kimlik_no"] = item
    for alan in eksik_isimler:
        item = _gocmen_isim_fallback_itemini_dogrula(ham_itemlar.get(alan))
        if item is not None:
            kabul_edilen[alan] = item

    # Detector'siz hizli yolda ham hucre zaten bir kez okundu. Gecersiz kalan
    # alani ayni ham goruntuyle tekrarlamadan dogrudan kontrollu guclu varyanta
    # gec; eski/harici OCR girdisinde ise yukaridaki tek ham hucre denemesi kalir.
    if eksik_no and "kimlik_no" not in kabul_edilen:
        adaylar, sure = _gocmen_onislemli_hucre_adaylari_tani(
            kart, ["kimlik_no"], _GOCMEN_NO_ALLOWLIST,
            label_haritasi, deger_x1, None, geometri_gri,
        )
        toplam_sure += sure
        item = _gocmen_dogrulanmis_aday_sec(
            adaylar.get("kimlik_no", []), _gocmen_no_fallback_itemini_dogrula,
            celiskili_tekil_adaylari_reddet=True,
        )
        if item is not None:
            kabul_edilen["kimlik_no"] = item

    onislemli_isimler = [alan for alan in eksik_isimler if alan not in kabul_edilen]
    if onislemli_isimler:
        adaylar, sure = _gocmen_onislemli_hucre_adaylari_tani(
            kart, onislemli_isimler, None,
            label_haritasi, deger_x1, _GOCMEN_ISIM_BLOCKLIST, geometri_gri,
        )
        toplam_sure += sure
        for alan in onislemli_isimler:
            item = _gocmen_dogrulanmis_aday_sec(
                adaylar.get(alan, []), _gocmen_isim_fallback_itemini_dogrula,
            )
            if item is not None:
                kabul_edilen[alan] = item

    tarih_sonuc = None
    if eksik_tarih:
        item, tarih_sonuc = _gocmen_tarih_fallback_itemini_dogrula(
            ham_itemlar.get("gecerlilik"),
        )
        if item is None:
            # Ilk sabit/dinamik hucre okumasinda tablo cizgisi veya dusuk
            # kontrast tarihi bozduysa sadece bu hucreye daha guclu onislem uygula.
            onislemli, sure = _gocmen_onislemli_tarih_hucresi_tani(
                kart, label_haritasi.get("gecerlilik"), label_haritasi, deger_x1,
                geometri_gri,
            )
            toplam_sure += sure
            item, tarih_sonuc = _gocmen_tarih_fallback_itemini_dogrula(onislemli)
        if item is not None:
            kabul_edilen["gecerlilik"] = item

    if not kabul_edilen:
        return ilk_sonuc, ocr, toplam_sure, denenler, []

    zengin_ocr = [*ocr, *kabul_edilen.values()]
    yeniden = gocmen_bilgilerini_bul(zengin_ocr)
    sonuc = dict(ilk_sonuc)
    kullanilanlar = []

    if "kimlik_no" in kabul_edilen:
        item = kabul_edilen["kimlik_no"]
        sonuc["kimlik_no"] = item["text"]
        sonuc["kimlik_no_conf"] = float(item["conf"])
        sonuc["kimlik_no_item"] = item
        kullanilanlar.append("kimlik_no")

    for alan in ("ad", "soyad"):
        if alan not in kabul_edilen:
            continue
        item = kabul_edilen[alan]
        if _gocmen_isim_fallbacki_karsi_alanla_celisiyor(yeniden, alan, item):
            continue
        # Parser etiketi de gormusse onun birlestirdigi kutuyu kullan; label ana
        # detector gecisinde kacmissa sabit hucrenin konservatif sonucu kalir.
        parser_degeri = yeniden.get(alan)
        if (
            parser_degeri not in {None, "", "Bulunamadi"}
            and normalize_text(parser_degeri) == normalize_text(item["text"])
        ):
            sonuc[alan] = parser_degeri
            sonuc[f"{alan}_conf"] = float(yeniden.get(f"{alan}_conf", item["conf"]))
            sonuc[f"{alan}_item"] = yeniden.get(f"{alan}_item") or item
            sonuc[f"{alan}_label"] = yeniden.get(f"{alan}_label")
        else:
            sonuc[alan] = item["text"]
            sonuc[f"{alan}_conf"] = float(item["conf"])
            sonuc[f"{alan}_item"] = item
        kullanilanlar.append(alan)

    if "gecerlilik" in kabul_edilen and tarih_sonuc is not None:
        for alan in (
            "baslangic_tarihi", "bitis_tarihi", "belge_gecerli",
            "gecerlilik_durumu", "gecerlilik_items",
        ):
            sonuc[alan] = tarih_sonuc[alan]
        kullanilanlar.append("gecerlilik")

    _gocmen_guveni_guncelle(sonuc)
    return sonuc, zengin_ocr, toplam_sure, denenler, kullanilanlar


# =========================================================
# TÜRKÇE SECOND PASS
# =========================================================

def turkce_second_pass_gerekli_mi(text):
    if not text:
        return False
    norm = normalize_text(text)
    return any(harf in norm for harf in ["S", "I", "U", "O", "C", "G"])


def isim_kutusunu_hafif_genislet(kart, item):
    if item is None:
        return None

    h, w = kart.shape[:2]
    x1, y1 = max(0, item["x1"] - 3), max(0, item["y1"] - 6)
    x2, y2 = min(w, item["x2"] + 3), min(h, item["y2"] + 4)

    roi = kart[y1:y2, x1:x2]
    return roi if roi.size else None


def turkce_harf_iyilestir(kart, item):
    if item is None:
        return None, 0.0, 0.0

    ilk_text = isim_temizle(item.get("text", ""))
    ilk_conf = float(item.get("conf", 0.0))

    if not ilk_text:
        return None, ilk_conf, 0.0
    if not turkce_second_pass_gerekli_mi(ilk_text):
        return ilk_text, ilk_conf, 0.0

    baslangic = time.perf_counter()
    roi = isim_kutusunu_hafif_genislet(kart, item)
    if roi is None:
        return ilk_text, ilk_conf, 0.0

    # Küçük value crop'ında recognition için yeterli.
    roi = cv2.resize(roi, None, fx=1.65, fy=1.65, interpolation=cv2.INTER_LINEAR)

    try:
        # HIZ KRİTİK: kutu zaten belli olduğu için readtext() ile tekrar text
        # detector çalıştırmıyoruz. reader.recognize() yalnızca verilen
        # yatay kutuyu recognize eder — çok daha hızlı.
        roi_h, roi_w = roi.shape[:2]
        sonuclar = reader_getir().recognize(
            roi, horizontal_list=[[0, roi_w, 0, roi_h]], free_list=[],
            decoder="greedy", beamWidth=5, batch_size=1, workers=0,
            allowlist=None, blocklist=None, detail=1, rotation_info=None,
            paragraph=False, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003,
        )
    except Exception:
        try:
            sonuclar = reader_getir().readtext(roi, detail=1, paragraph=False, decoder="greedy")
        except Exception:
            return ilk_text, ilk_conf, time.perf_counter() - baslangic

    sure = time.perf_counter() - baslangic
    if not sonuclar:
        return ilk_text, ilk_conf, sure

    parcalar = []
    for sonuc in sonuclar:
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

    # Ana OCR ile aynı normalize edilmiş kelimeyse, Türkçe karakter açısından
    # daha iyi olanı seç.
    if normalize_text(ikinci_text) == normalize_text(ilk_text):
        ilk_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ilk_text))
        ikinci_tr = len(re.findall(r"[ÇĞİÖŞÜ]", ikinci_text))

        if ikinci_tr > ilk_tr:
            return ikinci_text, ikinci_conf, sure
        # Aynı yazım ailesindeyse confidence ciddi yüksekse ikinci sonucu da kullanabiliriz.
        if ikinci_conf > ilk_conf + 0.12:
            return ikinci_text, ikinci_conf, sure

    return ilk_text, ilk_conf, sure


# =========================================================
# YENİ TC
# =========================================================

SOYAD_LABEL_HEDEFLERI = ["SOYADI SURNAME", "SURNAME", "SOYADI"]
AD_LABEL_HEDEFLERI = [
    "ADI GIVEN NAME S", "ADI GIVEN NAMES", "ADI GIVEN NAME",
    "GIVEN NAME S", "GIVEN NAMES", "GIVEN NAME",
]


def fuzzy_label_bul(ocr_sonuclari, hedefler, esik=0.40):
    en_iyi_item, en_iyi_skor = None, 0.0
    for item in ocr_sonuclari:
        for hedef in hedefler:
            skor = benzerlik(item["text"], hedef)
            if skor > en_iyi_skor:
                en_iyi_skor, en_iyi_item = skor, item
    return en_iyi_item if en_iyi_skor >= esik else None


def buyuk_harf_orani(text):
    harfler = [c for c in str(text) if c.isalpha()]
    if not harfler:
        return 0.0
    return sum(1 for c in harfler if c.isupper()) / len(harfler)


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


def label_altindaki_degeri_bul(ocr_sonuclari, label, sonraki_label=None):
    if label is None:
        return None

    adaylar = []
    for item in ocr_sonuclari:
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
# TC OKU
# =========================================================

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
        "guven": guven, "kimlik_no_conf": kimlik_no_conf,
        "ad_conf": ad_conf, "soyad_conf": soyad_conf, "belge_gecerli": None,
        "gecerlilik_durumu": None, "baslangic_tarihi": "", "bitis_tarihi": "",
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [], "ocr_suresi": sure,
    }


# =========================================================
# ESKİ TC
# =========================================================

def _alan_bulundu(sonuc, alan):
    deger = sonuc.get(alan, "Bulunamadi")
    return bool(deger) and deger != "Bulunamadi"


def _item_merkez_y(item):
    if item is None:
        return None
    return (float(item["y1"]) + float(item["y2"])) / 2.0


def _item_yukseklik(item):
    if item is None:
        return 1.0
    return max(1.0, float(item["y2"]) - float(item["y1"]))


def _eski_tc_isim_geometrisi_uygun(sonuc, alan, aday_item):
    """Farklı OCR geçişlerinden gelen ad/soyad aynı fiziksel satıra düşmesin."""
    if aday_item is None:
        return False

    diger_alan = "ad" if alan == "soyad" else "soyad"
    diger_item = sonuc.get(f"{diger_alan}_item")
    if diger_item is None:
        return True

    aday_y = _item_merkez_y(aday_item)
    diger_y = _item_merkez_y(diger_item)
    tolerans = max(20.0, (_item_yukseklik(aday_item) + _item_yukseklik(diger_item)) * 0.325)

    if abs(aday_y - diger_y) <= tolerans:
        return False
    if alan == "soyad":
        return aday_y < diger_y
    return aday_y > diger_y


def _eski_tc_sonuclarini_birlestir(ilk_sonuc, fallback_sonuc):
    """İlk OCR'daki doğru alanları korur, yalnız eksikleri fallback'ten doldurur.

    Önceki winner-takes-all seçim aynı sayıda alan bulunduğunda doğru TC'yi veya
    doğru bir ismi kaybedebiliyordu. Alan bazlı ve konservatif birleştirme, hızlı
    OCR'ın mevcut sonuçlarını değiştirmeden geniş OCR'ın yalnız eksikleri
    tamamlamasını sağlar.
    """
    sonuc = dict(ilk_sonuc)
    kullanilan_fallback = False

    if not _alan_bulundu(sonuc, "tc_no") and _alan_bulundu(fallback_sonuc, "tc_no"):
        sonuc["tc_no"] = fallback_sonuc["tc_no"]
        sonuc["tc_item"] = fallback_sonuc.get("tc_item")
        sonuc["tc_label"] = fallback_sonuc.get("tc_label")
        kullanilan_fallback = True

    for alan in ("soyad", "ad"):
        if _alan_bulundu(sonuc, alan) or not _alan_bulundu(fallback_sonuc, alan):
            continue

        aday_item = fallback_sonuc.get(f"{alan}_item")
        if not _eski_tc_isim_geometrisi_uygun(sonuc, alan, aday_item):
            continue

        sonuc[alan] = fallback_sonuc[alan]
        sonuc[f"{alan}_conf"] = float(fallback_sonuc.get(f"{alan}_conf", 0.0))
        sonuc[f"{alan}_item"] = aday_item
        sonuc[f"{alan}_label"] = fallback_sonuc.get(f"{alan}_label")
        kullanilan_fallback = True

    bulunan = sum(_alan_bulundu(sonuc, alan) for alan in ("tc_no", "ad", "soyad"))
    if bulunan == 3:
        sonuc["guven"] = (
            "yuksek"
            if float(sonuc.get("ad_conf", 0.0)) >= 0.50 and float(sonuc.get("soyad_conf", 0.0)) >= 0.50
            else "orta"
        )
    elif bulunan > 0:
        sonuc["guven"] = "orta"
    else:
        sonuc["guven"] = "dusuk"

    return sonuc, kullanilan_fallback

def eski_tc_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    # ---- 1. Hızlı OCR ---------------------------------------------------
    ocr, ilk_ocr_suresi = easyocr_oku_eski_tc(kart)
    sonuc = eski_tc_bilgilerini_bul(ocr)

    # ---- 2. Gerçek parser sonucuna göre fallback (hepsi bulunduysa asla çalışmaz) ----
    ilk_tam = (
        sonuc.get("tc_no", "Bulunamadi") != "Bulunamadi"
        and sonuc.get("ad", "Bulunamadi") != "Bulunamadi"
        and sonuc.get("soyad", "Bulunamadi") != "Bulunamadi"
    )
    fallback_ocr_suresi = 0.0

    if not ilk_tam:
        ocr_fallback, fallback_ocr_suresi = easyocr_oku_eski_tc_fallback(kart)

        if ocr_fallback:
            fallback_sonuc = eski_tc_bilgilerini_bul(ocr_fallback)
            sonuc, fallback_kullanildi = _eski_tc_sonuclarini_birlestir(sonuc, fallback_sonuc)
            if fallback_kullanildi:
                # Kutular aynı kart koordinat sisteminde; debug görünümünde iki
                # geçişteki kanıtları da gösterebilmek için birlikte sakla.
                ocr = [*ocr, *ocr_fallback]

    # ---- 3. Second pass — sadece düşük conf + şüpheli harf --------------
    ad_item, soyad_item = sonuc.get("ad_item"), sonuc.get("soyad_item")
    ad_ham, soyad_ham = sonuc.get("ad", "Bulunamadi"), sonuc.get("soyad", "Bulunamadi")
    ad_ham_conf = float(sonuc.get("ad_conf", 0.0))
    soyad_ham_conf = float(sonuc.get("soyad_conf", 0.0))

    if ad_item is not None and ad_ham != "Bulunamadi" and ad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(ad_ham):
        ad_second, ad_second_conf, ad_second_sure = turkce_harf_iyilestir(kart, ad_item)
    else:
        ad_second, ad_second_conf, ad_second_sure = ad_ham, ad_ham_conf, 0.0

    if (
        soyad_item is not None and soyad_ham != "Bulunamadi"
        and soyad_ham_conf < 0.58 and turkce_second_pass_gerekli_mi(soyad_ham)
    ):
        soyad_second, soyad_second_conf, soyad_second_sure = turkce_harf_iyilestir(kart, soyad_item)
    else:
        soyad_second, soyad_second_conf, soyad_second_sure = soyad_ham, soyad_ham_conf, 0.0

    second_pass_suresi = ad_second_sure + soyad_second_sure

    # ---- 4. Son değerler --------------------------------------------------
    ad = ad_second if ad_second else ad_ham
    soyad = soyad_second if soyad_second else soyad_ham
    ad_conf = float(ad_second_conf if ad_second else ad_ham_conf)
    soyad_conf = float(soyad_second_conf if soyad_second else soyad_ham_conf)
    tc_no = sonuc.get("tc_no", "Bulunamadi")
    kimlik_no_conf = float(sonuc.get("tc_item", {}).get("conf", 0.0)) if sonuc.get("tc_item") else 0.0

    bulunan = sum([tc_no != "Bulunamadi", ad != "Bulunamadi", soyad != "Bulunamadi"])
    if bulunan == 3:
        guven = "yuksek" if (ad_conf >= 0.50 and soyad_conf >= 0.50) else "orta"
    elif bulunan > 0:
        guven = "orta"
    else:
        guven = "dusuk"

    # ---- 5. Debug -----------------------------------------------------
    debug_resmi = None
    if debug:
        debug_resmi = debug_resmi_olustur(kart, [
            (sonuc.get("tc_item"), "TC", (0, 255, 0)),
            (sonuc.get("soyad_item"), "SOYAD", (0, 255, 255)),
            (sonuc.get("ad_item"), "AD", (255, 255, 0)),
        ])

    ana_ocr_suresi = ilk_ocr_suresi + fallback_ocr_suresi

    return {
        "sayfa_no": sayfa_no, "belge_tipi": "eski_tc", "tc_no": tc_no, "ad": ad, "soyad": soyad,
        "guven": guven, "kimlik_no_conf": kimlik_no_conf, "ad_conf": ad_conf, "soyad_conf": soyad_conf,
        "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None, "gecerlilik_durumu": None,
        "ocr_suresi": ana_ocr_suresi + second_pass_suresi, "ana_ocr_suresi": ana_ocr_suresi,
        "ilk_ocr_suresi": ilk_ocr_suresi, "fallback_ocr_suresi": fallback_ocr_suresi,
        "second_pass_suresi": second_pass_suresi,
        "debug_resmi": debug_resmi, "tum_ocr": ocr if debug else [],
    }


# =========================================================
# GÖÇMEN
# =========================================================

def _gocmen_hizli_alan_guvenli_mi(sonuc, alan):
    if alan == "kimlik_no":
        kimlik_no = sonuc.get("kimlik_no")
        if kimlik_no in {None, "", "Bulunamadi"}:
            return False
        item = sonuc.get("kimlik_no_item") or {}
        # Dinamik grid ile kanitlanmis YKN hucresindeki tam 11 haneli deger,
        # EasyOCR confidence'i 0.39 gibi sinirin hemen altinda olsa da checksum
        # geciyorsa rastgele bir sayi degildir. Burada confidence yuzunden dogru
        # sonucu silip pahali detector'a dusmeyiz; prefix+MERNIS checksum zorunlu.
        if (
            item.get("_dogrudan_hucre_ocr")
            and item.get("_fallback_alan") == "kimlik_no"
            and yabanci_no_gecerli_mi(str(kimlik_no))
        ):
            return True
        return float(sonuc.get("kimlik_no_conf", 0.0)) >= 0.40
    if alan in {"ad", "soyad"}:
        return (
            sonuc.get(alan) not in {None, "", "Bulunamadi"}
            and float(sonuc.get(f"{alan}_conf", 0.0)) >= 0.55
        )
    if alan == "gecerlilik":
        if sonuc.get("bitis_tarihi") in {None, "", "Bulunamadi"}:
            return False
        itemlar = sonuc.get("gecerlilik_items") or []
        return bool(itemlar) and min(
            float(item.get("conf", 0.0)) for item in itemlar
        ) >= 0.35
    return False


def _gocmen_hizli_sonuc_guvenli_mi(sonuc):
    return all(
        _gocmen_hizli_alan_guvenli_mi(sonuc, alan)
        for alan in ("kimlik_no", "ad", "soyad", "gecerlilik")
    )


def _gocmen_detector_eksiklerini_hizli_sonuctan_tamamla(sonuc, hizli_sonuc):
    """Detector sonucu eksikse yalniz guvenli direct alanlari geri kazandirir."""
    sonuc = dict(sonuc)
    kullanilanlar = []

    if (
        sonuc.get("kimlik_no") in {None, "", "Bulunamadi"}
        and _gocmen_hizli_alan_guvenli_mi(hizli_sonuc, "kimlik_no")
    ):
        for anahtar in ("kimlik_no", "kimlik_no_conf", "kimlik_no_item"):
            sonuc[anahtar] = hizli_sonuc.get(anahtar)
        kullanilanlar.append("kimlik_no")

    for alan in ("ad", "soyad"):
        if (
            sonuc.get(alan) in {None, "", "Bulunamadi"}
            and _gocmen_hizli_alan_guvenli_mi(hizli_sonuc, alan)
        ):
            for anahtar in (
                alan, f"{alan}_conf", f"{alan}_item", f"{alan}_label",
            ):
                sonuc[anahtar] = hizli_sonuc.get(anahtar)
            kullanilanlar.append(alan)

    if (
        sonuc.get("bitis_tarihi") in {None, "", "Bulunamadi"}
        and _gocmen_hizli_alan_guvenli_mi(hizli_sonuc, "gecerlilik")
    ):
        for anahtar in (
            "baslangic_tarihi", "bitis_tarihi", "belge_gecerli",
            "gecerlilik_durumu", "gecerlilik_items",
        ):
            sonuc[anahtar] = hizli_sonuc.get(anahtar)
        kullanilanlar.append("gecerlilik")

    _gocmen_guveni_guncelle(sonuc)
    return sonuc, kullanilanlar


def gocmen_bilgilerini_oku(kart, sayfa_no=None, debug=False):
    # =====================================================
    # 1. STRICT / HIZLI VALUE HUCRELERI
    # =====================================================

    hizli_ocr, hizli_ana_suresi = easyocr_oku_gocmen(
        kart
    )

    hizli_sonuc = gocmen_bilgilerini_bul(
        hizli_ocr
    )

    hizli_fallback_suresi = 0.0
    hizli_denenen = []
    hizli_kullanilan = []

    if hizli_ocr:
        (
            hizli_sonuc,
            hizli_ocr,
            hizli_fallback_suresi,
            hizli_denenen,
            hizli_kullanilan,
        ) = _gocmen_eksik_hucre_fallback(
            kart,
            hizli_ocr,
            hizli_sonuc,
        )

    # =====================================================
    # 2. DAR DEGER BANDI KURTARMA
    #
    # Net kartta belge tipi dogru ama YKN/AD/SOYAD'dan biri
    # eksikse, tum kart detector yerine yalniz tablonun sag
    # sutunundaki 3 satiri detect et.
    #
    # Ayrica tek isim bulunmus fakat yanlis alana atanmis
    # vakada da YKN -> AD -> SOYAD fiziksel sirasi yeniden
    # kurulur.
    # =====================================================

    kritik_eksik = (
        hizli_sonuc.get(
            "kimlik_no"
        ) in {
            None,
            "",
            "Bulunamadi",
        }
        or
        hizli_sonuc.get(
            "ad"
        ) in {
            None,
            "",
            "Bulunamadi",
        }
        or
        hizli_sonuc.get(
            "soyad"
        ) in {
            None,
            "",
            "Bulunamadi",
        }
        or (
            hizli_sonuc.get(
                "ad"
            ) not in {
                None,
                "",
                "Bulunamadi",
            }
            and
            hizli_sonuc.get(
                "soyad"
            ) not in {
                None,
                "",
                "Bulunamadi",
            }
            and
            normalize_text(
                hizli_sonuc.get(
                    "ad"
                )
            )
            ==
            normalize_text(
                hizli_sonuc.get(
                    "soyad"
                )
            )
        )
    )

    dar_bant_suresi = 0.0
    dar_bant_kullanilan = []
    dar_bant_ocr = []

    if kritik_eksik:
        (
            dar_sonuc,
            dar_bant_ocr,
            dar_bant_suresi,
            dar_bant_kullanilan,
        ) = _gocmen_dar_bant_kurtarma(
            kart,
            hizli_sonuc,
        )

        # Dar bant en az YKN'yi dogrulamissa onu kullan.
        if "kimlik_no" in dar_bant_kullanilan:
            hizli_sonuc = dar_sonuc

            # Debug/fallback tarafinda yeni bulunan kutular da gorunsun.
            hizli_ocr = [
                *hizli_ocr,
                *dar_bant_ocr,
            ]

            hizli_kullanilan = list(
                dict.fromkeys(
                    [
                        *hizli_kullanilan,
                        *dar_bant_kullanilan,
                    ]
                )
            )

    # =====================================================
    # 3. HIZLI YOL YETERLI MI?
    # =====================================================

    dogrudan_isaretli = any(
        bool(
            item.get(
                "_dogrudan_hucre_ocr"
            )
        )
        for item in hizli_ocr
    )

    dar_bant_isaretli = bool(
        dar_bant_kullanilan
    )

    legacy_detector_girdisi = (
        bool(
            hizli_ocr
        )
        and
        not dogrudan_isaretli
        and
        not dar_bant_isaretli
    )

    # Dar bantta dogrulanmis YKN elde ettiysek belge zaten dogru
    # tablo sutunundan okunmustur. Full detector'a tekrar dusme.
    hizli_yol_kullanildi = (
        dogrudan_isaretli
        or
        dar_bant_isaretli
    )

    detector_fallback_kullanildi = not (
        hizli_yol_kullanildi
        or
        legacy_detector_girdisi
    )

    # =====================================================
    # 4. SON ÇARE: GENIS LABEL-AWARE DETECTOR
    # =====================================================

    if (
        hizli_yol_kullanildi
        or
        legacy_detector_girdisi
    ):
        sonuc = hizli_sonuc
        ocr = hizli_ocr

        detector_suresi = 0.0
        detector_fallback_suresi = 0.0

        fallback_denenen = list(
            dict.fromkeys(
                hizli_denenen
            )
        )

        fallback_kullanilan = list(
            dict.fromkeys(
                [
                    *hizli_kullanilan,
                    *dar_bant_kullanilan,
                ]
            )
        )

    else:
        detector_ocr, detector_suresi = _gocmen_detector_ocr_oku(
            kart
        )

        detector_sonuc = gocmen_bilgilerini_bul(
            detector_ocr
        )

        (
            detector_sonuc,
            detector_ocr,
            detector_fallback_suresi,
            detector_denenen,
            detector_kullanilan,
        ) = _gocmen_eksik_hucre_fallback(
            kart,
            detector_ocr,
            detector_sonuc,
        )

        sonuc, hizlidan_tamamlanan = (
            _gocmen_detector_eksiklerini_hizli_sonuctan_tamamla(
                detector_sonuc,
                hizli_sonuc,
            )
        )

        ocr = [
            *detector_ocr,
            *dar_bant_ocr,
        ]

        fallback_denenen = list(
            dict.fromkeys(
                [
                    *hizli_denenen,
                    *detector_denenen,
                ]
            )
        )

        fallback_kullanilan = list(
            dict.fromkeys(
                [
                    *detector_kullanilan,
                    *hizlidan_tamamlanan,
                    *dar_bant_kullanilan,
                ]
            )
        )

    # =====================================================
    # 5. SÜRELER
    # =====================================================

    ana_ocr_suresi = (
        hizli_ana_suresi
        +
        dar_bant_suresi
        +
        detector_suresi
    )

    fallback_suresi = (
        hizli_fallback_suresi
        +
        detector_fallback_suresi
    )

    # =====================================================
    # 6. DEBUG
    # =====================================================

    debug_resmi = None

    if debug:
        alanlar = [
            (
                sonuc.get(
                    "kimlik_no_item"
                ),
                "YKN",
                (
                    0,
                    255,
                    0,
                ),
            ),
            (
                sonuc.get(
                    "soyad_item"
                ),
                "SOYAD",
                (
                    0,
                    255,
                    255,
                ),
            ),
            (
                sonuc.get(
                    "ad_item"
                ),
                "AD",
                (
                    255,
                    255,
                    0,
                ),
            ),
        ]

        tarih_items = sonuc.get(
            "gecerlilik_items",
            [],
        )

        tek_bitis = (
            len(
                tarih_items
            )
            ==
            1
            and
            sonuc.get(
                "baslangic_tarihi"
            ) in {
                None,
                "",
                "Bulunamadi",
            }
            and
            sonuc.get(
                "bitis_tarihi"
            ) not in {
                None,
                "",
                "Bulunamadi",
            }
        )

        if (
            len(
                tarih_items
            )
            >=
            1
            and
            not tek_bitis
        ):
            alanlar.append(
                (
                    tarih_items[0],
                    "BASLANGIC",
                    (
                        255,
                        0,
                        255,
                    ),
                )
            )

        elif tek_bitis:
            alanlar.append(
                (
                    tarih_items[0],
                    "BITIS",
                    (
                        180,
                        0,
                        255,
                    ),
                )
            )

        if len(
            tarih_items
        ) >= 2:
            alanlar.append(
                (
                    tarih_items[1],
                    "BITIS",
                    (
                        180,
                        0,
                        255,
                    ),
                )
            )

        debug_resmi = debug_resmi_olustur(
            kart,
            alanlar,
        )

    # =====================================================
    # 7. RETURN
    # =====================================================

    return {
        "sayfa_no":
            sayfa_no,

        "belge_tipi":
            "gocmen",

        "tc_no":
            sonuc["kimlik_no"],

        "ad":
            sonuc["ad"],

        "soyad":
            sonuc["soyad"],

        "guven":
            sonuc["guven"],

        "kimlik_no_conf":
            sonuc["kimlik_no_conf"],

        "ad_conf":
            sonuc["ad_conf"],

        "soyad_conf":
            sonuc["soyad_conf"],

        "baslangic_tarihi":
            sonuc["baslangic_tarihi"],

        "bitis_tarihi":
            sonuc["bitis_tarihi"],

        "belge_gecerli":
            sonuc["belge_gecerli"],

        "gecerlilik_durumu":
            sonuc["gecerlilik_durumu"],

        "debug_resmi":
            debug_resmi,

        "tum_ocr":
            (
                ocr
                if debug
                else []
            ),

        "ocr_suresi":
            (
                ana_ocr_suresi
                +
                fallback_suresi
            ),

        "ana_ocr_suresi":
            ana_ocr_suresi,

        "fallback_ocr_suresi":
            fallback_suresi,

        "fallback_denenen_alanlar":
            fallback_denenen,

        "fallback_kullanilan_alanlar":
            fallback_kullanilan,

        "hizli_yol_kullanildi":
            hizli_yol_kullanildi,

        "hizli_hucre_ocr_suresi":
            (
                hizli_ana_suresi
                +
                hizli_fallback_suresi
            ),

        "dar_bant_kurtarma_kullanildi":
            bool(
                dar_bant_kullanilan
            ),

        "dar_bant_ocr_suresi":
            dar_bant_suresi,

        "detector_fallback_kullanildi":
            detector_fallback_kullanildi,

        "detector_ocr_suresi":
            (
                detector_suresi
                +
                detector_fallback_suresi
            ),
    }


# =========================================================
# ANA
# =========================================================

def bilgileri_cimbizla(orijinal_kart, sayfa_no=None, debug=False, belge_tipi="tc"):
    if belge_tipi not in {"tc", "eski_tc", "gocmen"}:
        return {
            "sayfa_no": sayfa_no, "belge_tipi": belge_tipi, "tc_no": "Bulunamadi",
            "ad": "Bulunamadi", "soyad": "Bulunamadi", "guven": "dusuk",
            "kimlik_no_conf": 0.0, "ad_conf": 0.0, "soyad_conf": 0.0,
            "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None,
            "gecerlilik_durumu": None, "debug_resmi": None, "tum_ocr": [], "ocr_suresi": 0.0,
            "hata": f"Desteklenmeyen belge tipi: {belge_tipi}",
        }

    if orijinal_kart is None:
        return {
            "sayfa_no": sayfa_no, "belge_tipi": belge_tipi, "tc_no": "Bulunamadi",
            "ad": "Bulunamadi", "soyad": "Bulunamadi", "guven": "dusuk",
            "kimlik_no_conf": 0.0, "ad_conf": 0.0, "soyad_conf": 0.0,
            "baslangic_tarihi": "", "bitis_tarihi": "", "belge_gecerli": None,
            "gecerlilik_durumu": None, "debug_resmi": None, "tum_ocr": [], "ocr_suresi": 0.0,
            "hata": "OCR için kart görüntüsü bulunamadı.",
        }

    if belge_tipi == "gocmen":
        return gocmen_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    if belge_tipi == "eski_tc":
        return eski_tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
    return tc_bilgilerini_oku(orijinal_kart, sayfa_no, debug)
