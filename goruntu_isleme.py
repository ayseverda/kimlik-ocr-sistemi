import cv2
import numpy as np
import os
import time


# =========================================================
# SABİTLER
# =========================================================

MAX_CALISMA_BOYUTU = 1300
SIFT_AZAMI_OZELLIK = 3200

MODUL_DIZINI = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# BELGE TİPİ KONFİGÜRASYONU
# =========================================================
#
# Eskiden TC için tek temiz bir yol (homografi -> geometri kontrolü -> warp),
# göçmen için ONA EK olarak 3 fazla katman (affine yedek, kontur+tablo-imza
# tespiti, kontur-ön-tespit — SIFT'i tamamen atlayan ayrı bir yol) ve toplam
# 5 farklı kalite metriği (radyal yakınsama, tablo çizgisi tespiti, baskın
# ton oranı, vb. ~700 satır) vardı. Üç belge tipi de şimdi AYNI tek pipeline'ı
# (kart_tespit_et_ve_duzelt) kullanıyor; farklar sadece bu sözlükte.

BELGE_KONFIG = {
    "tc": {
        "referans": "referans_kimlik.jpg",
        "hedef_w": 1000, "hedef_h": 630,
        "min_inlier": 8, "min_inlier_orani": 0.0,
        "aspect_min": 1.15, "aspect_max": 2.20,
    },
    "eski_tc": {
        "referans": "referans_eski_tc.jpg",
        "hedef_w": 750, "hedef_h": 1050,
        "min_inlier": 9, "min_inlier_orani": 0.18,
        "aspect_min": 1.05, "aspect_max": 2.10,
    },
    "gocmen": {
        "referans": "referans_gocmen.jpg",
        "hedef_w": 720, "hedef_h": 1100,
        "min_inlier": 10, "min_inlier_orani": 0.12,
        "aspect_min": 1.00, "aspect_max": 3.30,
    },
}


_DETECTOR = None
_DETECTOR_TIPI = None
_CLAHE = None
_FLANN = None
_REFERANS_CACHE = {}


def referans_yolunu_coz(yol):
    """Mutlak olmayan referans yollarını modül dizinine göre çözer."""
    yol = os.fspath(yol)
    if not os.path.isabs(yol):
        yol = os.path.join(MODUL_DIZINI, yol)
    return os.path.abspath(yol)


# =========================================================
# DETECTOR / CLAHE / MATCHER
# =========================================================

def detector_getir():
    global _DETECTOR, _DETECTOR_TIPI

    if _DETECTOR is not None:
        return _DETECTOR, _DETECTOR_TIPI

    if hasattr(cv2, "SIFT_create"):
        _DETECTOR = cv2.SIFT_create(
            nfeatures=SIFT_AZAMI_OZELLIK, contrastThreshold=0.018, edgeThreshold=12, sigma=1.6,
        )
        _DETECTOR_TIPI = "SIFT"
    else:
        _DETECTOR = cv2.ORB_create(nfeatures=7000, scaleFactor=1.2, nlevels=10, fastThreshold=5)
        _DETECTOR_TIPI = "ORB"

    return _DETECTOR, _DETECTOR_TIPI


def clahe_getir():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def matcher_getir(detector_tipi):
    global _FLANN
    if detector_tipi == "SIFT":
        if _FLANN is None:
            _FLANN = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 40})
        return _FLANN
    return cv2.BFMatcher(cv2.NORM_HAMMING)


# =========================================================
# REFERANS
# =========================================================

def referansi_hazirla(belge_tipi, yol):
    """Her referans türü kendi cache girdisinde tutulur; dosya değişmediği
    sürece yeniden okunmaz/yeniden feature çıkarılmaz."""
    global _REFERANS_CACHE

    try:
        abs_yol = referans_yolunu_coz(yol)
    except (TypeError, ValueError, OSError) as exc:
        return None, None, None, f"Geçersiz referans yolu ({belge_tipi}): {exc}"

    if not os.path.isfile(abs_yol):
        return None, None, None, f"Referans bulunamadı ({belge_tipi}): {abs_yol}"

    mtime = os.path.getmtime(abs_yol)
    cache = _REFERANS_CACHE.get(belge_tipi)
    if cache is not None and cache.get("yol") == abs_yol and cache.get("mtime") == mtime:
        return cache["resim"], cache["kp"], cache["des"], None

    resim = cv2.imread(abs_yol)
    if resim is None:
        return None, None, None, f"Referans resmi okunamadı ({belge_tipi}): {abs_yol}"

    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    detector, _ = detector_getir()
    kp, des = detector.detectAndCompute(gri, None)
    if des is None:
        return None, None, None, f"Referansta özellik bulunamadı ({belge_tipi}): {abs_yol}"

    _REFERANS_CACHE[belge_tipi] = {"yol": abs_yol, "mtime": mtime, "resim": resim, "kp": kp, "des": des}
    return resim, kp, des, None


# =========================================================
# ÇALIŞMA GÖRÜNTÜSÜNÜN DESCRIPTOR'LARI (SAYFA BAŞINA 1 KEZ)
# =========================================================
#
# Aynı fotoğraf için detectAndCompute() 3 referans için 3 kere değil, TEK
# sefer çalışır; 3 referansın hepsi bu descriptor'ları kullanır (~2.5x hız —
# daha önce sentetik veriyle doğrulandı).

def resim_descriptor_cikar(resim):
    detector, detector_tipi = detector_getir()
    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    kp, des = detector.detectAndCompute(gri, None)
    return kp, des, detector_tipi


def referansla_eslestir(kp_ref, des_ref, kp_resim, des_resim, detector_tipi):
    if des_resim is None:
        return None, [], 0

    matcher = matcher_getir(detector_tipi)
    try:
        raw = matcher.knnMatch(des_ref, des_resim, k=2)
    except cv2.error:
        return None, [], 0

    ratio = 0.74 if detector_tipi == "SIFT" else 0.80
    iyi = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(iyi) < 8:
        return None, iyi, 0

    src = np.float32([kp_ref[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
    dst = np.float32([kp_resim[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, iyi, 0
    return H, iyi, int(mask.sum())


def eslesme_skoru(iyi, inlier):
    if iyi <= 0:
        return 0.0
    return inlier * 3.5 + iyi * 0.30 + (inlier / iyi) * 100


def tek_referansi_test_et(belge_tipi, kp_resim, des_resim, detector_tipi):
    konfig = BELGE_KONFIG[belge_tipi]
    referans, kp_ref, des_ref, hata = referansi_hazirla(belge_tipi, konfig["referans"])
    if hata:
        return {"belge_tipi": belge_tipi, "basarili": False, "skor": 0.0, "hata": hata}

    H, iyi, inlier = referansla_eslestir(kp_ref, des_ref, kp_resim, des_resim, detector_tipi)
    iyi_sayisi = len(iyi)
    inlier_orani = inlier / iyi_sayisi if iyi_sayisi else 0.0
    basarili = H is not None and inlier >= konfig["min_inlier"] and inlier_orani >= konfig["min_inlier_orani"]

    return {
        "belge_tipi": belge_tipi, "basarili": basarili, "referans": referans, "H": H,
        "iyi_eslesme": iyi_sayisi, "inlier": inlier, "inlier_orani": inlier_orani,
        "skor": eslesme_skoru(iyi_sayisi, inlier), "detector": detector_tipi,
    }


def belge_tipini_bul(resim):
    h, w = resim.shape[:2]
    olcek = 1.0
    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = MAX_CALISMA_BOYUTU / max(h, w)
        calisma = cv2.resize(resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    else:
        calisma = resim

    kp_resim, des_resim, detector_tipi = resim_descriptor_cikar(calisma)

    adaylar = [
        tek_referansi_test_et(tip, kp_resim, des_resim, detector_tipi)
        for tip in BELGE_KONFIG
    ]
    basarili_adaylar = sorted(
        (a for a in adaylar if a["basarili"]), key=lambda x: x["skor"], reverse=True
    )
    return basarili_adaylar, olcek, {a["belge_tipi"]: a.get("hata") for a in adaylar if a.get("hata")}


# =========================================================
# GEOMETRİ + PERSPEKTİF (tek, genel — belge tipine özel dallanma yok,
# sadece BELGE_KONFIG'teki sayılar farklı)
# =========================================================

def koseleri_bul(referans, H):
    h, w = referans.shape[:2]
    ref_koseler = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ref_koseler, H).reshape(4, 2)


def koseler_gecerli_mi(koseler, shape, konfig):
    koseler = np.asarray(koseler, dtype=np.float32)
    if koseler.shape != (4, 2) or not np.all(np.isfinite(koseler)):
        return False

    h, w = shape[:2]
    kontur = koseler.astype(np.int32).reshape(-1, 1, 2)

    # Bow-tie/konkav dörtgen warp'ta görüntüyü ince bir şerit haline getirir;
    # cv2'nin kendi dışbükeylik kontrolü bunu ucuzca eler.
    if not cv2.isContourConvex(kontur):
        return False

    alan = abs(cv2.contourArea(kontur))
    if not 0.02 <= alan / max(1, h * w) <= 1.5:
        return False

    tol_x, tol_y = w * 0.35, h * 0.35
    if np.any(koseler[:, 0] < -tol_x) or np.any(koseler[:, 0] > w + tol_x):
        return False
    if np.any(koseler[:, 1] < -tol_y) or np.any(koseler[:, 1] > h + tol_y):
        return False

    tl, tr, br, bl = koseler
    ust, sag = np.linalg.norm(tr - tl), np.linalg.norm(br - tr)
    alt, sol = np.linalg.norm(br - bl), np.linalg.norm(bl - tl)
    if min(ust, alt, sol, sag) < 20:
        return False

    genislik, yukseklik = (ust + alt) / 2.0, (sol + sag) / 2.0
    aspect = max(genislik, yukseklik) / min(genislik, yukseklik)
    return konfig["aspect_min"] <= aspect <= konfig["aspect_max"]


def perspektif_duzelt(resim, koseler, konfig):
    hedef_w, hedef_h = konfig["hedef_w"], konfig["hedef_h"]
    src = np.asarray(koseler, dtype=np.float32)
    dst = np.float32([[0, 0], [hedef_w - 1, 0], [hedef_w - 1, hedef_h - 1], [0, hedef_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        resim, M, (hedef_w, hedef_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def cikti_kaliteli_mi(kart):
    """Warp çıktısının çökmüş/dejenere olmadığını (büyük tek renk alan +
    ince çizgisel şerit değil) tek, ucuz bir kontrolle doğrular. Eskiden
    beş ayrı metrik (radyal yakınsama, tablo imzası, baskın ton oranı,
    hücre-bazlı ayrıntı kapsamı...) vardı; burada tek bir std/dinamik-aralık
    kontrolü var — çok daha az kod, "çökmüş warp"ı yakalamak için yeterli."""
    if kart is None or kart.size == 0:
        return False
    gri = cv2.cvtColor(kart, cv2.COLOR_BGR2GRAY) if kart.ndim == 3 else kart
    ornek = gri[::4, ::4]  # ucuz olsun diye seyrek örnekle
    if ornek.size == 0:
        return False
    p05, p95 = np.percentile(ornek, (5, 95))
    return (p95 - p05) >= 12.0 and float(np.std(ornek)) >= 4.0


# =========================================================
# DEBUG (yalnızca kart sınırı — basit, ucuz)
# =========================================================

def kart_debug_resmi(resim, koseler):
    debug = resim.copy()
    poly = np.asarray(koseler, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [poly], True, (0, 255, 0), 5, cv2.LINE_AA)
    for isim, nokta in zip(("TL", "TR", "BR", "BL"), koseler):
        x, y = map(int, nokta)
        cv2.circle(debug, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(debug, isim, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return debug


# =========================================================
# GİRDİ DOĞRULAMA
# =========================================================

def _girdi_hazirla(resim):
    if resim is None:
        return None, "Görüntü bulunamadı."
    if not isinstance(resim, np.ndarray) or resim.size == 0:
        return None, "Görüntü geçersiz veya boş."
    if resim.ndim not in (2, 3) or resim.shape[0] < 2 or resim.shape[1] < 2:
        return None, f"Geçersiz görüntü şekli: {resim.shape}."
    if resim.dtype != np.uint8:
        return None, f"Geçersiz görüntü veri tipi: {resim.dtype}; uint8 bekleniyor."

    if resim.ndim == 2:
        hazir = cv2.cvtColor(resim, cv2.COLOR_GRAY2BGR)
    elif resim.shape[2] == 1:
        hazir = cv2.cvtColor(resim[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif resim.shape[2] == 3:
        hazir = resim
    elif resim.shape[2] == 4:
        hazir = cv2.cvtColor(resim, cv2.COLOR_BGRA2BGR)
    else:
        return None, f"Geçersiz kanal sayısı: {resim.shape[2]}."

    return np.ascontiguousarray(hazir), None


# =========================================================
# TEK PIPELINE
# =========================================================

def kart_tespit_et_ve_duzelt(resim, debug_kart=False):
    """Üç belge tipi de (tc/eski_tc/gocmen) AYNI akışı izler:
    1) referanslarla eşleştir, skora göre sırala
    2) en iyi adaydan başlayarak: köşeleri bul -> geometri makul mü -> warp
       -> çıktı dejenere değil mi
    3) ilk başarılı aday kabul edilir; hiçbiri geçemezse 'bulunamadı' döner.
    Eskiden göçmen için homografi başarısız olunca 3 farklı ek yöntem
    (affine, kontur+tablo-imzası, SIFT'siz kontur ön-tespit) deneniyordu;
    o üç yöntem de kaldırıldı — artık üç belge tipi de sadece bu tek yolu
    kullanıyor."""
    baslangic = time.perf_counter()
    sonuc = {
        "basarili": False, "kart": None, "belge_tipi": None, "debug_resmi": None,
        "koseler": None, "iyi_eslesme": 0, "inlier": 0, "inlier_orani": 0.0,
        "skor": 0.0, "mesaj": "", "referans_hatalari": {}, "tespit_suresi": 0.0,
    }

    resim, girdi_hatasi = _girdi_hazirla(resim)
    if girdi_hatasi:
        sonuc["mesaj"] = girdi_hatasi
        sonuc["tespit_suresi"] = time.perf_counter() - baslangic
        return sonuc

    adaylar, olcek, referans_hatalari = belge_tipini_bul(resim)
    sonuc["referans_hatalari"] = referans_hatalari

    if not adaylar:
        ayrinti = f" Referans hataları: {' | '.join(referans_hatalari.values())}" if referans_hatalari else ""
        sonuc["mesaj"] = "Belge tipi bulunamadı." + ayrinti
        sonuc["tespit_suresi"] = time.perf_counter() - baslangic
        return sonuc

    sonuc.update({
        "belge_tipi": adaylar[0]["belge_tipi"], "iyi_eslesme": adaylar[0]["iyi_eslesme"],
        "inlier": adaylar[0]["inlier"], "inlier_orani": adaylar[0]["inlier_orani"],
        "skor": adaylar[0]["skor"],
    })

    for aday in adaylar:
        belge_tipi = aday["belge_tipi"]
        konfig = BELGE_KONFIG[belge_tipi]

        koseler = koseleri_bul(aday["referans"], aday["H"])
        if olcek != 1.0:
            koseler = koseler / olcek

        if not koseler_gecerli_mi(koseler, resim.shape, konfig):
            continue

        kart = perspektif_duzelt(resim, koseler, konfig)
        if not cikti_kaliteli_mi(kart):
            continue

        sonuc.update({
            "basarili": True, "kart": kart, "belge_tipi": belge_tipi, "koseler": koseler,
            "iyi_eslesme": aday["iyi_eslesme"], "inlier": aday["inlier"],
            "inlier_orani": aday["inlier_orani"], "skor": aday["skor"],
            "mesaj": "Kart tespit edildi ve yönü düzeltildi.",
        })
        if debug_kart:
            sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)
        sonuc["tespit_suresi"] = time.perf_counter() - baslangic
        return sonuc

    sonuc["mesaj"] = f"Belge tipi eşleşmesi bulundu ({adaylar[0]['belge_tipi']}), ancak güvenilir şekilde düzeltilemedi."
    sonuc["tespit_suresi"] = time.perf_counter() - baslangic
    return sonuc