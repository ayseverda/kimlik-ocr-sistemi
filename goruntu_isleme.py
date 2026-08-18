import cv2
import numpy as np
import os


# =========================================================
# SABİTLER
# =========================================================

HEDEF_GENISLIK = 1000
HEDEF_YUKSEKLIK = 630
MAX_CALISMA_BOYUTU = 1800


# =========================================================
# CACHE
# =========================================================

_DETECTOR = None
_DETECTOR_TIPI = None
_CLAHE = None
_FLANN = None  # SIFT eşleştirme için: brute-force yerine yaklaşık en-yakın-komşu, çok daha hızlı

_REFERANS_CACHE = {"yol": None, "mtime": None, "resim": None, "kp": None, "des": None}


# =========================================================
# DETECTOR
# =========================================================

def detector_getir():
    global _DETECTOR, _DETECTOR_TIPI

    if _DETECTOR is not None:
        return _DETECTOR, _DETECTOR_TIPI

    if hasattr(cv2, "SIFT_create"):
        _DETECTOR = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10)
        _DETECTOR_TIPI = "SIFT"
    else:
        _DETECTOR = cv2.ORB_create(
            nfeatures=6000, scaleFactor=1.2, nlevels=10, edgeThreshold=15, fastThreshold=7
        )
        _DETECTOR_TIPI = "ORB"

    return _DETECTOR, _DETECTOR_TIPI


# =========================================================
# CLAHE
# =========================================================

def clahe_getir():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


# =========================================================
# MATCHER
# =========================================================

def matcher_getir(detector_tipi):
    """SIFT için FLANN (KD-tree, yaklaşık en-yakın-komşu) — aynı ratio-test
    prensibiyle çalışır ama binlerce descriptor'da BFMatcher'dan belirgin
    hızlıdır. ORB için Hamming mesafesi FLANN-LSH gerektirdiğinden ve
    descriptor sayısı zaten daha az olduğundan BFMatcher'da kalınır."""
    global _FLANN

    if detector_tipi == "SIFT":
        if _FLANN is None:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=50)
            _FLANN = cv2.FlannBasedMatcher(index_params, search_params)
        return _FLANN

    return cv2.BFMatcher(cv2.NORM_HAMMING)


# =========================================================
# REFERANSI HAZIRLA
# =========================================================

def referansi_hazirla(referans_yolu):
    """Referans kart sadece ilk seferde işlenir.
    Dosya değişirse cache otomatik yenilenir."""
    global _REFERANS_CACHE

    if not os.path.exists(referans_yolu):
        return None, None, None, "Referans dosyası bulunamadı."

    mutlak_yol = os.path.abspath(referans_yolu)
    mtime = os.path.getmtime(mutlak_yol)

    if (
        _REFERANS_CACHE["yol"] == mutlak_yol
        and _REFERANS_CACHE["mtime"] == mtime
        and _REFERANS_CACHE["resim"] is not None
        and _REFERANS_CACHE["des"] is not None
    ):
        return _REFERANS_CACHE["resim"], _REFERANS_CACHE["kp"], _REFERANS_CACHE["des"], None

    referans = cv2.imread(mutlak_yol)
    if referans is None:
        return None, None, None, "Referans görüntüsü okunamadı."

    detector, _ = detector_getir()
    clahe = clahe_getir()

    gri = cv2.cvtColor(referans, cv2.COLOR_BGR2GRAY)
    gri = clahe.apply(gri)
    kp, des = detector.detectAndCompute(gri, None)

    if des is None:
        return None, None, None, "Referansta yeterli feature bulunamadı."

    _REFERANS_CACHE = {"yol": mutlak_yol, "mtime": mtime, "resim": referans, "kp": kp, "des": des}
    return referans, kp, des, None


# =========================================================
# NOKTALARI SIRALA
# =========================================================

def noktalari_sirala(pts):
    pts = np.asarray(pts, dtype=np.float32)
    toplam = pts.sum(axis=1)
    fark = np.diff(pts, axis=1).reshape(-1)

    return np.array([
        pts[np.argmin(toplam)], pts[np.argmin(fark)],
        pts[np.argmax(toplam)], pts[np.argmax(fark)],
    ], dtype=np.float32)


# =========================================================
# KÖŞELER GEÇERLİ Mİ?
# =========================================================

def koseler_gecerli_mi(koseler, resim_shape):
    h, w = resim_shape[:2]
    koseler = np.asarray(koseler, dtype=np.float32).reshape(4, 2)
    kontur = koseler.astype(np.int32).reshape(-1, 1, 2)

    if not cv2.isContourConvex(kontur):
        return False

    alan = abs(cv2.contourArea(kontur))
    if not 0.02 <= alan / (h * w) <= 1.20:
        return False

    tolerans_x, tolerans_y = w * 0.25, h * 0.25
    if np.any(koseler[:, 0] < -tolerans_x) or np.any(koseler[:, 0] > w + tolerans_x):
        return False
    if np.any(koseler[:, 1] < -tolerans_y) or np.any(koseler[:, 1] > h + tolerans_y):
        return False

    tl, tr, br, bl = noktalari_sirala(koseler)
    genislik = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    yukseklik = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    if min(genislik, yukseklik) <= 0:
        return False

    oran = max(genislik, yukseklik) / min(genislik, yukseklik)
    return 1.15 <= oran <= 2.10


# =========================================================
# PERSPEKTİF
# =========================================================

def perspektif_duzelt(resim, koseler):
    koseler = np.asarray(koseler, dtype=np.float32).reshape(4, 2)

    hedef = np.array([
        [0, 0], [HEDEF_GENISLIK - 1, 0],
        [HEDEF_GENISLIK - 1, HEDEF_YUKSEKLIK - 1], [0, HEDEF_YUKSEKLIK - 1],
    ], dtype=np.float32)

    matris = cv2.getPerspectiveTransform(koseler, hedef)
    sonuc = cv2.warpPerspective(
        resim, matris, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )

    margin_x = int(HEDEF_GENISLIK * 0.012)
    margin_y = int(HEDEF_YUKSEKLIK * 0.012)
    sonuc = sonuc[margin_y:HEDEF_YUKSEKLIK - margin_y, margin_x:HEDEF_GENISLIK - margin_x]
    return cv2.resize(sonuc, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK), interpolation=cv2.INTER_CUBIC)


# =========================================================
# FEATURE MATCH
# =========================================================

def feature_eslestir(kp_ref, des_ref, resim):
    detector, detector_tipi = detector_getir()
    clahe = clahe_getir()

    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    gri = clahe.apply(gri)
    kp_resim, des_resim = detector.detectAndCompute(gri, None)

    if des_resim is None:
        return None, [], kp_resim, detector_tipi, 0

    matcher = matcher_getir(detector_tipi)

    try:
        eslesmeler = matcher.knnMatch(des_ref, des_resim, k=2)
    except cv2.error:
        return None, [], kp_resim, detector_tipi, 0

    ratio = 0.72 if detector_tipi == "SIFT" else 0.78
    iyi = [m for pair in eslesmeler if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

    if len(iyi) < 10:
        return None, iyi, kp_resim, detector_tipi, 0

    kaynak = np.float32([kp_ref[m.queryIdx].pt for m in iyi]).reshape(-1, 1, 2)
    hedef = np.float32([kp_resim[m.trainIdx].pt for m in iyi]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(kaynak, hedef, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, iyi, kp_resim, detector_tipi, 0

    return H, iyi, kp_resim, detector_tipi, int(mask.sum())


# =========================================================
# KÖŞELERİ BUL
# =========================================================

def koseleri_bul(referans, H):
    h, w = referans.shape[:2]
    ref_koseler = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ref_koseler, H).reshape(4, 2)


# =========================================================
# DEBUG
# =========================================================

def match_debug_resmi(referans, resim, kp_ref, kp_resim, matchler, maksimum=60):
    if not matchler:
        return None
    secilen = sorted(matchler, key=lambda m: m.distance)[:maksimum]
    return cv2.drawMatches(
        referans, kp_ref, resim, kp_resim, secilen, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def kart_debug_resmi(resim, koseler):
    debug = resim.copy()
    polygon = koseler.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(debug, [polygon], True, (0, 255, 0), 5, cv2.LINE_AA)

    for i, (x, y) in enumerate(koseler.astype(np.int32)):
        cv2.circle(debug, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(debug, str(i + 1), (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def kart_tespit_et_ve_duzelt(resim, referans_yolu="referans_kimlik.jpg", debug_match=False, debug_kart=False):
    sonuc = {
        "basarili": False, "kart": None, "debug_resmi": None, "match_debug": None,
        "koseler": None, "iyi_eslesme": 0, "inlier": 0, "detector": "", "mesaj": "",
    }

    if resim is None:
        sonuc["mesaj"] = "Görüntü okunamadı."
        return sonuc

    referans, kp_ref, des_ref, hata = referansi_hazirla(referans_yolu)
    if hata:
        sonuc["mesaj"] = hata
        return sonuc

    h, w = resim.shape[:2]
    olcek = 1.0
    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = MAX_CALISMA_BOYUTU / max(h, w)
        calisma = cv2.resize(resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    else:
        calisma = resim

    H, iyi, kp_resim, detector_tipi, inlier_sayisi = feature_eslestir(kp_ref, des_ref, calisma)

    sonuc["iyi_eslesme"] = len(iyi)
    sonuc["inlier"] = inlier_sayisi
    sonuc["detector"] = detector_tipi

    if debug_match:
        sonuc["match_debug"] = match_debug_resmi(referans, calisma, kp_ref, kp_resim, iyi)

    if H is None:
        sonuc["mesaj"] = f"Yeterli özellik eşleşmesi yok. İyi eşleşme: {len(iyi)}"
        return sonuc

    if inlier_sayisi < 8:
        sonuc["mesaj"] = f"Homography güvenilir değil. Inlier: {inlier_sayisi}"
        return sonuc

    koseler = koseleri_bul(referans, H)
    if olcek != 1.0:
        koseler = koseler / olcek

    if not koseler_gecerli_mi(koseler, resim.shape):
        sonuc["mesaj"] = "Feature eşleşmesi bulundu ancak kart sınırları mantıklı değil."
        return sonuc

    kart = perspektif_duzelt(resim, koseler)

    if debug_kart:
        sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)

    sonuc.update({
        "basarili": True, "kart": kart, "koseler": koseler,
        "mesaj": "Kimlik başarıyla tespit edildi.",
    })
    return sonuc