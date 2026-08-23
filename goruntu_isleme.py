import cv2
import numpy as np
import os


# =========================================================
# SABİTLER
# =========================================================

MAX_CALISMA_BOYUTU = 1800

BELGE_REFERANSLARI = {
    "tc": "referans_kimlik.jpg",
    "gocmen": "referans_gocmen.jpg",
    "eski_tc": "referans_eski_tc.jpg",
}


_DETECTOR = None
_DETECTOR_TIPI = None
_CLAHE = None
_FLANN = None
_REFERANS_CACHE = {}


# =========================================================
# DETECTOR / CLAHE / MATCHER
# =========================================================

def detector_getir():
    global _DETECTOR, _DETECTOR_TIPI

    if _DETECTOR is not None:
        return _DETECTOR, _DETECTOR_TIPI

    if hasattr(cv2, "SIFT_create"):
        _DETECTOR = cv2.SIFT_create(nfeatures=5500, contrastThreshold=0.018, edgeThreshold=12, sigma=1.6)
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
            _FLANN = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 60})
        return _FLANN

    return cv2.BFMatcher(cv2.NORM_HAMMING)


# =========================================================
# REFERANS
# =========================================================

def referansi_hazirla(belge_tipi, yol):
    """Her referans türü kendi cache girdisinde tutulur; dosya değişmediği
    sürece yeniden okunmaz/yeniden feature çıkarılmaz."""
    global _REFERANS_CACHE

    if not os.path.exists(yol):
        return None, None, None, f"Referans bulunamadı: {yol}"

    abs_yol = os.path.abspath(yol)
    mtime = os.path.getmtime(abs_yol)
    cache = _REFERANS_CACHE.get(belge_tipi)

    if cache is not None and cache.get("yol") == abs_yol and cache.get("mtime") == mtime:
        return cache["resim"], cache["kp"], cache["des"], None

    resim = cv2.imread(abs_yol)
    if resim is None:
        return None, None, None, "Referans resmi okunamadı."

    gri = clahe_getir().apply(cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY))
    detector, _ = detector_getir()
    kp, des = detector.detectAndCompute(gri, None)

    if des is None:
        return None, None, None, "Referansta feature bulunamadı."

    _REFERANS_CACHE[belge_tipi] = {"yol": abs_yol, "mtime": mtime, "resim": resim, "kp": kp, "des": des}
    return resim, kp, des, None


# =========================================================
# ÇALIŞMA GÖRÜNTÜSÜNÜN DESCRIPTOR'LARI (SAYFA BAŞINA 1 KEZ)
# =========================================================
#
# ÖNEMLİ HIZ DÜZELTMESİ: eskiden her sayfa için 3 referans (tc, göçmen,
# eski_tc) sırayla test ediliyordu ve her testte 'calisma' görüntüsünün SIFT
# keypoint/descriptor'ları BAŞTAN çıkarılıyordu — yani aynı fotoğraf için
# detectAndCompute() 3 kere çalışıyordu. Bu, sayfa başına en pahalı adımın
# gereksiz yere 3 katına çıkması demekti. Descriptor'lar yalnızca görüntüye
# bağlı olduğu için (hangi referansla karşılaştırıldığından bağımsız),
# sentetik veriyle doğrulandığı gibi 1 kez çıkarılıp 3 referansa karşı da
# kullanılması BİREBİR AYNI sonucu ~2.5x daha hızlı verir.

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


# =========================================================
# SKOR
# =========================================================

def eslesme_skoru(iyi, inlier):
    if iyi <= 0:
        return 0.0
    oran = inlier / iyi
    return inlier * 3.5 + iyi * 0.30 + oran * 100


# =========================================================
# TEK REFERANS TEST
# =========================================================

def tek_referansi_test_et(belge_tipi, referans_yolu, kp_resim, des_resim, detector_tipi):
    referans, kp_ref, des_ref, hata = referansi_hazirla(belge_tipi, referans_yolu)
    if hata:
        return {"belge_tipi": belge_tipi, "basarili": False, "skor": 0.0, "hata": hata}

    H, iyi, inlier = referansla_eslestir(kp_ref, des_ref, kp_resim, des_resim, detector_tipi)
    iyi_sayisi = len(iyi)
    inlier_orani = inlier / iyi_sayisi if iyi_sayisi else 0.0
    skor = eslesme_skoru(iyi_sayisi, inlier)

    if belge_tipi == "eski_tc":
        basarili = H is not None and inlier >= 7 and inlier_orani >= 0.10
    elif belge_tipi == "gocmen":
        basarili = H is not None and inlier >= 10 and inlier_orani >= 0.12
    else:
        basarili = H is not None and inlier >= 8

    return {
        "belge_tipi": belge_tipi, "basarili": basarili, "referans": referans,
        "kp_ref": kp_ref, "kp_resim": kp_resim, "H": H, "iyi": iyi,
        "iyi_eslesme": iyi_sayisi, "inlier": inlier, "inlier_orani": inlier_orani,
        "skor": skor, "detector": detector_tipi,
    }


# =========================================================
# BELGE TİPİ
# =========================================================

def belge_tipini_bul(resim):
    h, w = resim.shape[:2]
    olcek = 1.0

    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = MAX_CALISMA_BOYUTU / max(h, w)
        calisma = cv2.resize(resim, None, fx=olcek, fy=olcek, interpolation=cv2.INTER_AREA)
    else:
        calisma = resim

    # Sayfa başına TEK sefer: 3 referansın hepsi bu descriptor'ları kullanır.
    kp_resim, des_resim, detector_tipi = resim_descriptor_cikar(calisma)

    sonuclar = {
        tip: tek_referansi_test_et(tip, yol, kp_resim, des_resim, detector_tipi)
        for tip, yol in BELGE_REFERANSLARI.items()
    }

    adaylar = [x for x in sonuclar.values() if x.get("basarili", False)]

    if not adaylar:
        return {"basarili": False, "belge_tipi": None, "sonuclar": sonuclar, "calisma": calisma, "olcek": olcek}

    en_iyi = max(adaylar, key=lambda x: x["skor"])
    return {
        "basarili": True, "belge_tipi": en_iyi["belge_tipi"], "en_iyi": en_iyi,
        "sonuclar": sonuclar, "calisma": calisma, "olcek": olcek,
    }


# =========================================================
# REFERANS KÖŞELERİ
# =========================================================

def koseleri_bul(referans, H):
    h, w = referans.shape[:2]
    # Sıra referans yönünü temsil ediyor: TL -> TR -> BR -> BL. Sonradan sıralanmıyor.
    ref_koseler = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(ref_koseler, H).reshape(4, 2)


# =========================================================
# GEOMETRİ KONTROL
# =========================================================

def koseler_gecerli_mi(koseler, shape, belge_tipi):
    if koseler is None:
        return False

    koseler = np.asarray(koseler, dtype=np.float32)
    if not np.all(np.isfinite(koseler)):
        return False

    h, w = shape[:2]
    kontur = koseler.astype(np.int32).reshape(-1, 1, 2)
    alan_orani = abs(cv2.contourArea(kontur)) / max(1, h * w)

    # Eski TC / göçmen için daha toleranslı: eğik/yan/fotoğraf şeklinde gelebiliyor.
    if belge_tipi == "eski_tc":
        if not (0.01 <= alan_orani <= 2.0):
            return False
        tol_x, tol_y = w * 0.60, h * 0.60
    elif belge_tipi == "gocmen":
        if not (0.01 <= alan_orani <= 2.5):
            return False
        tol_x, tol_y = w * 0.80, h * 0.80
    else:
        if not (0.02 <= alan_orani <= 1.3):
            return False
        tol_x, tol_y = w * 0.25, h * 0.25

    if np.any(koseler[:, 0] < -tol_x) or np.any(koseler[:, 0] > w + tol_x):
        return False
    if np.any(koseler[:, 1] < -tol_y) or np.any(koseler[:, 1] > h + tol_y):
        return False

    tl, tr, br, bl = koseler
    ust, alt = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    sol, sag = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)

    if min(ust, alt, sol, sag) < 20:
        return False

    genislik, yukseklik = (ust + alt) / 2.0, (sol + sag) / 2.0
    aspect = max(genislik, yukseklik) / min(genislik, yukseklik)

    if belge_tipi == "eski_tc":
        return 1.05 <= aspect <= 2.10
    if belge_tipi == "tc":
        return 1.15 <= aspect <= 2.20
    if belge_tipi == "gocmen":
        return 1.00 <= aspect <= 3.30
    return False


# =========================================================
# PERSPEKTİF DÜZELT
# =========================================================

def perspektif_duzelt(resim, koseler, belge_tipi):
    """koseler homography'den referans TL->TR->BR->BL sırasıyla gelir; kart
    görüntüde 90 derece dönmüş olsa bile correspondence doğru kalır, bu
    yüzden köşeler burada tekrar sıralanmıyor."""
    src = np.asarray(koseler, dtype=np.float32)

    if belge_tipi == "eski_tc":
        hedef_w, hedef_h = 750, 1050
    elif belge_tipi == "gocmen":
        hedef_w, hedef_h = 800, 1100
    else:
        hedef_w, hedef_h = 1000, 630

    dst = np.float32([[0, 0], [hedef_w - 1, 0], [hedef_w - 1, hedef_h - 1], [0, hedef_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        resim, M, (hedef_w, hedef_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# =========================================================
# DEBUG — yalnızca kart sınırı (basit, ucuz); SIFT eşleşme çizgisi kaldırıldı
# =========================================================
#
# Not: iki görüntü arasında eşleşen keypoint'leri çizgilerle birleştiren
# 'match_debug_resmi' (cv2.drawMatches) kaldırıldı. Bu görsel yalnızca
# geliştirme sırasında "SIFT hangi noktaları eşleştirdi" diye incelemek
# içindi; TC/ad/soyad çıkarımını hiçbir şekilde etkilemiyordu, ekstra kod ve
# (açıldığında) ekstra render maliyeti dışında bir katkısı yoktu. Kart
# sınırını (bulunan dörtgeni) tek görüntü üzerinde gösteren 'kart_debug_resmi'
# — asıl işe yarayan, "kart doğru bulundu mu" sorusuna cevap veren debug —
# olduğu gibi kalıyor.

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
# ANA
# =========================================================

def kart_tespit_et_ve_duzelt(resim, debug_kart=False):
    sonuc = {
        "basarili": False, "kart": None, "belge_tipi": None, "debug_resmi": None,
        "koseler": None, "fallback": False, "duzeltildi": False,
        "iyi_eslesme": 0, "inlier": 0, "inlier_orani": 0.0, "skor": 0.0, "mesaj": "",
    }

    if resim is None:
        sonuc["mesaj"] = "Görüntü bulunamadı."
        return sonuc

    tip_sonuc = belge_tipini_bul(resim)
    if not tip_sonuc.get("basarili", False):
        sonuc["mesaj"] = "Belge tipi bulunamadı."
        return sonuc

    belge_tipi = tip_sonuc["belge_tipi"]
    en_iyi = tip_sonuc["en_iyi"]

    sonuc.update({
        "belge_tipi": belge_tipi,
        "iyi_eslesme": en_iyi.get("iyi_eslesme", 0),
        "inlier": en_iyi.get("inlier", 0),
        "inlier_orani": en_iyi.get("inlier_orani", 0.0),
        "skor": en_iyi.get("skor", 0.0),
    })

    try:
        koseler = koseleri_bul(en_iyi["referans"], en_iyi["H"])
        olcek = tip_sonuc["olcek"]
        if olcek != 1.0:
            koseler = koseler / olcek
        sonuc["koseler"] = koseler
    except Exception:
        koseler = None

    # ---- Önce normal geometri -----------------------------------------------
    geometri_ok = koseler is not None and koseler_gecerli_mi(koseler, resim.shape, belge_tipi)

    if geometri_ok:
        try:
            kart = perspektif_duzelt(resim, koseler, belge_tipi)
            sonuc.update({
                "kart": kart, "basarili": True, "duzeltildi": True, "fallback": False,
                "mesaj": "Kart tespit edildi ve yönü düzeltildi.",
            })
            if debug_kart:
                sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)
            return sonuc
        except Exception:
            pass

    # ---- Eski TC özel: homography güçlüyse köşe kontrolü başarısız olsa bile dene ----
    if belge_tipi == "eski_tc" and koseler is not None and en_iyi.get("inlier", 0) >= 7:
        try:
            kart = perspektif_duzelt(resim, koseler, "eski_tc")
            gri = cv2.cvtColor(kart, cv2.COLOR_BGR2GRAY)
            if kart.size > 0 and np.std(gri) > 8:
                sonuc.update({
                    "kart": kart, "basarili": True, "duzeltildi": True, "fallback": False,
                    "mesaj": "Eski T.C. kimlik güçlü SIFT eşleşmesiyle yönü düzeltilerek işlendi.",
                })
                if debug_kart:
                    sonuc["debug_resmi"] = kart_debug_resmi(resim, koseler)
                return sonuc
        except Exception:
            pass

    # ---- Diğer belgeler için fallback: perspektif düzeltilemedi, ham görüntü ----
    if belge_tipi != "eski_tc" and en_iyi.get("inlier", 0) >= 8:
        sonuc.update({
            "basarili": True, "kart": resim.copy(), "fallback": True,
            "mesaj": "Belge tipi bulundu fakat perspektif düzeltilemedi.",
        })
        return sonuc

    sonuc["mesaj"] = "Kimlik güvenilir şekilde düzeltilemedi."
    return sonuc