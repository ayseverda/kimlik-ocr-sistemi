import cv2
import numpy as np
import os


HEDEF_GENISLIK = 1000
HEDEF_YUKSEKLIK = 630


# =========================================================
# NOKTALARI SIRALA
# =========================================================

def noktalari_sirala(pts):
    """
    4 köşeyi şu sıraya getirir:

    sol üst
    sağ üst
    sağ alt
    sol alt
    """

    pts = np.asarray(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    toplam = pts.sum(axis=1)

    rect[0] = pts[np.argmin(toplam)]
    rect[2] = pts[np.argmax(toplam)]

    fark = np.diff(pts, axis=1).reshape(-1)

    rect[1] = pts[np.argmin(fark)]
    rect[3] = pts[np.argmax(fark)]

    return rect


# =========================================================
# KÖŞELER MANTIKLI MI?
# =========================================================

def koseler_gecerli_mi(koseler, resim_shape):
    """
    Homography'nin saçma bir dörtgen üretmesini engeller.
    """

    h, w = resim_shape[:2]

    koseler = np.asarray(
        koseler,
        dtype=np.float32
    ).reshape(4, 2)

    # -----------------------------
    # Convex dörtgen mi?
    # -----------------------------

    kontur = koseler.astype(
        np.int32
    ).reshape(-1, 1, 2)

    if not cv2.isContourConvex(kontur):
        return False

    # -----------------------------
    # Alan yeterince büyük mü?
    # -----------------------------

    alan = abs(
        cv2.contourArea(kontur)
    )

    resim_alani = h * w

    alan_orani = alan / resim_alani

    # Görselin %2'sinden küçükse
    # muhtemelen yanlış eşleşmedir.
    if alan_orani < 0.02:
        return False

    # Tüm görüntüden de büyük olamaz
    if alan_orani > 1.20:
        return False

    # -----------------------------
    # Çok uzakta saçma noktalar var mı?
    # -----------------------------

    tolerans_x = w * 0.25
    tolerans_y = h * 0.25

    for x, y in koseler:

        if x < -tolerans_x:
            return False

        if x > w + tolerans_x:
            return False

        if y < -tolerans_y:
            return False

        if y > h + tolerans_y:
            return False

    # -----------------------------
    # Kart oranı mantıklı mı?
    # -----------------------------

    sirali = noktalari_sirala(
        koseler
    )

    tl, tr, br, bl = sirali

    ust = np.linalg.norm(
        tr - tl
    )

    alt = np.linalg.norm(
        br - bl
    )

    sol = np.linalg.norm(
        bl - tl
    )

    sag = np.linalg.norm(
        br - tr
    )

    genislik = (
        ust + alt
    ) / 2

    yukseklik = (
        sol + sag
    ) / 2

    if yukseklik == 0:
        return False

    oran = max(
        genislik,
        yukseklik
    ) / min(
        genislik,
        yukseklik
    )

    # Kimlik gerçek oranı ~1.586.
    # Perspektiften dolayı geniş tolerans.
    if not 1.15 <= oran <= 2.10:
        return False

    return True


# =========================================================
# PERSPEKTİF DÜZELTME
# =========================================================

def perspektif_duzelt(resim, koseler):

    koseler = np.asarray(
        koseler,
        dtype=np.float32
    ).reshape(4, 2)

    hedef = np.array(
        [
            [0, 0],
            [HEDEF_GENISLIK - 1, 0],
            [HEDEF_GENISLIK - 1, HEDEF_YUKSEKLIK - 1],
            [0, HEDEF_YUKSEKLIK - 1]
        ],
        dtype=np.float32
    )

    matris = cv2.getPerspectiveTransform(
        koseler,
        hedef
    )

    sonuc = cv2.warpPerspective(
        resim,
        matris,
        (
            HEDEF_GENISLIK,
            HEDEF_YUKSEKLIK
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # -------------------------------------------------
    # KENAR TEMİZLEME
    # -------------------------------------------------
    # Her taraftan yaklaşık %1 kırp.
    # 1000x630 için:
    # yatayda ~10 px
    # dikeyde ~6 px
    # -------------------------------------------------

    margin_x = int(
        HEDEF_GENISLIK * 0.012
    )

    margin_y = int(
        HEDEF_YUKSEKLIK * 0.012
    )

    sonuc = sonuc[
        margin_y:HEDEF_YUKSEKLIK - margin_y,
        margin_x:HEDEF_GENISLIK - margin_x
    ]

    # Kırptıktan sonra tekrar standart boyuta getir
    sonuc = cv2.resize(
        sonuc,
        (
            HEDEF_GENISLIK,
            HEDEF_YUKSEKLIK
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return sonuc
# =========================================================
# FEATURE DETECTOR
# =========================================================

def detector_olustur():
    """
    Bilgisayarda SIFT varsa onu kullan.
    Yoksa ORB'a düş.
    """

    if hasattr(cv2, "SIFT_create"):

        detector = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.02,
            edgeThreshold=10
        )

        return detector, "SIFT"

    detector = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=10,
        edgeThreshold=15,
        fastThreshold=7
    )

    return detector, "ORB"


# =========================================================
# FEATURE MATCHING
# =========================================================

def feature_eslestir(
    referans,
    resim
):
    """
    Referans kimlikle kullanıcının görüntüsünü eşleştirir.

    return:
        homography
        iyi_matchler
        kp_ref
        kp_resim
        detector_tipi
        inlier_sayisi
    """

    detector, detector_tipi = detector_olustur()

    ref_gri = cv2.cvtColor(
        referans,
        cv2.COLOR_BGR2GRAY
    )

    resim_gri = cv2.cvtColor(
        resim,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------
    # Kontrast güçlendirme
    # ---------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    ref_gri = clahe.apply(
        ref_gri
    )

    resim_gri = clahe.apply(
        resim_gri
    )

    # ---------------------------------------------
    # Özellik noktaları
    # ---------------------------------------------

    kp_ref, des_ref = detector.detectAndCompute(
        ref_gri,
        None
    )

    kp_resim, des_resim = detector.detectAndCompute(
        resim_gri,
        None
    )

    if des_ref is None or des_resim is None:

        return (
            None,
            [],
            kp_ref,
            kp_resim,
            detector_tipi,
            0
        )

    # ---------------------------------------------
    # Matcher
    # ---------------------------------------------

    if detector_tipi == "SIFT":

        matcher = cv2.BFMatcher(
            cv2.NORM_L2
        )

    else:

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

    try:

        eslesmeler = matcher.knnMatch(
            des_ref,
            des_resim,
            k=2
        )

    except cv2.error:

        return (
            None,
            [],
            kp_ref,
            kp_resim,
            detector_tipi,
            0
        )

    # ---------------------------------------------
    # Lowe Ratio Test
    # ---------------------------------------------

    iyi = []

    for pair in eslesmeler:

        if len(pair) < 2:
            continue

        m, n = pair

        if detector_tipi == "SIFT":
            oran = 0.72
        else:
            oran = 0.78

        if m.distance < oran * n.distance:

            iyi.append(m)

    # Homography için minimum 4 gerekir,
    # biz daha güvenli davranıyoruz.
    if len(iyi) < 10:

        return (
            None,
            iyi,
            kp_ref,
            kp_resim,
            detector_tipi,
            0
        )

    # ---------------------------------------------
    # Kaynak / hedef noktaları
    # ---------------------------------------------

    kaynak = np.float32(
        [
            kp_ref[m.queryIdx].pt
            for m in iyi
        ]
    ).reshape(-1, 1, 2)

    hedef = np.float32(
        [
            kp_resim[m.trainIdx].pt
            for m in iyi
        ]
    ).reshape(-1, 1, 2)

    # ---------------------------------------------
    # Homography
    # ---------------------------------------------

    H, mask = cv2.findHomography(
        kaynak,
        hedef,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:

        return (
            None,
            iyi,
            kp_ref,
            kp_resim,
            detector_tipi,
            0
        )

    inlier_sayisi = int(
        mask.sum()
    )

    return (
        H,
        iyi,
        kp_ref,
        kp_resim,
        detector_tipi,
        inlier_sayisi
    )


# =========================================================
# REFERANSIN KÖŞELERİNİ FOTOĞRAFA TAŞI
# =========================================================

def koseleri_bul(
    referans,
    H
):

    h, w = referans.shape[:2]

    ref_koseler = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ]
    ).reshape(-1, 1, 2)

    bulunan = cv2.perspectiveTransform(
        ref_koseler,
        H
    )

    return bulunan.reshape(
        4,
        2
    )


# =========================================================
# DEBUG EŞLEŞME RESMİ
# =========================================================

def match_debug_resmi(
    referans,
    resim,
    kp_ref,
    kp_resim,
    matchler,
    maksimum=60
):

    if not matchler:

        return None

    secilen = sorted(
        matchler,
        key=lambda m: m.distance
    )[:maksimum]

    debug = cv2.drawMatches(
        referans,
        kp_ref,
        resim,
        kp_resim,
        secilen,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return debug


# =========================================================
# ANA FONKSİYON
# =========================================================

def kart_tespit_et_ve_duzelt(
    resim,
    referans_yolu="referans_kimlik.jpg"
):
    """
    ANA AŞAMA

    1. Referans kartı yükle
    2. SIFT/ORB feature matching
    3. Homography
    4. Kart köşelerini hesapla
    5. Perspective warp
    """

    sonuc = {

        "basarili": False,

        "kart": None,

        "debug_resmi": None,

        "match_debug": None,

        "koseler": None,

        "iyi_eslesme": 0,

        "inlier": 0,

        "detector": "",

        "mesaj": ""
    }

    # ---------------------------------------------
    # Gelen fotoğraf
    # ---------------------------------------------

    if resim is None:

        sonuc[
            "mesaj"
        ] = "Görüntü okunamadı."

        return sonuc

    # ---------------------------------------------
    # Referans
    # ---------------------------------------------

    if not os.path.exists(
        referans_yolu
    ):

        sonuc[
            "mesaj"
        ] = (
            f"Referans bulunamadı: "
            f"{referans_yolu}"
        )

        return sonuc

    referans = cv2.imread(
        referans_yolu
    )

    if referans is None:

        sonuc[
            "mesaj"
        ] = "Referans kimlik görüntüsü okunamadı."

        return sonuc

    # ---------------------------------------------
    # Çok büyük resmi küçült
    #
    # Sonra köşeleri orijinal ölçeğe geri taşıyoruz.
    # ---------------------------------------------

    orijinal = resim.copy()

    h, w = resim.shape[:2]

    max_boyut = 1800

    olcek = 1.0

    if max(h, w) > max_boyut:

        olcek = max_boyut / max(
            h,
            w
        )

        calisma = cv2.resize(
            resim,
            None,
            fx=olcek,
            fy=olcek,
            interpolation=cv2.INTER_AREA
        )

    else:

        calisma = resim.copy()

    # ---------------------------------------------
    # MATCH
    # ---------------------------------------------

    (
        H,
        iyi,
        kp_ref,
        kp_resim,
        detector_tipi,
        inlier_sayisi
    ) = feature_eslestir(
        referans,
        calisma
    )

    sonuc[
        "iyi_eslesme"
    ] = len(iyi)

    sonuc[
        "inlier"
    ] = inlier_sayisi

    sonuc[
        "detector"
    ] = detector_tipi

    # ---------------------------------------------
    # Match debug
    # ---------------------------------------------

    sonuc[
        "match_debug"
    ] = match_debug_resmi(
        referans,
        calisma,
        kp_ref,
        kp_resim,
        iyi
    )

    # ---------------------------------------------
    # Yetersiz eşleşme
    # ---------------------------------------------

    if H is None:

        sonuc[
            "mesaj"
        ] = (
            f"Yeterli özellik eşleşmesi yok. "
            f"İyi eşleşme: {len(iyi)}"
        )

        return sonuc

    # RANSAC gerçekten kaç eşleşmeyi kabul etti?
    if inlier_sayisi < 8:

        sonuc[
            "mesaj"
        ] = (
            f"Homography güvenilir değil. "
            f"Inlier: {inlier_sayisi}"
        )

        return sonuc

    # ---------------------------------------------
    # Kart köşelerini bul
    # ---------------------------------------------

    koseler = koseleri_bul(
        referans,
        H
    )

    # Küçülttüysek orijinale taşı
    if olcek != 1.0:

        koseler = koseler / olcek

    # ---------------------------------------------
    # Köşeler mantıklı mı?
    # ---------------------------------------------

    if not koseler_gecerli_mi(
        koseler,
        orijinal.shape
    ):

        sonuc[
            "mesaj"
        ] = (
            "Feature eşleşmesi bulundu ancak "
            "hesaplanan kart sınırları mantıklı değil."
        )

        return sonuc

    # ---------------------------------------------
    # Debug polygon
    # ---------------------------------------------

    debug = orijinal.copy()

    polygon = koseler.astype(
        np.int32
    ).reshape(-1, 1, 2)

    cv2.polylines(
        debug,
        [polygon],
        True,
        (0, 255, 0),
        5,
        cv2.LINE_AA
    )

    # Köşelere numara koy
    for i, nokta in enumerate(
        koseler.astype(np.int32)
    ):

        x, y = nokta

        cv2.circle(
            debug,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug,
            str(i + 1),
            (
                x + 12,
                y - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ---------------------------------------------
    # Perspective warp
    # ---------------------------------------------

    kart = perspektif_duzelt(
        orijinal,
        koseler
    )

    sonuc[
        "basarili"
    ] = True

    sonuc[
        "kart"
    ] = kart

    sonuc[
        "debug_resmi"
    ] = debug

    sonuc[
        "koseler"
    ] = koseler

    sonuc[
        "mesaj"
    ] = "Kimlik başarıyla tespit edildi."

    return sonuc