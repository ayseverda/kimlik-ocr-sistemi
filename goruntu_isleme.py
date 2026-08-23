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
# DETECTOR
# =========================================================

def detector_getir():
    global _DETECTOR
    global _DETECTOR_TIPI

    if _DETECTOR is not None:
        return (
            _DETECTOR,
            _DETECTOR_TIPI
        )

    if hasattr(
        cv2,
        "SIFT_create"
    ):
        _DETECTOR = cv2.SIFT_create(
            nfeatures=5500,
            contrastThreshold=0.018,
            edgeThreshold=12,
            sigma=1.6
        )

        _DETECTOR_TIPI = "SIFT"

    else:
        _DETECTOR = cv2.ORB_create(
            nfeatures=7000,
            scaleFactor=1.2,
            nlevels=10,
            fastThreshold=5
        )

        _DETECTOR_TIPI = "ORB"

    return (
        _DETECTOR,
        _DETECTOR_TIPI
    )


def clahe_getir():
    global _CLAHE

    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    return _CLAHE


def matcher_getir(
    detector_tipi
):
    global _FLANN

    if detector_tipi == "SIFT":
        if _FLANN is None:
            _FLANN = cv2.FlannBasedMatcher(
                {
                    "algorithm": 1,
                    "trees": 5
                },
                {
                    "checks": 60
                }
            )

        return _FLANN

    return cv2.BFMatcher(
        cv2.NORM_HAMMING
    )


# =========================================================
# REFERANS
# =========================================================

def referansi_hazirla(
    belge_tipi,
    yol
):
    global _REFERANS_CACHE

    if not os.path.exists(yol):
        return (
            None,
            None,
            None,
            f"Referans bulunamadı: {yol}"
        )

    abs_yol = os.path.abspath(yol)

    mtime = os.path.getmtime(
        abs_yol
    )

    cache = _REFERANS_CACHE.get(
        belge_tipi
    )

    if (
        cache is not None
        and
        cache.get("yol") == abs_yol
        and
        cache.get("mtime") == mtime
    ):
        return (
            cache["resim"],
            cache["kp"],
            cache["des"],
            None
        )

    resim = cv2.imread(
        abs_yol
    )

    if resim is None:
        return (
            None,
            None,
            None,
            "Referans resmi okunamadı."
        )

    gri = cv2.cvtColor(
        resim,
        cv2.COLOR_BGR2GRAY
    )

    gri = clahe_getir().apply(
        gri
    )

    detector, _ = detector_getir()

    kp, des = detector.detectAndCompute(
        gri,
        None
    )

    if des is None:
        return (
            None,
            None,
            None,
            "Referansta feature bulunamadı."
        )

    _REFERANS_CACHE[
        belge_tipi
    ] = {
        "yol":
            abs_yol,

        "mtime":
            mtime,

        "resim":
            resim,

        "kp":
            kp,

        "des":
            des
    }

    return (
        resim,
        kp,
        des,
        None
    )


# =========================================================
# FEATURE MATCH
# =========================================================

def feature_eslestir(
    kp_ref,
    des_ref,
    resim
):
    detector, detector_tipi = detector_getir()

    gri = cv2.cvtColor(
        resim,
        cv2.COLOR_BGR2GRAY
    )

    gri = clahe_getir().apply(
        gri
    )

    kp_resim, des_resim = detector.detectAndCompute(
        gri,
        None
    )

    if des_resim is None:
        return (
            None,
            [],
            kp_resim,
            detector_tipi,
            0
        )

    matcher = matcher_getir(
        detector_tipi
    )

    try:
        raw = matcher.knnMatch(
            des_ref,
            des_resim,
            k=2
        )

    except cv2.error:
        return (
            None,
            [],
            kp_resim,
            detector_tipi,
            0
        )

    if detector_tipi == "SIFT":
        ratio = 0.74
    else:
        ratio = 0.80

    iyi = []

    for pair in raw:
        if len(pair) != 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            ratio * n.distance
        ):
            iyi.append(m)

    if len(iyi) < 8:
        return (
            None,
            iyi,
            kp_resim,
            detector_tipi,
            0
        )

    src = np.float32(
        [
            kp_ref[
                m.queryIdx
            ].pt
            for m in iyi
        ]
    ).reshape(
        -1,
        1,
        2
    )

    dst = np.float32(
        [
            kp_resim[
                m.trainIdx
            ].pt
            for m in iyi
        ]
    ).reshape(
        -1,
        1,
        2
    )

    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        5.0
    )

    if (
        H is None
        or
        mask is None
    ):
        return (
            None,
            iyi,
            kp_resim,
            detector_tipi,
            0
        )

    inlier = int(
        mask.sum()
    )

    return (
        H,
        iyi,
        kp_resim,
        detector_tipi,
        inlier
    )


# =========================================================
# SKOR
# =========================================================

def eslesme_skoru(
    iyi,
    inlier
):
    if iyi <= 0:
        return 0.0

    oran = (
        inlier
        /
        iyi
    )

    return (
        inlier * 3.5
        +
        iyi * 0.30
        +
        oran * 100
    )


# =========================================================
# TEK REFERANS TEST
# =========================================================

def tek_referansi_test_et(
    belge_tipi,
    referans_yolu,
    calisma
):
    (
        referans,
        kp_ref,
        des_ref,
        hata
    ) = referansi_hazirla(
        belge_tipi,
        referans_yolu
    )

    if hata:
        return {
            "belge_tipi":
                belge_tipi,

            "basarili":
                False,

            "skor":
                0.0,

            "hata":
                hata
        }

    (
        H,
        iyi,
        kp_resim,
        detector_tipi,
        inlier
    ) = feature_eslestir(
        kp_ref,
        des_ref,
        calisma
    )

    iyi_sayisi = len(iyi)

    inlier_orani = (
        inlier
        /
        iyi_sayisi
        if iyi_sayisi
        else 0.0
    )

    skor = eslesme_skoru(
        iyi_sayisi,
        inlier
    )

    if belge_tipi == "eski_tc":
        basarili = (
            H is not None
            and
            inlier >= 7
            and
            inlier_orani >= 0.10
        )

    elif belge_tipi == "gocmen":
        basarili = (
            H is not None
            and
            inlier >= 10
            and
            inlier_orani >= 0.12
        )

    else:
        basarili = (
            H is not None
            and
            inlier >= 8
        )

    return {
        "belge_tipi":
            belge_tipi,

        "basarili":
            basarili,

        "referans":
            referans,

        "kp_ref":
            kp_ref,

        "kp_resim":
            kp_resim,

        "H":
            H,

        "iyi":
            iyi,

        "iyi_eslesme":
            iyi_sayisi,

        "inlier":
            inlier,

        "inlier_orani":
            inlier_orani,

        "skor":
            skor,

        "detector":
            detector_tipi
    }


# =========================================================
# BELGE TİPİ
# =========================================================

def belge_tipini_bul(
    resim
):
    h, w = resim.shape[:2]

    olcek = 1.0

    if max(h, w) > MAX_CALISMA_BOYUTU:
        olcek = (
            MAX_CALISMA_BOYUTU
            /
            max(h, w)
        )

        calisma = cv2.resize(
            resim,
            None,
            fx=olcek,
            fy=olcek,
            interpolation=cv2.INTER_AREA
        )

    else:
        calisma = resim

    sonuclar = {}

    for tip, yol in BELGE_REFERANSLARI.items():
        sonuclar[tip] = tek_referansi_test_et(
            tip,
            yol,
            calisma
        )

    adaylar = [
        x
        for x in sonuclar.values()
        if x.get(
            "basarili",
            False
        )
    ]

    if not adaylar:
        return {
            "basarili":
                False,

            "belge_tipi":
                None,

            "sonuclar":
                sonuclar,

            "calisma":
                calisma,

            "olcek":
                olcek
        }

    en_iyi = max(
        adaylar,
        key=lambda x: x["skor"]
    )

    return {
        "basarili":
            True,

        "belge_tipi":
            en_iyi["belge_tipi"],

        "en_iyi":
            en_iyi,

        "sonuclar":
            sonuclar,

        "calisma":
            calisma,

        "olcek":
            olcek
    }


# =========================================================
# REFERANS KÖŞELERİ
# =========================================================

def koseleri_bul(
    referans,
    H
):
    h, w = referans.shape[:2]

    # ÇOK ÖNEMLİ:
    # Sıra referans yönünü temsil ediyor.
    #
    # TL -> TR -> BR -> BL
    #
    # BUNU SONRADAN SIRALAMIYORUZ.
    ref_koseler = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ]
    ).reshape(
        -1,
        1,
        2
    )

    koseler = cv2.perspectiveTransform(
        ref_koseler,
        H
    ).reshape(
        4,
        2
    )

    return koseler


# =========================================================
# GEOMETRİ KONTROL
# =========================================================

def koseler_gecerli_mi(
    koseler,
    shape,
    belge_tipi
):
    if koseler is None:
        return False

    koseler = np.asarray(
        koseler,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(
            koseler
        )
    ):
        return False

    h, w = shape[:2]

    kontur = (
        koseler
        .astype(np.int32)
        .reshape(
            -1,
            1,
            2
        )
    )

    alan = abs(
        cv2.contourArea(
            kontur
        )
    )

    alan_orani = (
        alan
        /
        max(
            1,
            h * w
        )
    )

    # Eski TC için daha toleranslı:
    # eğik/yan/fotoğraf şeklinde gelebiliyor.
    if belge_tipi == "eski_tc":
        if not (
            0.01
            <=
            alan_orani
            <=
            2.0
        ):
            return False

        tol_x = w * 0.60
        tol_y = h * 0.60

    elif belge_tipi == "gocmen":
        if not (
            0.01
            <=
            alan_orani
            <=
            2.5
        ):
            return False

        tol_x = w * 0.80
        tol_y = h * 0.80

    else:
        if not (
            0.02
            <=
            alan_orani
            <=
            1.3
        ):
            return False

        tol_x = w * 0.25
        tol_y = h * 0.25

    if np.any(
        koseler[:, 0] < -tol_x
    ):
        return False

    if np.any(
        koseler[:, 0] > w + tol_x
    ):
        return False

    if np.any(
        koseler[:, 1] < -tol_y
    ):
        return False

    if np.any(
        koseler[:, 1] > h + tol_y
    ):
        return False

    # Karşılıklı kenar uzunlukları
    tl, tr, br, bl = koseler

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

    if min(
        ust,
        alt,
        sol,
        sag
    ) < 20:
        return False

    genislik = (
        ust + alt
    ) / 2.0

    yukseklik = (
        sol + sag
    ) / 2.0

    aspect = (
        max(
            genislik,
            yukseklik
        )
        /
        min(
            genislik,
            yukseklik
        )
    )

    if belge_tipi == "eski_tc":
        return (
            1.05
            <=
            aspect
            <=
            2.10
        )

    if belge_tipi == "tc":
        return (
            1.15
            <=
            aspect
            <=
            2.20
        )

    if belge_tipi == "gocmen":
        return (
            1.00
            <=
            aspect
            <=
            3.30
        )

    return False


# =========================================================
# PERSPEKTİF DÜZELT
# =========================================================

def perspektif_duzelt(
    resim,
    koseler,
    belge_tipi
):
    """
    Kritik nokta:

    koseler homography'den şu sırayla geliyor:

        referans TL
        referans TR
        referans BR
        referans BL

    Kart görüntüde 90 derece dönmüş olsa bile bu
    correspondence doğru.

    O yüzden KÖŞELERİ TEKRAR SIRALAMIYORUZ.
    """

    src = np.asarray(
        koseler,
        dtype=np.float32
    )

    if belge_tipi == "eski_tc":
        hedef_w = 750
        hedef_h = 1050

    elif belge_tipi == "gocmen":
        hedef_w = 800
        hedef_h = 1100

    else:
        hedef_w = 1000
        hedef_h = 630

    dst = np.float32(
        [
            [0, 0],
            [hedef_w - 1, 0],
            [hedef_w - 1, hedef_h - 1],
            [0, hedef_h - 1]
        ]
    )

    M = cv2.getPerspectiveTransform(
        src,
        dst
    )

    duzeltilmis = cv2.warpPerspective(
        resim,
        M,
        (
            hedef_w,
            hedef_h
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return duzeltilmis


# =========================================================
# DEBUG
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
        key=lambda x: x.distance
    )[:maksimum]

    return cv2.drawMatches(
        referans,
        kp_ref,
        resim,
        kp_resim,
        secilen,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    )


def kart_debug_resmi(
    resim,
    koseler
):
    debug = resim.copy()

    poly = (
        np.asarray(
            koseler,
            dtype=np.int32
        )
        .reshape(
            -1,
            1,
            2
        )
    )

    cv2.polylines(
        debug,
        [poly],
        True,
        (0, 255, 0),
        5,
        cv2.LINE_AA
    )

    # Referans köşe isimleri
    isimler = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for isim, nokta in zip(
        isimler,
        koseler
    ):
        x, y = map(
            int,
            nokta
        )

        cv2.circle(
            debug,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            debug,
            isim,
            (
                x + 10,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    return debug


# =========================================================
# ANA
# =========================================================

def kart_tespit_et_ve_duzelt(
    resim,
    referans_yolu=None,
    debug_match=False,
    debug_kart=False,
    otomatik_belge_tipi=True
):
    sonuc = {
        "basarili":
            False,

        "kart":
            None,

        "belge_tipi":
            None,

        "debug_resmi":
            None,

        "match_debug":
            None,

        "koseler":
            None,

        "fallback":
            False,

        "duzeltildi":
            False,

        "iyi_eslesme":
            0,

        "inlier":
            0,

        "inlier_orani":
            0.0,

        "skor":
            0.0,

        "mesaj":
            ""
    }

    if resim is None:
        sonuc["mesaj"] = (
            "Görüntü bulunamadı."
        )

        return sonuc

    tip_sonuc = belge_tipini_bul(
        resim
    )

    if not tip_sonuc.get(
        "basarili",
        False
    ):
        sonuc["mesaj"] = (
            "Belge tipi bulunamadı."
        )

        return sonuc

    belge_tipi = tip_sonuc[
        "belge_tipi"
    ]

    en_iyi = tip_sonuc[
        "en_iyi"
    ]

    sonuc[
        "belge_tipi"
    ] = belge_tipi

    sonuc[
        "iyi_eslesme"
    ] = en_iyi.get(
        "iyi_eslesme",
        0
    )

    sonuc[
        "inlier"
    ] = en_iyi.get(
        "inlier",
        0
    )

    sonuc[
        "inlier_orani"
    ] = en_iyi.get(
        "inlier_orani",
        0.0
    )

    sonuc[
        "skor"
    ] = en_iyi.get(
        "skor",
        0.0
    )

    if debug_match:
        sonuc[
            "match_debug"
        ] = match_debug_resmi(
            en_iyi[
                "referans"
            ],
            tip_sonuc[
                "calisma"
            ],
            en_iyi[
                "kp_ref"
            ],
            en_iyi[
                "kp_resim"
            ],
            en_iyi[
                "iyi"
            ]
        )

    try:
        koseler = koseleri_bul(
            en_iyi[
                "referans"
            ],
            en_iyi[
                "H"
            ]
        )

        olcek = tip_sonuc[
            "olcek"
        ]

        if olcek != 1.0:
            koseler = (
                koseler
                /
                olcek
            )

        sonuc[
            "koseler"
        ] = koseler

    except Exception as e:
        koseler = None

    # =====================================================
    # ÖNCE NORMAL GEOMETRİ
    # =====================================================

    geometri_ok = (
        koseler is not None
        and
        koseler_gecerli_mi(
            koseler,
            resim.shape,
            belge_tipi
        )
    )

    if geometri_ok:
        try:
            kart = perspektif_duzelt(
                resim,
                koseler,
                belge_tipi
            )

            sonuc[
                "kart"
            ] = kart

            sonuc[
                "basarili"
            ] = True

            sonuc[
                "duzeltildi"
            ] = True

            sonuc[
                "fallback"
            ] = False

            sonuc[
                "mesaj"
            ] = (
                "Kart tespit edildi ve yönü düzeltildi."
            )

            if debug_kart:
                sonuc[
                    "debug_resmi"
                ] = kart_debug_resmi(
                    resim,
                    koseler
                )

            return sonuc

        except Exception:
            pass

    # =====================================================
    # ESKİ TC ÖZEL:
    #
    # Homography güçlü ise köşe kontrolü biraz başarısız
    # olsa bile perspektifi dene.
    #
    # BÖYLECE YAN KARTI RAW BIRAKMIYORUZ.
    # =====================================================

    if (
        belge_tipi == "eski_tc"
        and
        koseler is not None
        and
        en_iyi.get(
            "inlier",
            0
        ) >= 7
    ):
        try:
            kart = perspektif_duzelt(
                resim,
                koseler,
                "eski_tc"
            )

            # Warp tamamen boş/siyah çıkmadı mı?
            gri = cv2.cvtColor(
                kart,
                cv2.COLOR_BGR2GRAY
            )

            if (
                kart.size > 0
                and
                np.std(gri) > 8
            ):
                sonuc[
                    "kart"
                ] = kart

                sonuc[
                    "basarili"
                ] = True

                sonuc[
                    "duzeltildi"
                ] = True

                sonuc[
                    "fallback"
                ] = False

                sonuc[
                    "mesaj"
                ] = (
                    "Eski T.C. kimlik güçlü SIFT eşleşmesiyle "
                    "yönü düzeltilerek işlendi."
                )

                if debug_kart:
                    sonuc[
                        "debug_resmi"
                    ] = kart_debug_resmi(
                        resim,
                        koseler
                    )

                return sonuc

        except Exception:
            pass

    # =====================================================
    # DİĞER BELGELER İÇİN FALLBACK
    # =====================================================

    if (
        belge_tipi != "eski_tc"
        and
        en_iyi.get(
            "inlier",
            0
        ) >= 8
    ):
        sonuc[
            "basarili"
        ] = True

        sonuc[
            "kart"
        ] = resim.copy()

        sonuc[
            "fallback"
        ] = True

        sonuc[
            "mesaj"
        ] = (
            "Belge tipi bulundu fakat perspektif "
            "düzeltilemedi."
        )

        return sonuc

    sonuc[
        "mesaj"
    ] = (
        "Kimlik güvenilir şekilde düzeltilemedi."
    )

    return sonuc