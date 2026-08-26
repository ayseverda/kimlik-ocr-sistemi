import os
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = ModuleSpec("cv2", loader=None)
    cv2_stub.error = type("error", (Exception,), {})
    sys.modules["cv2"] = cv2_stub

import goruntu_isleme as gi


def aday(belge_tipi, skor, referans):
    return {
        "belge_tipi": belge_tipi,
        "basarili": True,
        "referans": referans,
        "H": object(),
        "iyi_eslesme": 20,
        "inlier": 15,
        "inlier_orani": 0.75,
        "skor": skor,
    }


class GoruntuIslemeTestleri(unittest.TestCase):
    def setUp(self):
        self.resim = np.zeros((100, 160, 3), dtype=np.uint8)
        self.koseler = np.array([[0, 0], [159, 0], [159, 99], [0, 99]], dtype=np.float32)
        self.kaliteli_kart = self._kaliteli_kart_uret()

    @staticmethod
    def _kaliteli_kart_uret():
        """Yazi satirlari ve fotograf bolgesi karta yayilmis sentetik belge."""
        kart = np.full((1100, 800, 3), 235, dtype=np.uint8)
        for y in range(70, 1030, 85):
            kart[y:y + 8, 45:755] = 35
        kart[120:520, 520:745] = 115
        for y in range(130, 510, 24):
            kart[y:y + 7, 530:735] = 55 if (y // 24) % 2 else 190
        return kart

    @staticmethod
    def _bozuk_cizgisel_warp_uret():
        """Bos zemin + diagonal serit bozulmasini taklit eder."""
        kart = np.full((1100, 800, 3), 245, dtype=np.uint8)
        for y in range(kart.shape[0]):
            x = int(140 * (1.0 - y / (kart.shape[0] - 1)))
            kart[y, max(0, x - 28):min(kart.shape[1], x + 28)] = 15
        return kart

    @staticmethod
    def _bozuk_radyal_warp_uret(w=750, h=1050):
        """Ekran goruntusundeki tek koseden yayilan fan warp'i taklit eder."""
        yy, xx = np.indices((h, w), dtype=np.float64)
        merkez_x, merkez_y = -18.0, h + 20.0
        aci = np.arctan2(merkez_y - yy, xx - merkez_x)
        # Yumusal sinuzoidal bantlar, gercek INTER_CUBIC warp'taki bulanmis
        # siyah/beyaz isinlari sert piksel merdivenlerinden daha iyi taklit eder.
        isinlar = np.sin(aci * 54.0)
        gri = np.clip(131.0 + 107.0 * isinlar, 0, 255).astype(np.uint8)
        return np.repeat(gri[:, :, None], 3, axis=2)

    @staticmethod
    def _affine_eslesmeli_gocmen_adayi():
        referans = np.zeros((100, 160, 3), dtype=np.uint8)
        noktalar = [
            (0.0, 0.0), (159.0, 0.0), (159.0, 99.0), (0.0, 99.0),
            (80.0, 0.0), (159.0, 50.0), (80.0, 99.0), (0.0, 50.0),
        ]
        sonuc = aday("gocmen", 120.0, referans)
        sonuc["kp_ref"] = [types.SimpleNamespace(pt=nokta) for nokta in noktalar]
        sonuc["kp_resim"] = [types.SimpleNamespace(pt=nokta) for nokta in noktalar]
        sonuc["iyi"] = [
            types.SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(len(noktalar))
        ]
        return sonuc

    def _gocmen_tablolu_kart_uret(self, ayirici_x=392):
        kart = self.kaliteli_kart.copy()
        kart[350:1035, ayirici_x:ayirici_x + 5] = 28
        for y in range(390, 1030, 80):
            kart[y:y + 4, 35:765] = 28
        return kart

    def test_referans_yolu_calisma_dizininden_bagimsizdir(self):
        beklenen = Path(gi.__file__).resolve().parent / "referans_kimlik.jpg"
        self.assertEqual(Path(gi.referans_yolunu_coz("referans_kimlik.jpg")), beklenen)

    def test_gocmen_warp_hedefi_referans_oranini_korur(self):
        sahte_kart = np.zeros((1100, 720, 3), dtype=np.uint8)
        with (
            patch.object(gi.cv2, "getPerspectiveTransform", return_value=object(), create=True),
            patch.object(gi.cv2, "warpPerspective", return_value=sahte_kart, create=True) as warp,
            patch.object(gi.cv2, "INTER_CUBIC", 1, create=True),
            patch.object(gi.cv2, "BORDER_REPLICATE", 2, create=True),
        ):
            sonuc = gi.perspektif_duzelt(self.resim, self.koseler, "gocmen")

        self.assertIs(sahte_kart, sonuc)
        self.assertEqual((720, 1100), warp.call_args.args[2])

    def test_eksik_referans_hatasi_mutlak_yolu_icerir(self):
        eksik = "kesinlikle-olmayan-referans.jpg"
        _, _, _, hata = gi.referansi_hazirla("tc", eksik)

        self.assertIn("Referans bulunamadı", hata)
        self.assertIn(os.path.join(gi.MODUL_DIZINI, eksik), hata)

    def test_gecersiz_girdiler_acik_hata_dondurur(self):
        bos = gi.kart_tespit_et_ve_duzelt(np.empty((0, 3, 3), dtype=np.uint8))
        kanal = gi.kart_tespit_et_ve_duzelt(np.zeros((10, 10, 5), dtype=np.uint8))
        tip = gi.kart_tespit_et_ve_duzelt(np.zeros((10, 10, 3), dtype=np.float32))

        self.assertFalse(bos["basarili"])
        self.assertIn("boş", bos["mesaj"].lower())
        self.assertIn("kanal", kanal["mesaj"].lower())
        self.assertIn("uint8", tip["mesaj"])

    def test_debug_hatasi_basarili_kart_sonucunu_degistirmez(self):
        tc = aday("tc", 100.0, "tc-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "tc",
            "en_iyi": tc,
            "adaylar": [tc],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "_gocmen_hizli_on_tespit_dene", return_value={"basarili": False}),
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=True),
            patch.object(gi, "perspektif_duzelt", return_value=self.kaliteli_kart.copy()),
            patch.object(gi, "kart_debug_resmi", side_effect=RuntimeError("çizim hatası")),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim, debug_kart=True)

        self.assertTrue(sonuc["basarili"])
        self.assertTrue(sonuc["duzeltildi"])
        self.assertFalse(sonuc["fallback"])
        self.assertIsNotNone(sonuc["kart"])
        self.assertIn("çizim hatası", sonuc["debug_hatasi"])

    def test_ilk_aday_gecersizse_siradaki_gecerli_aday_denenir(self):
        gocmen = aday("gocmen", 120.0, "gocmen-ref")
        tc = aday("tc", 100.0, "tc-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen, tc],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", side_effect=[False, True]),
            patch.object(gi, "perspektif_duzelt", return_value=self.kaliteli_kart.copy()),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"])
        self.assertEqual(sonuc["belge_tipi"], "tc")
        self.assertEqual(sonuc["aday_sirasi"], 2)
        self.assertTrue(sonuc["fallback"])
        self.assertTrue(sonuc["duzeltildi"])
        self.assertIn("En yüksek skorlu gocmen", sonuc["mesaj"])

    def test_gocmen_homografi_gecersiz_affine_gecerliyse_kart_kabul_edilir(self):
        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        maske = np.ones((8, 1), dtype=np.uint8)

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", side_effect=[False, True]),
            patch.object(gi, "perspektif_duzelt", return_value=self._gocmen_tablolu_kart_uret()),
            patch.object(gi.cv2, "RANSAC", 8, create=True),
            patch.object(
                gi.cv2,
                "estimateAffinePartial2D",
                return_value=(affine, maske),
                create=True,
            ) as affine_tahmin,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"], sonuc)
        self.assertTrue(sonuc["fallback"])
        self.assertEqual(sonuc["duzeltme_yontemi"], "affine_partial")
        self.assertEqual(sonuc["aday_sonuclari"][0]["yontem"], "affine_partial")
        self.assertFalse(sonuc["aday_sonuclari"][0]["homografi_geometri_gecerli"])
        self.assertTrue(sonuc["aday_sonuclari"][0]["affine_geometri_gecerli"])
        self.assertTrue(sonuc["aday_sonuclari"][0]["affine_tablo_imzasi_gecerli"])
        affine_tahmin.assert_called_once()

    def test_affine_genel_kalitesi_iyi_ama_tablo_imzasiz_gocmen_reddedilir(self):
        genel_kalite, _ = gi._kart_ciktisi_kaliteli_mi(self.kaliteli_kart, "gocmen")
        self.assertTrue(genel_kalite)

        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        maske = np.ones((8, 1), dtype=np.uint8)

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", side_effect=[False, True]),
            patch.object(gi, "perspektif_duzelt", return_value=self.kaliteli_kart.copy()),
            patch.object(gi.cv2, "RANSAC", 8, create=True),
            patch.object(
                gi.cv2,
                "estimateAffinePartial2D",
                return_value=(affine, maske),
                create=True,
            ),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        aday_sonucu = sonuc["aday_sonuclari"][0]
        self.assertFalse(sonuc["basarili"])
        self.assertTrue(aday_sonucu["affine_cikti_kalitesi_gecerli"])
        self.assertFalse(aday_sonucu["affine_tablo_imzasi_gecerli"])
        self.assertIn("tablo imzasi", aday_sonucu["affine_hatasi"])

    def test_gocmen_homografi_ve_affine_gecersizse_reddedilir(self):
        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=False),
            patch.object(gi, "perspektif_duzelt") as warp,
            patch.object(gi.cv2, "RANSAC", 8, create=True),
            patch.object(
                gi.cv2,
                "estimateAffinePartial2D",
                return_value=(None, None),
                create=True,
            ) as affine_tahmin,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertIsNone(sonuc["duzeltme_yontemi"])
        self.assertIn("Homografi:", sonuc["aday_sonuclari"][0]["hata"])
        self.assertIn("Affine model bulunamadi", sonuc["aday_sonuclari"][0]["hata"])
        affine_tahmin.assert_called_once()
        warp.assert_not_called()

    def test_homografi_ve_affine_reddedilince_tablo_konturu_kabul_edilir(self):
        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }
        kontur_adayi = {
            "koseler": self.koseler.copy(),
            "kaynak": "dortgen_kontur",
            "skor": 0.9,
        }
        kontur_karti = self._gocmen_tablolu_kart_uret()
        kontur_karti[:140, :] = 25

        with (
            patch.object(
                gi, "_gocmen_hizli_on_tespit_dene", return_value={"basarili": False}
            ),
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", side_effect=[False, True]),
            patch.object(
                gi, "_gocmen_affine_koseleri_bul",
                return_value=(None, {}, "Affine inlier destegi guvenilir degil."),
            ),
            patch.object(
                gi, "_gocmen_kontur_kose_adaylari",
                return_value=([kontur_adayi], {"gecerli_aday": 1}, ""),
            ) as kontur_bul,
            patch.object(
                gi, "perspektif_duzelt",
                return_value=kontur_karti,
            ),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"], sonuc)
        self.assertEqual(sonuc["duzeltme_yontemi"], "kontur_tablo")
        self.assertEqual(sonuc["aday_sonuclari"][0]["kontur_deneme_sayisi"], 1)
        self.assertEqual(sonuc["aday_sonuclari"][0]["kontur_kaynak"], "dortgen_kontur")
        self.assertTrue(sonuc["aday_sonuclari"][0]["kontur_tablo_imzasi_gecerli"])
        self.assertTrue(sonuc["aday_sonuclari"][0]["kontur_form_imzasi_gecerli"])
        kontur_bul.assert_called_once_with(self.resim, azami_aday=3)

    def test_kontur_fallback_birlesik_sayfayi_reddedip_on_karti_secer(self):
        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }
        birlesik_sayfa_koseleri = self.koseler.copy()
        on_kart_koseleri = self.koseler.copy() + np.asarray([4.0, 0.0])
        kontur_adaylari = [
            {
                "koseler": birlesik_sayfa_koseleri,
                "kaynak": "kenar_zarfi",
                "skor": 1.0,
            },
            {
                "koseler": on_kart_koseleri,
                "kaynak": "dortgen_kontur",
                "skor": 0.9,
            },
        ]

        # Birlesik sayfa hem genel tabloyu hem koyu ust bantli form kontrolunu
        # gecebilir: satirlar soldaki on karttan, kuvvetli dikey cizgi ise iki
        # kartin sinirindan gelir. Ancak satirlar bu sahte ayiricinin saginda
        # ayni y konumlarinda devam etmez. Gercek on kartta iki taraf da vardir.
        birlesik_sayfa = np.full((1100, 800, 3), 235, dtype=np.uint8)
        birlesik_sayfa[:140, :] = 25
        birlesik_sayfa[350:1035, 392:397] = 28
        for y in range(390, 1030, 80):
            birlesik_sayfa[y:y + 4, 35:385] = 28
        on_kart = birlesik_sayfa.copy()
        for y in range(390, 1030, 80):
            on_kart[y:y + 4, 35:765] = 28
        tablo_ok, _ = gi._gocmen_tablo_imzasi_gecerli_mi(birlesik_sayfa)
        birlesik_form_ok, _ = gi._gocmen_hizli_form_imzasi_gecerli_mi(
            birlesik_sayfa
        )
        birlesik_uzamsal_ok, _ = gi._gocmen_tek_form_uzamsal_imzasi_gecerli_mi(
            birlesik_sayfa
        )
        on_uzamsal_ok, _ = gi._gocmen_tek_form_uzamsal_imzasi_gecerli_mi(on_kart)
        self.assertTrue(tablo_ok)
        self.assertTrue(birlesik_form_ok)
        self.assertFalse(birlesik_uzamsal_ok)
        self.assertTrue(on_uzamsal_ok)

        basarisiz = {
            "basarili": False,
            "kart": None,
            "geometri_gecerli": False,
            "cikti_kalitesi_gecerli": False,
            "kalite_metrikleri": {},
            "hata": "Kart geometrisi guvenilir degil.",
        }
        birlesik_deneme = {
            "basarili": True,
            "kart": birlesik_sayfa,
            "geometri_gecerli": True,
            "cikti_kalitesi_gecerli": True,
            "kalite_metrikleri": {"gecerli": True},
            "hata": "",
        }
        on_kart_deneme = {
            **birlesik_deneme,
            "kart": on_kart,
        }

        # A4 dis cercevesi gercek bir dortgen ve gocmen oranina yakin olsa da
        # ayni uzamsal kapi SIFT-oncesi hizli yolu da kapatmalidir.
        a4_koseleri = np.asarray(
            [[50, 5], [110, 5], [110, 95], [50, 95]], dtype=np.float32
        )
        a4_adayi = {
            "koseler": a4_koseleri,
            "kaynak": "dortgen_kontur",
            "skor": 1.0,
        }
        with (
            patch.object(
                gi, "_gocmen_kontur_kose_adaylari",
                return_value=([a4_adayi], {"gecerli_aday": 1}, ""),
            ),
            patch.object(
                gi, "_koselerle_karti_duzeltmeyi_dene",
                return_value=birlesik_deneme,
            ),
        ):
            on_tespit = gi._gocmen_hizli_on_tespit_dene(self.resim)

        self.assertFalse(on_tespit["basarili"])
        self.assertEqual(on_tespit["siki_aday_sayisi"], 1)
        self.assertEqual(on_tespit["dogrulanan_aday_sayisi"], 0)
        self.assertIn("uzamsal", on_tespit["hata"])

        with (
            patch.object(
                gi, "_gocmen_hizli_on_tespit_dene", return_value={"basarili": False}
            ),
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(
                gi, "_gocmen_affine_koseleri_bul",
                return_value=(None, {}, "Affine inlier destegi guvenilir degil."),
            ),
            patch.object(
                gi, "_gocmen_kontur_kose_adaylari",
                return_value=(kontur_adaylari, {"gecerli_aday": 2}, ""),
            ) as kontur_bul,
            patch.object(
                gi, "_koselerle_karti_duzeltmeyi_dene",
                side_effect=[basarisiz, birlesik_deneme, on_kart_deneme],
            ) as warp_dene,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"], sonuc)
        self.assertEqual(sonuc["duzeltme_yontemi"], "kontur_tablo")
        self.assertIs(sonuc["kart"], on_kart)
        np.testing.assert_array_equal(sonuc["koseler"], on_kart_koseleri)
        aday_sonucu = sonuc["aday_sonuclari"][0]
        self.assertEqual(aday_sonucu["kontur_deneme_sayisi"], 2)
        self.assertEqual(aday_sonucu["kontur_kaynak"], "dortgen_kontur")
        self.assertTrue(aday_sonucu["kontur_tablo_imzasi_gecerli"])
        self.assertTrue(aday_sonucu["kontur_form_imzasi_gecerli"])
        self.assertEqual(warp_dene.call_count, 3)
        kontur_bul.assert_called_once_with(self.resim, azami_aday=3)

    def test_gocmen_homografi_basariliysa_affine_hic_cagrilmaz(self):
        gocmen = self._affine_eslesmeli_gocmen_adayi()
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "_gocmen_hizli_on_tespit_dene", return_value={"basarili": False}),
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=True),
            patch.object(gi, "perspektif_duzelt", return_value=self.kaliteli_kart.copy()),
            patch.object(
                gi.cv2,
                "estimateAffinePartial2D",
                side_effect=AssertionError("affine mutlu yolda calismamali"),
                create=True,
            ) as affine_tahmin,
            patch.object(
                gi,
                "_gocmen_kontur_kose_adaylari",
                side_effect=AssertionError("kontur mutlu yolda calismamali"),
            ) as kontur_tahmin,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"])
        self.assertFalse(sonuc["fallback"])
        self.assertEqual(sonuc["duzeltme_yontemi"], "homografi")
        self.assertEqual(sonuc["aday_sonuclari"][0]["denenen_yontemler"], ["homografi"])
        affine_tahmin.assert_not_called()
        kontur_tahmin.assert_not_called()

    def test_kenar_zarfi_formu_beyaz_marjlardan_ayirir(self):
        kenarlar = np.zeros((120, 80), dtype=np.uint8)
        kenarlar[10:111, 12] = 255
        kenarlar[10:111, 67] = 255
        kenarlar[10, 12:68] = 255
        kenarlar[110, 12:68] = 255

        koseler = gi._gocmen_kenar_zarfi_adayi(kenarlar, (240, 160, 3))

        self.assertIsNotNone(koseler)
        self.assertEqual(koseler.shape, (4, 2))
        self.assertGreater(koseler[0, 0], 0)
        self.assertLess(koseler[1, 0], 160)
        metrikler = gi._gocmen_kontur_aday_metrikleri(koseler, (240, 160, 3))
        self.assertTrue(0.43 <= metrikler["en_boy_orani"] <= 0.90, metrikler)

    def test_hizli_gocmen_imzasi_koyu_bant_ve_cok_satir_ister(self):
        yalniz_tablo = self._gocmen_tablolu_kart_uret()
        koyu_bantli = yalniz_tablo.copy()
        koyu_bantli[:140, :] = 25

        reddedildi, zayif_metrikler = gi._gocmen_hizli_form_imzasi_gecerli_mi(
            yalniz_tablo
        )
        kabul, guclu_metrikler = gi._gocmen_hizli_form_imzasi_gecerli_mi(
            koyu_bantli
        )

        self.assertFalse(reddedildi, zayif_metrikler)
        self.assertTrue(kabul, guclu_metrikler)
        self.assertGreaterEqual(guclu_metrikler["tablo"]["yatay_cizgi_grubu"], 7)
        self.assertGreaterEqual(guclu_metrikler["ust_bant_koyu_orani"], 0.38)

    def test_sift_oncesi_kenar_zarfi_belge_tipi_sayilmaz(self):
        koseler = np.asarray(
            [[50, 5], [110, 5], [110, 95], [50, 95]], dtype=np.float32
        )
        aday = {"koseler": koseler, "kaynak": "kenar_zarfi", "skor": 1.0}
        with (
            patch.object(
                gi, "_gocmen_kontur_kose_adaylari",
                return_value=([aday], {"gecerli_aday": 1}, ""),
            ),
            patch.object(gi, "_koselerle_karti_duzeltmeyi_dene") as warp,
        ):
            sonuc = gi._gocmen_hizli_on_tespit_dene(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertEqual(sonuc["siki_aday_sayisi"], 0)
        self.assertIn("kapali", sonuc["hata"])
        warp.assert_not_called()

    def test_sift_oncesi_birden_fazla_dortgen_belirsiz_sayilir(self):
        adaylar = [
            {
                "koseler": np.asarray(
                    [[3, 5], [63, 5], [63, 95], [3, 95]], dtype=np.float32
                ),
                "kaynak": "dortgen_kontur", "skor": 1.0,
            },
            {
                "koseler": np.asarray(
                    [[96, 5], [156, 5], [156, 95], [96, 95]], dtype=np.float32
                ),
                "kaynak": "dortgen_kontur", "skor": 0.9,
            },
        ]
        deneme = {
            "basarili": True,
            "kart": self._gocmen_tablolu_kart_uret(),
            "kalite_metrikleri": {"gecerli": True},
            "geometri_gecerli": True,
            "cikti_kalitesi_gecerli": True,
            "hata": "",
        }
        with (
            patch.object(
                gi, "_gocmen_kontur_kose_adaylari",
                return_value=(adaylar, {"gecerli_aday": 2}, ""),
            ),
            patch.object(gi, "_koselerle_karti_duzeltmeyi_dene", return_value=deneme),
            patch.object(
                gi, "_gocmen_tek_form_uzamsal_imzasi_gecerli_mi",
                return_value=(True, {"gecerli": True}),
            ),
        ):
            sonuc = gi._gocmen_hizli_on_tespit_dene(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertEqual(sonuc["dogrulanan_aday_sayisi"], 2)
        self.assertIn("Birden fazla", sonuc["hata"])

    def test_siki_kontur_on_tespiti_sifti_tamamen_atlar(self):
        kart = self._gocmen_tablolu_kart_uret()
        on_tespit = {
            "basarili": True,
            "kart": kart,
            "koseler": self.koseler,
            "kaynak": "kenar_zarfi",
            "kontur_metrikleri": {"gecerli_aday": 1},
            "form_metrikleri": {"gecerli": True},
            "kalite_metrikleri": {"gecerli": True},
        }

        with (
            patch.object(gi, "_gocmen_hizli_on_tespit_dene", return_value=on_tespit),
            patch.object(
                gi, "belge_tipini_bul",
                side_effect=AssertionError("siki on tespitte SIFT calismamali"),
            ) as sift,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"])
        self.assertEqual(sonuc["belge_tipi"], "gocmen")
        self.assertEqual(sonuc["duzeltme_yontemi"], "kontur_on_tespit")
        self.assertEqual(sonuc["iyi_eslesme"], 0)
        sift.assert_not_called()

    def test_sift_tespit_goruntusu_1300_uzun_kenarla_sinirlanir(self):
        buyuk = np.zeros((3000, 2000, 3), dtype=np.uint8)
        kucuk = np.zeros((1300, 867, 3), dtype=np.uint8)
        basarisiz = {
            "belge_tipi": "gocmen", "basarili": False, "skor": 0.0,
        }

        with (
            patch.object(gi.cv2, "INTER_AREA", 3, create=True),
            patch.object(gi.cv2, "resize", return_value=kucuk, create=True) as resize,
            patch.object(gi, "resim_descriptor_cikar", return_value=([], "des", "SIFT")),
            patch.object(gi, "BELGE_REFERANSLARI", {"gocmen": "ref.jpg"}),
            patch.object(gi, "tek_referansi_test_et", return_value=basarisiz),
        ):
            sonuc = gi.belge_tipini_bul(buyuk)

        self.assertFalse(sonuc["basarili"])
        self.assertEqual(sonuc["calisma"].shape[:2], (1300, 867))
        self.assertAlmostEqual(resize.call_args.kwargs["fx"], 1300 / 3000)
        self.assertAlmostEqual(resize.call_args.kwargs["fy"], 1300 / 3000)

    def test_sayfa_flann_indeksi_bir_kez_hazirlanip_referanslarda_kullanilir(self):
        descriptor = np.zeros((12, 128), dtype=np.float32)
        matcher = MagicMock()
        matcher.knnMatch.return_value = []

        with patch.object(
            gi.cv2, "FlannBasedMatcher", return_value=matcher, create=True
        ) as olustur:
            hazir = gi.sayfa_matcher_hazirla(descriptor, "SIFT")

        self.assertIs(hazir, matcher)
        olustur.assert_called_once()
        matcher.add.assert_called_once()
        np.testing.assert_array_equal(matcher.add.call_args.args[0][0], descriptor)
        matcher.train.assert_called_once_with()

        H, iyi, inlier = gi.referansla_eslestir(
            [], descriptor[:4], [], descriptor, "SIFT", hazir
        )
        self.assertIsNone(H)
        self.assertEqual(iyi, [])
        self.assertEqual(inlier, 0)
        self.assertEqual(matcher.knnMatch.call_count, 1)
        np.testing.assert_array_equal(matcher.knnMatch.call_args.args[0], descriptor[:4])
        self.assertEqual(matcher.knnMatch.call_args.kwargs, {"k": 2})

    def test_a4_icindeki_kucuk_kart_sayfa_yayilimi_yuzunden_reddedilmez(self):
        ref_h, ref_w = 974, 639
        # Inlier'lar referansin %30'una yayiliyor; kart ise A4 sayfasinin
        # yaklasik %25 x %31'ini kapliyor. Eski sayfa-normalize %12 kapisi,
        # inlier yayilimini kart boyutu yerine A4'e bolerek bunu reddediyordu.
        noktalar = [
            (224.0, 341.0), (416.0, 341.0), (416.0, 633.0), (224.0, 633.0),
            (320.0, 341.0), (416.0, 487.0), (320.0, 633.0), (224.0, 487.0),
        ]
        olcek = 0.32
        tx, ty = 260.0, 280.0
        hedefler = [(x * olcek + tx, y * olcek + ty) for x, y in noktalar]
        gocmen = aday("gocmen", 120.0, np.zeros((ref_h, ref_w, 3), dtype=np.uint8))
        gocmen["kp_ref"] = [types.SimpleNamespace(pt=x) for x in noktalar]
        gocmen["kp_resim"] = [types.SimpleNamespace(pt=x) for x in hedefler]
        gocmen["iyi"] = [
            types.SimpleNamespace(queryIdx=i, trainIdx=i) for i in range(len(noktalar))
        ]
        affine = np.array([[olcek, 0.0, tx], [0.0, olcek, ty]], dtype=np.float64)
        maske = np.ones((len(noktalar), 1), dtype=np.uint8)

        with (
            patch.object(gi.cv2, "RANSAC", 8, create=True),
            patch.object(
                gi.cv2, "estimateAffinePartial2D",
                return_value=(affine, maske), create=True,
            ),
        ):
            koseler, metrikler, hata = gi._gocmen_affine_koseleri_bul(
                gocmen, 1.0, (1000, 800, 3),
            )

        self.assertEqual("", hata)
        self.assertIsNotNone(koseler)
        self.assertLess(metrikler["hedef_x_yayilimi"], 0.12)
        self.assertLess(metrikler["hedef_y_yayilimi"], 0.12)
        self.assertTrue(gi.koseler_gecerli_mi(koseler, (1000, 800, 3), "gocmen"))

    def test_affine_tablo_ayiricisi_beklenen_sutunda_olmali(self):
        dogru, dogru_metrikler = gi._gocmen_tablo_imzasi_gecerli_mi(
            self._gocmen_tablolu_kart_uret(392)
        )
        yanlis, yanlis_metrikler = gi._gocmen_tablo_imzasi_gecerli_mi(
            self._gocmen_tablolu_kart_uret(200)
        )

        self.assertTrue(dogru, dogru_metrikler)
        self.assertTrue(0.36 <= dogru_metrikler["deger_ayirici_x"] <= 0.64)
        self.assertFalse(yanlis, yanlis_metrikler)
        self.assertLess(yanlis_metrikler["deger_ayirici_x"], 0.36)

    def test_geometri_gecersizse_ham_resim_basarili_sayilmaz(self):
        tc = aday("tc", 100.0, "tc-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "tc",
            "en_iyi": tc,
            "adaylar": [tc],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=False),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertFalse(sonuc["duzeltildi"])
        self.assertFalse(sonuc["fallback"])
        self.assertIsNone(sonuc["kart"])

    def test_yeni_tc_duzgun_dortgeni_kabul_edilir(self):
        self.assertTrue(gi.koseler_gecerli_mi(self.koseler, self.resim.shape, "tc"))

    def test_gocmen_duzgun_dortgeni_kabul_edilir(self):
        koseler = np.array(
            [[80, 60], [820, 80], [790, 1130], [100, 1110]],
            dtype=np.float32,
        )
        self.assertTrue(gi.koseler_gecerli_mi(koseler, (1200, 900, 3), "gocmen"))

    def test_self_intersection_ve_konkav_dortgen_reddedilir(self):
        self_intersection = np.array(
            [[100, 100], [800, 100], [100, 1050], [800, 1050]],
            dtype=np.float32,
        )
        konkav = np.array(
            [[100, 100], [800, 100], [420, 350], [100, 1050]],
            dtype=np.float32,
        )

        self.assertFalse(gi.koseler_gecerli_mi(self_intersection, (1200, 900, 3), "gocmen"))
        self.assertFalse(gi.koseler_gecerli_mi(konkav, (1200, 900, 3), "gocmen"))

    def test_gocmen_asiri_disaridaki_dortgen_reddedilir(self):
        # Eski kontrol bu kareyi alan/aspect/tolerans sinirlarinda kabul
        # ediyordu; gercekte yalnizca yuzde 20'ye yakini goruntu icinde.
        koseler = np.array(
            [[-400, -400], [300, -400], [300, 300], [-400, 300]],
            dtype=np.float32,
        )
        self.assertFalse(gi.koseler_gecerli_mi(koseler, (1200, 900, 3), "gocmen"))

    def test_gocmen_cok_ince_trapez_reddedilir(self):
        # Ortalama genislik/yukseklik orani tek basina 2.3 ve eski esigi
        # geciyordu; alt kenarin ust kenara gore 14 kat kucuk olmasi guvensiz.
        koseler = np.array(
            [[100, 100], [800, 100], [450, 900], [400, 900]],
            dtype=np.float32,
        )
        self.assertFalse(gi.koseler_gecerli_mi(koseler, (1200, 900, 3), "gocmen"))

    def test_sentetik_iyi_warp_kalite_kontrolunu_gecer(self):
        kaliteli, metrikler = gi._kart_ciktisi_kaliteli_mi(self.kaliteli_kart, "gocmen")

        self.assertTrue(kaliteli, metrikler)
        self.assertGreaterEqual(metrikler["yatay_ayrinti_kapsami"], 0.50)
        self.assertTrue(metrikler["gecerli"])

    def test_bos_ve_cizgisel_warp_kalite_kontrolunde_reddedilir(self):
        bozuk = self._bozuk_cizgisel_warp_uret()
        kaliteli, metrikler = gi._kart_ciktisi_kaliteli_mi(bozuk, "gocmen")

        self.assertFalse(kaliteli, metrikler)
        self.assertLess(metrikler["yatay_ayrinti_kapsami"], 0.50)

    def test_radyal_fan_warp_ayrinti_dolu_olsa_da_reddedilir(self):
        bozuk = self._bozuk_radyal_warp_uret()
        kaliteli, metrikler = gi._kart_ciktisi_kaliteli_mi(bozuk, "eski_tc")

        # Fan goruntusu eski doluluk kontrollerini gececek kadar tum alana
        # yayilir; onu ayiran yeni sonlu-yakinsama metrigidir.
        self.assertGreaterEqual(metrikler["yatay_ayrinti_kapsami"], 0.80)
        self.assertGreaterEqual(metrikler["dikey_ayrinti_kapsami"], 0.80)
        self.assertGreaterEqual(metrikler["radyal_yakinsama_skoru"], 0.58)
        self.assertGreaterEqual(metrikler["radyal_ayrinti_kapsami"], 0.50)
        self.assertFalse(metrikler["radyal_gecerli"])
        self.assertFalse(kaliteli, metrikler)

    def test_gocmen_720_hedefinde_radyal_fan_esigi_dellemez(self):
        bozuk = self._bozuk_radyal_warp_uret(w=720, h=1100)
        kaliteli, metrikler = gi._kart_ciktisi_kaliteli_mi(bozuk, "gocmen")

        self.assertGreaterEqual(metrikler["radyal_yakinsama_skoru"], 0.58)
        self.assertGreaterEqual(metrikler["radyal_ayrinti_kapsami"], 0.45)
        self.assertFalse(metrikler["radyal_gecerli"])
        self.assertFalse(kaliteli, metrikler)

    def test_radyal_fan_warp_ana_akista_basarili_sayilmaz(self):
        eski = aday("eski_tc", 140.0, "eski-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "eski_tc",
            "en_iyi": eski,
            "adaylar": [eski],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=True),
            patch.object(gi, "perspektif_duzelt", return_value=self._bozuk_radyal_warp_uret()),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertFalse(sonuc["aday_sonuclari"][0]["cikti_kalitesi_gecerli"])
        self.assertFalse(
            sonuc["aday_sonuclari"][0]["kalite_metrikleri"]["radyal_gecerli"]
        )

    def test_normal_belge_radyal_fan_sayilmaz(self):
        kaliteli, metrikler = gi._kart_ciktisi_kaliteli_mi(self.kaliteli_kart, "gocmen")

        self.assertTrue(kaliteli, metrikler)
        self.assertTrue(metrikler["radyal_gecerli"])

    def test_gecersiz_eski_tc_geometrisi_yuksek_inlier_ile_bypass_edilmez(self):
        eski = aday("eski_tc", 140.0, "eski-ref")
        eski["inlier"] = 30
        eski["inlier_orani"] = 0.90
        gocmen = aday("gocmen", 120.0, "gocmen-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "eski_tc",
            "en_iyi": eski,
            "adaylar": [eski, gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", side_effect=[False, True]),
            patch.object(gi, "perspektif_duzelt", return_value=self.kaliteli_kart.copy()) as warp,
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertTrue(sonuc["basarili"])
        self.assertEqual(sonuc["belge_tipi"], "gocmen")
        self.assertEqual(sonuc["aday_sirasi"], 2)
        self.assertFalse(sonuc["aday_sonuclari"][0]["toleransli"])
        self.assertEqual(warp.call_count, 1)

    def test_eski_tc_yedi_inlier_artik_aday_sayilmaz(self):
        with (
            patch.object(gi, "referansi_hazirla", return_value=("ref", [], "des", None)),
            patch.object(gi, "referansla_eslestir", return_value=(object(), [object()] * 20, 7)),
        ):
            sonuc = gi.tek_referansi_test_et("eski_tc", "ref.jpg", [], "des", "SIFT")

        self.assertFalse(sonuc["basarili"])
        self.assertEqual(sonuc["inlier"], 7)

    def test_bozuk_warp_ana_akista_basarili_sayilmaz(self):
        gocmen = aday("gocmen", 120.0, "gocmen-ref")
        tip_sonuc = {
            "basarili": True,
            "belge_tipi": "gocmen",
            "en_iyi": gocmen,
            "adaylar": [gocmen],
            "sonuclar": {},
            "olcek": 1.0,
        }

        with (
            patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc),
            patch.object(gi, "koseleri_bul", return_value=self.koseler),
            patch.object(gi, "koseler_gecerli_mi", return_value=True),
            patch.object(gi, "perspektif_duzelt", return_value=self._bozuk_cizgisel_warp_uret()),
        ):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertFalse(sonuc["basarili"])
        self.assertFalse(sonuc["aday_sonuclari"][0]["cikti_kalitesi_gecerli"])
        self.assertIn("kalitesi", sonuc["aday_sonuclari"][0]["hata"])

    def test_tespit_suresi_basarili_ve_erken_donuslerde_doldurulur(self):
        gecersiz = gi.kart_tespit_et_ve_duzelt(None)
        tip_sonuc = {"basarili": False, "sonuclar": {}}
        with patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc):
            bulunamadi = gi.kart_tespit_et_ve_duzelt(self.resim)

        for sonuc in (gecersiz, bulunamadi):
            self.assertIn("tespit_suresi", sonuc)
            self.assertIsInstance(sonuc["tespit_suresi"], float)
            self.assertGreaterEqual(sonuc["tespit_suresi"], 0.0)

    def test_referans_hatalari_ana_sonucta_kaybolmaz(self):
        tip_sonuc = {
            "basarili": False,
            "sonuclar": {
                "tc": {"hata": "tc referansı eksik"},
                "gocmen": {"hata": "göçmen referansı eksik"},
            },
        }
        with patch.object(gi, "belge_tipini_bul", return_value=tip_sonuc):
            sonuc = gi.kart_tespit_et_ve_duzelt(self.resim)

        self.assertIn("tc referansı eksik", sonuc["mesaj"])
        self.assertIn("göçmen referansı eksik", sonuc["mesaj"])
        self.assertEqual(len(sonuc["referans_hatalari"]), 2)


if __name__ == "__main__":
    unittest.main()
