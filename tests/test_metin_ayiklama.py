import importlib.util
import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import Mock, patch

import numpy as np


if importlib.util.find_spec("cv2") is None:
    cv2 = types.ModuleType("cv2")
    cv2.__spec__ = ModuleSpec("cv2", loader=None)
    sys.modules["cv2"] = cv2

if importlib.util.find_spec("easyocr") is None:
    easyocr = types.ModuleType("easyocr")
    easyocr.__spec__ = ModuleSpec("easyocr", loader=None)

    class _Reader:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Bu testte OCR modeli oluşturulmamalı.")

    easyocr.Reader = _Reader
    sys.modules["easyocr"] = easyocr


from metin_ayiklama import (
    _eski_tc_sonuclarini_birlestir,
    _gocmen_deger_sutunu_sol_siniri,
    _gocmen_hucre_onislem_varyantlari,
    _gocmen_hucre_kutusu,
    _gocmen_label_haritasi,
    bilgileri_cimbizla,
    easyocr_oku_gocmen,
    gocmen_bilgilerini_oku,
)


def _item(text, y1, y2, conf=0.80):
    return {"text": text, "conf": conf, "x1": 100, "y1": y1, "x2": 250, "y2": y2}


def _gocmen_item(text, x1, y1, x2, y2, conf=0.90):
    return {"text": text, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _gocmen_ocr(no=True, ad=True, soyad=True, tarih=True):
    ocr = [
        _gocmen_item("YABANCI KIMLIK NO", 20, 464, 300, 494),
        _gocmen_item("ADI", 20, 495, 100, 536),
        _gocmen_item("SOYADI", 20, 536, 120, 578),
        _gocmen_item("BELGENIN GECERLILIK TARIHI", 20, 862, 360, 929),
    ]
    if no:
        ocr.append(_gocmen_item("99251771180", 376, 464, 610, 494))
    if ad:
        ocr.append(_gocmen_item("ESRA SABAH MUHAMMED ALI", 376, 495, 760, 536))
    if soyad:
        ocr.append(_gocmen_item("KRUT", 376, 536, 500, 578))
    if tarih:
        ocr.append(_gocmen_item("01.01.2020 - 31.12.2099", 370, 862, 760, 929))
    return ocr


def _gocmen_gridli_kart():
    """Direct yolun gerektirdigi value divider ve hedef satir sinirlari."""
    kart = np.full((1100, 800, 3), 255, dtype=np.uint8)
    kart[425:950, 365:368] = 0
    for y in (458, 498, 538, 578, 864, 930):
        kart[y:y + 3, 360:790] = 0
    return kart


def _gocmen_kaymis_olcekli_gridli_kart():
    """Referansa gore hem dikey kaymis hem %8 kuculmus tablo."""
    kart = np.full((1100, 800, 3), 255, dtype=np.uint8)
    olcek, kayma = 0.92, 80
    cizgiler = [
        int(round(olcek * y + kayma))
        for y in (458, 498, 538, 578, 864, 930)
    ]
    kart[330:1000, 401:404] = 0
    for y in cizgiler:
        kart[y:y + 3, 396:790] = 0
    return kart, cizgiler


class EskiTcSonucBirlestirmeTesti(unittest.TestCase):
    def test_ilk_sonucu_koruyup_eksik_alani_fallbackten_doldurur(self):
        ilk = {
            "tc_no": "10000000146", "tc_item": _item("10000000146", 100, 125),
            "soyad": "YILMAZ", "soyad_item": _item("YILMAZ", 200, 230), "soyad_conf": 0.91,
            "ad": "Bulunamadi", "ad_item": None, "ad_conf": 0.0,
        }
        fallback = {
            "tc_no": "Bulunamadi", "tc_item": None,
            "soyad": "YANLIS", "soyad_item": _item("YANLIS", 200, 230), "soyad_conf": 0.99,
            "ad": "AYSE", "ad_item": _item("AYSE", 260, 290), "ad_conf": 0.82,
        }

        sonuc, kullanildi = _eski_tc_sonuclarini_birlestir(ilk, fallback)

        self.assertTrue(kullanildi)
        self.assertEqual("10000000146", sonuc["tc_no"])
        self.assertEqual("YILMAZ", sonuc["soyad"])
        self.assertEqual("AYSE", sonuc["ad"])

    def test_ayni_satiri_iki_farkli_isim_alani_olarak_kullanmaz(self):
        ilk = {
            "tc_no": "10000000146", "ad": "ALI", "ad_item": _item("ALI", 250, 280), "ad_conf": 0.80,
            "soyad": "Bulunamadi", "soyad_item": None, "soyad_conf": 0.0,
        }
        fallback = {
            "tc_no": "10000000146", "soyad": "ALI", "soyad_item": _item("ALI", 252, 282),
            "soyad_conf": 0.95, "ad": "ALI", "ad_item": _item("ALI", 252, 282), "ad_conf": 0.95,
        }

        sonuc, kullanildi = _eski_tc_sonuclarini_birlestir(ilk, fallback)

        self.assertFalse(kullanildi)
        self.assertEqual("Bulunamadi", sonuc["soyad"])
        self.assertEqual("ALI", sonuc["ad"])


class CiktiSozlesmesiTesti(unittest.TestCase):
    def test_desteklenmeyen_belge_tipi_sessizce_tcye_dusmez(self):
        sonuc = bilgileri_cimbizla(object(), belge_tipi="yanlis_tip")

        self.assertEqual("yanlis_tip", sonuc["belge_tipi"])
        self.assertIn("Desteklenmeyen", sonuc["hata"])
        self.assertEqual("Bulunamadi", sonuc["tc_no"])

    def test_bos_kart_tutarli_cikti_semasi_dondurur(self):
        sonuc = bilgileri_cimbizla(None, belge_tipi="gocmen")

        for alan in (
            "tc_no", "ad", "soyad", "guven", "kimlik_no_conf", "ad_conf",
            "soyad_conf", "baslangic_tarihi", "bitis_tarihi", "belge_gecerli",
            "gecerlilik_durumu", "ocr_suresi", "hata",
        ):
            self.assertIn(alan, sonuc)


class GocmenRoiTesti(unittest.TestCase):
    def _basarili_reader(self):
        reader = Mock()

        def recognize_side_effect(_kart, **kwargs):
            if kwargs["allowlist"] == "0123456789":
                return [(None, "99251771180", 0.93)]
            if kwargs["allowlist"] == "0123456789./- ":
                kutu_sayisi = len(kwargs["horizontal_list"])
                if kutu_sayisi == 2:
                    return [
                        (None, "99251771180", 0.93),
                        (None, "01.01.2020 - 31.12.2099", 0.91),
                    ]
                return [(None, "01.01.2020 - 31.12.2099", 0.91)]
            return [
                (None, "ESRA SABAH MUHAMMED ALI", 0.92),
                (None, "KRUT", 0.94),
            ]

        reader.recognize.side_effect = recognize_side_effect
        return reader

    def test_detector_yerine_yalniz_deger_hucrelerini_recognize_eder(self):
        kart = _gocmen_gridli_kart()
        reader = self._basarili_reader()

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama.easyocr_oku") as detector_oku,
        ):
            sonuc, sure = easyocr_oku_gocmen(kart)

        self.assertGreaterEqual(sure, 0.0)
        self.assertEqual(2, reader.recognize.call_count)
        detector_oku.assert_not_called()
        reader.readtext.assert_not_called()
        self.assertIn("99251771180", [x["text"] for x in sonuc])
        self.assertIn("ESRA SABAH MUHAMMED ALI", [x["text"] for x in sonuc])
        self.assertIn("KRUT", [x["text"] for x in sonuc])
        self.assertIn("01.01.2020 - 31.12.2099", [x["text"] for x in sonuc])
        self.assertTrue(all(
            cagri.kwargs["free_list"] == []
            and cagri.kwargs["contrast_ths"] == 0.0
            for cagri in reader.recognize.call_args_list
        ))

    def test_ayiriciya_bitisik_ilk_ykn_hanesi_crop_icinde_kalir(self):
        kart = _gocmen_gridli_kart()
        # Dikey ayirici x=365:368 araliginda. Ilk 9'un sol govdesi cizginin
        # hemen ardindaki x=368 pikselinden baslayan eski taramayi temsil eder.
        ilk_rakam_x = 368
        kart[468:490, ilk_rakam_x:ilk_rakam_x + 2] = 0
        kart[468:471, ilk_rakam_x:ilk_rakam_x + 10] = 0
        reader = self._basarili_reader()

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        ykn_kutusu = reader.recognize.call_args_list[0].kwargs[
            "horizontal_list"
        ][0]
        self.assertLessEqual(ykn_kutusu[0], ilk_rakam_x)
        self.assertEqual("99251771180", sonuc["tc_no"])
        self.assertEqual(2, reader.recognize.call_count)
        detector_oku.assert_not_called()

    def test_hizli_yol_iki_cagriyla_tum_oncelikli_alanlari_dondurur(self):
        kart = _gocmen_gridli_kart()
        reader = self._basarili_reader()

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        self.assertEqual("99251771180", sonuc["tc_no"])
        self.assertEqual("ESRA SABAH MUHAMMED ALI", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])
        self.assertEqual("31.12.2099", sonuc["bitis_tarihi"])
        self.assertEqual(2, reader.recognize.call_count)
        self.assertEqual([], sonuc["fallback_denenen_alanlar"])
        self.assertTrue(sonuc["hizli_yol_kullanildi"])
        self.assertFalse(sonuc["detector_fallback_kullanildi"])
        detector_oku.assert_not_called()
        reader.readtext.assert_not_called()

    def test_checksum_gecerli_direct_ykn_039_guvende_korunur(self):
        kart = _gocmen_gridli_kart()
        reader = Mock()
        reader.recognize.side_effect = [
            [
                (None, "99011944816", 0.39),
                (None, "01.01.2020 - 31.12.2099", 0.91),
            ],
            [
                (None, "DALAL MUHAMMED", 0.92),
                (None, "ABBOUD", 0.94),
            ],
        ]

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        self.assertEqual("99011944816", sonuc["tc_no"])
        self.assertEqual(0.39, sonuc["kimlik_no_conf"])
        self.assertEqual(2, reader.recognize.call_count)
        self.assertTrue(all(
            cagri.kwargs["batch_size"] == 2
            for cagri in reader.recognize.call_args_list
        ))
        self.assertTrue(sonuc["hizli_yol_kullanildi"])
        self.assertFalse(sonuc["detector_fallback_kullanildi"])
        detector_oku.assert_not_called()

    def test_kaymis_olcekli_gridde_hucreler_dinamik_ve_yol_iki_cagridir(self):
        kart, cizgiler = _gocmen_kaymis_olcekli_gridli_kart()
        reader = self._basarili_reader()

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        self.assertEqual(2, reader.recognize.call_count)
        self.assertTrue(sonuc["hizli_yol_kullanildi"])
        detector_oku.assert_not_called()
        sayisal_kutular = reader.recognize.call_args_list[0].kwargs["horizontal_list"]
        ykn_kutusu = sayisal_kutular[0]
        isim_kutulari = reader.recognize.call_args_list[1].kwargs["horizontal_list"]
        tarih_kutusu = sayisal_kutular[1]
        for kutu, (ust, alt) in zip(
            [ykn_kutusu, *isim_kutulari, tarih_kutusu],
            [
                (cizgiler[0], cizgiler[1]),
                (cizgiler[1], cizgiler[2]),
                (cizgiler[2], cizgiler[3]),
                (cizgiler[4], cizgiler[5]),
            ],
        ):
            self.assertGreater(kutu[2], ust)
            self.assertLess(kutu[2], alt)
            self.assertGreater(kutu[3], ust)
            self.assertLess(kutu[3], alt)

    def test_referanstaki_kisa_ykn_satiri_ad_satiriyla_karismaz(self):
        kart = np.full((1100, 800, 3), 255, dtype=np.uint8)
        kart[390:960, 365:368] = 0
        for y in (461, 488, 532, 573, 863, 930):
            kart[y:y + 3, 360:790] = 0
        reader = self._basarili_reader()

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        self.assertTrue(sonuc["hizli_yol_kullanildi"])
        self.assertEqual(2, reader.recognize.call_count)
        detector_oku.assert_not_called()
        ykn = reader.recognize.call_args_list[0].kwargs["horizontal_list"][0]
        ad = reader.recognize.call_args_list[1].kwargs["horizontal_list"][0]
        self.assertLess(ykn[3], 488)
        self.assertGreater(ad[2], 488)

    def test_tam_tarih_cifti_yoksa_sag_bitis_tek_ek_cagriyla_okunur(self):
        kart = _gocmen_gridli_kart()
        reader = Mock()
        reader.recognize.side_effect = [
            [
                (None, "99251771180", 0.93),
                (None, "11.07.2025", 0.91),  # tam hucre: tek tarih belirsiz
            ],
            [(None, "ESRA", 0.92), (None, "KRUT", 0.94)],
            [(None, "27.06.2027", 0.90)],  # dogrulanmis sag/bitis bolumu
        ]

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch("metin_ayiklama._gocmen_detector_ocr_oku") as detector_oku,
            patch("metin_ayiklama.debug_resmi_olustur", return_value="debug") as debug_olustur,
        ):
            sonuc = gocmen_bilgilerini_oku(kart, debug=True)

        self.assertEqual(3, reader.recognize.call_count)
        self.assertEqual("Bulunamadi", sonuc["baslangic_tarihi"])
        self.assertEqual("27.06.2027", sonuc["bitis_tarihi"])
        self.assertTrue(sonuc["hizli_yol_kullanildi"])
        detector_oku.assert_not_called()
        sag_kutu = reader.recognize.call_args_list[2].kwargs["horizontal_list"][0]
        tam_kutu = reader.recognize.call_args_list[0].kwargs["horizontal_list"][1]
        self.assertGreater(sag_kutu[0], tam_kutu[0])
        debug_alanlari = debug_olustur.call_args.args[1]
        self.assertIn("BITIS", [etiket for _, etiket, _ in debug_alanlari])
        self.assertNotIn("BASLANGIC", [etiket for _, etiket, _ in debug_alanlari])

    def test_direct_strong_sonrasi_detector_hucre_waterfallu_tekrarlanmaz(self):
        kart = _gocmen_gridli_kart()
        reader = Mock()
        reader.recognize.return_value = []

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch(
                "metin_ayiklama._gocmen_detector_ocr_oku",
                return_value=([], 0.40),
            ) as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        # 2 ham batch + kosullu sag bitis + 3 guclu varyant grubu. Dinamik grid
        # kuruldugu icin pahali detector'a gecilmez.
        self.assertEqual(6, reader.recognize.call_count)
        detector_oku.assert_not_called()
        self.assertEqual(
            ["kimlik_no", "ad", "soyad", "gecerlilik"],
            sonuc["fallback_denenen_alanlar"],
        )

    def test_grid_geometrisi_yoksa_direct_ocr_yapmadan_detectora_duser(self):
        kart = np.zeros((1100, 800, 3), dtype=np.uint8)

        with (
            patch("metin_ayiklama.reader_getir") as reader_getir,
            patch(
                "metin_ayiklama._gocmen_detector_ocr_oku",
                return_value=(_gocmen_ocr(), 0.80),
            ) as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        reader_getir.assert_not_called()
        detector_oku.assert_called_once()
        self.assertFalse(sonuc["hizli_yol_kullanildi"])
        self.assertTrue(sonuc["detector_fallback_kullanildi"])
        self.assertEqual("99251771180", sonuc["tc_no"])

    def test_direct_geometri_varken_dusuk_guven_detectoru_tetiklemez(self):
        kart = _gocmen_gridli_kart()
        reader = Mock()
        reader.recognize.side_effect = [
            [
                (None, "99251771180", 0.93),
                (None, "01.01.2020 - 31.12.2099", 0.91),
            ],
            [(None, "YANLIS AD", 0.49), (None, "KRUT", 0.94)],
        ]

        with (
            patch("metin_ayiklama.reader_getir", return_value=reader),
            patch(
                "metin_ayiklama._gocmen_detector_ocr_oku",
                return_value=(_gocmen_ocr(), 0.80),
            ) as detector_oku,
        ):
            sonuc = gocmen_bilgilerini_oku(kart)

        detector_oku.assert_not_called()
        self.assertEqual("YANLIS AD", sonuc["ad"])
        self.assertFalse(sonuc["detector_fallback_kullanildi"])

    def test_gecersiz_ykn_ham_hucre_tekrari_yapmadan_tek_guclu_fallback_alir(self):
        kart = _gocmen_gridli_kart()
        reader = Mock()
        reader.recognize.side_effect = [
            [
                (None, "99539448747", 0.97),  # hizli ham YKN, checksum gecmez
                (None, "01.01.2020 - 31.12.2099", 0.91),
            ],
            [
                (None, "ESRA SABAH MUHAMMED ALI", 0.92),
                (None, "KRUT", 0.94),
            ],
            [
                (None, "99539448742", 0.88),
                (None, "99539448742", 0.90),
            ],
        ]

        with patch("metin_ayiklama.reader_getir", return_value=reader):
            sonuc = gocmen_bilgilerini_oku(kart)

        self.assertEqual("99539448742", sonuc["tc_no"])
        # 2 hizli batch + dogrudan tek guclu YKN fallback'i. Ayni ham YKN
        # hucresi ikinci kez okunmaz.
        self.assertEqual(3, reader.recognize.call_count)
        self.assertEqual(["kimlik_no"], sonuc["fallback_denenen_alanlar"])
        self.assertEqual(["kimlik_no"], sonuc["fallback_kullanilan_alanlar"])
        reader.readtext.assert_not_called()


class GocmenHucreFallbackTesti(unittest.TestCase):
    def setUp(self):
        self.kart = np.zeros((1100, 800, 3), dtype=np.uint8)

    def test_tam_sonucta_recognize_fallback_hic_cagrilmaz(self):
        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(_gocmen_ocr(), 0.40)),
            patch("metin_ayiklama.reader_getir") as reader_getir,
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        reader_getir.assert_not_called()
        self.assertEqual([], sonuc["fallback_denenen_alanlar"])
        self.assertEqual([], sonuc["fallback_kullanilan_alanlar"])
        self.assertEqual(0.0, sonuc["fallback_ocr_suresi"])
        self.assertEqual(0.40, sonuc["ocr_suresi"])

    def test_kacirilan_uzun_adi_sabit_hucreden_detector_olmadan_tamamlar(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "ESRA SABAH MUHAMMED ALI", 0.92)]

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(_gocmen_ocr(ad=False), 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("ESRA SABAH MUHAMMED ALI", sonuc["ad"])
        self.assertEqual(["ad"], sonuc["fallback_denenen_alanlar"])
        self.assertEqual(["ad"], sonuc["fallback_kullanilan_alanlar"])
        reader.recognize.assert_called_once()
        self.assertIsNone(reader.recognize.call_args.kwargs["allowlist"])
        self.assertEqual("0123456789", reader.recognize.call_args.kwargs["blocklist"])
        ad_label = next(item for item in _gocmen_ocr(ad=False) if item["text"] == "ADI")
        merkez = (ad_label["y1"] + ad_label["y2"]) / 2
        kutu = reader.recognize.call_args.kwargs["horizontal_list"][0]
        self.assertLess(kutu[2], merkez)
        self.assertGreater(kutu[3], merkez)
        soyad_label = next(item for item in _gocmen_ocr(ad=False) if item["text"] == "SOYADI")
        soyad_merkez = (soyad_label["y1"] + soyad_label["y2"]) / 2
        self.assertLess(kutu[3], (merkez + soyad_merkez) / 2)
        reader.readtext.assert_not_called()

    def test_checksum_gecmeyen_ykn_hucre_fallbackinde_reddedilir(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "99086835448", 0.96)]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(no=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["tc_no"])
        self.assertEqual(["kimlik_no"], sonuc["fallback_denenen_alanlar"])
        self.assertEqual([], sonuc["fallback_kullanilan_alanlar"])

    def test_ana_ocr_yanlis_ykn_verirse_sabit_hucre_dogru_degeri_kurtarir(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "99086835446", 0.91)]
        ana_ocr = [
            *_gocmen_ocr(no=False),
            _gocmen_item("99086835448", 376, 464, 610, 494, conf=0.97),
        ]

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(ana_ocr, 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("99086835446", sonuc["tc_no"])
        self.assertEqual(["kimlik_no"], sonuc["fallback_kullanilan_alanlar"])

    def test_tum_eksikler_uc_tip_cagriyla_tamamlanir_ad_soyad_tek_cagri_olur(self):
        reader = Mock()

        def recognize_side_effect(_kart, **kwargs):
            allowlist = kwargs["allowlist"]
            if allowlist == "0123456789":
                return [(None, "99251771180", 0.91)]
            if allowlist == "0123456789./- ":
                return [(None, "01.01.2020 - 31.12.2099", 0.88)]
            return [(None, "ESRA SABAH MUHAMMED ALI", 0.92), (None, "KRUT", 0.94)]

        reader.recognize.side_effect = recognize_side_effect
        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(no=False, ad=False, soyad=False, tarih=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("99251771180", sonuc["tc_no"])
        self.assertEqual("ESRA SABAH MUHAMMED ALI", sonuc["ad"])
        self.assertEqual("KRUT", sonuc["soyad"])
        self.assertEqual("31.12.2099", sonuc["bitis_tarihi"])
        self.assertEqual(3, reader.recognize.call_count)
        isim_cagrisi = next(
            cagri for cagri in reader.recognize.call_args_list
            if cagri.kwargs["allowlist"] is None
        )
        self.assertEqual(2, len(isim_cagrisi.kwargs["horizontal_list"]))
        self.assertTrue(all(cagri.kwargs["free_list"] == [] for cagri in reader.recognize.call_args_list))
        reader.readtext.assert_not_called()

    def test_label_veya_gurultu_isim_olarak_kabul_edilmez(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "ADI", 0.99)]

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(_gocmen_ocr(ad=False), 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["ad"])
        self.assertEqual(["ad"], sonuc["fallback_denenen_alanlar"])
        self.assertEqual([], sonuc["fallback_kullanilan_alanlar"])

    def test_gecersiz_tarih_cifti_yuksek_guvenle_bile_kabul_edilmez(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "31.13.2025 - 42.14.2026", 0.99)]

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(_gocmen_ocr(tarih=False), 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["bitis_tarihi"])
        self.assertEqual(["gecerlilik"], sonuc["fallback_denenen_alanlar"])
        self.assertEqual([], sonuc["fallback_kullanilan_alanlar"])
        self.assertEqual("0123456789./- ", reader.recognize.call_args.kwargs["allowlist"])

    def test_ayiraclari_kaybolan_net_tarih_cifti_tamamlanir(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "1107202527062026", 0.88)]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(tarih=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("11.07.2025", sonuc["baslangic_tarihi"])
        self.assertEqual("27.06.2026", sonuc["bitis_tarihi"])
        self.assertEqual(["gecerlilik"], sonuc["fallback_kullanilan_alanlar"])

    def test_ilk_tarih_okumasi_bossa_onislemli_ikinci_gecis_kurtarir(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [],
            [(None, "1107202527062026", 0.86)],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(tarih=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("11.07.2025", sonuc["baslangic_tarihi"])
        self.assertEqual("27.06.2026", sonuc["bitis_tarihi"])
        self.assertEqual(["gecerlilik"], sonuc["fallback_kullanilan_alanlar"])
        self.assertEqual(2, reader.recognize.call_count)
        ikinci_gorsel = reader.recognize.call_args_list[1].args[0]
        ilk_kutu = reader.recognize.call_args_list[0].kwargs["horizontal_list"][0]
        tek_varyant_h = (ilk_kutu[3] - ilk_kutu[2]) * 2
        self.assertEqual(tek_varyant_h * 2 + 12, ikinci_gorsel.shape[0])
        self.assertEqual((ilk_kutu[1] - ilk_kutu[0]) * 2, ikinci_gorsel.shape[1])
        self.assertEqual(2, len(reader.recognize.call_args_list[1].kwargs["horizontal_list"]))

    def test_iki_farkli_gecerli_tarih_varyanti_konsensus_yoksa_reddedilir(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [(None, "01012025", 0.91)],
            [
                (None, "01/01/2025 - 01/01/2027", 0.94),
                (None, "02/01/2025 - 02/01/2027", 0.97),
            ],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(tarih=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["bitis_tarihi"])
        self.assertNotIn("gecerlilik", sonuc["fallback_kullanilan_alanlar"])

    def test_sabit_hucre_yeri_bulunan_label_satirina_gore_kayar(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "RAAD TAREQ KHUDHUR", 0.91)]
        ana_ocr = _gocmen_ocr(ad=False)
        for item in ana_ocr:
            if item["y1"] < 700:
                item["y1"] += 55
                item["y2"] += 55
        ad_label = next(item for item in ana_ocr if item["text"] == "ADI")

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(ana_ocr, 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            gocmen_bilgilerini_oku(self.kart)

        kutu = reader.recognize.call_args.kwargs["horizontal_list"][0]
        sabit_ad_y1 = int(1100 * 0.448)
        self.assertGreater(kutu[2], sabit_ad_y1 + 20)
        self.assertLess(kutu[2], (ad_label["y1"] + ad_label["y2"]) / 2)
        self.assertGreater(kutu[3], (ad_label["y1"] + ad_label["y2"]) / 2)

    def test_eksik_soyad_labeli_diger_labellarin_kaymasiyla_tahmin_edilir(self):
        ana_ocr = [
            item for item in _gocmen_ocr(soyad=False)
            if item["text"] != "SOYADI"
        ]
        for item in ana_ocr:
            if item["y1"] < 700:
                item["y1"] += 55
                item["y2"] += 55

        reader = Mock()

        def recognize_side_effect(_kart, **kwargs):
            kutu = kwargs["horizontal_list"][0]
            # Eski sabit kutu kaymis AD satirini okuyup soyada kopyaliyordu.
            text = "AL WIIB" if kutu[2] > 580 else "ILHAM"
            return [(None, text, 0.91)]

        reader.recognize.side_effect = recognize_side_effect
        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(ana_ocr, 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("AL WIIB", sonuc["soyad"])
        kutu = reader.recognize.call_args.kwargs["horizontal_list"][0]
        self.assertGreater(kutu[2], 580)
        self.assertEqual(["soyad"], sonuc["fallback_kullanilan_alanlar"])

    def test_ad_satirindan_okunan_fallback_soyada_zorla_yazilmaz(self):
        ana_ocr = [
            item for item in _gocmen_ocr(soyad=False)
            if item["text"] != "SOYADI"
        ]
        for item in ana_ocr:
            if item["y1"] < 700:
                item["y1"] += 55
                item["y2"] += 55

        reader = Mock()
        reader.recognize.return_value = [(None, "ESRA SABAH MUHAMMED ALI", 0.93)]
        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(ana_ocr, 0.40),
            ),
            patch("metin_ayiklama._gocmen_sablon_dikey_kaymasi", return_value=0.0),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["soyad"])
        self.assertNotIn("soyad", sonuc["fallback_kullanilan_alanlar"])

    def test_ykn_iki_onislem_varyantindan_checksum_geceni_secer(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [(None, "99539448747", 0.97)],  # tek rakam hatali, checksum gecmez
            [
                (None, "99539448747", 0.99),
                (None, "99539448742", 0.84),  # dogrudan checksum gecen varyant
            ],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(no=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("99539448742", sonuc["tc_no"])
        self.assertEqual(["kimlik_no"], sonuc["fallback_kullanilan_alanlar"])
        self.assertEqual(2, reader.recognize.call_count)
        self.assertEqual(
            2, len(reader.recognize.call_args_list[1].kwargs["horizontal_list"]),
        )

    def test_iki_farkli_checksum_gecerli_ykn_konsensus_yoksa_reddedilir(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [(None, "99539448747", 0.97)],  # ilk hucre okumasi gecersiz
            [
                (None, "99539448742", 0.96),
                (None, "99251771180", 0.99),
            ],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(no=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("Bulunamadi", sonuc["tc_no"])
        self.assertNotIn("kimlik_no", sonuc["fallback_kullanilan_alanlar"])

    def test_net_ad_raw_hucrede_kacarsa_cizgi_temizlemeli_varyant_kurtarir(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [],
            [
                (None, "SOYAM MUSTAFA YOUNUS", 0.89),
                (None, "SOYAM MUSTAFA YOUNUS", 0.93),
            ],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(ad=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("SOYAM MUSTAFA YOUNUS", sonuc["ad"])
        self.assertEqual(["ad"], sonuc["fallback_kullanilan_alanlar"])
        ikinci_cagri = reader.recognize.call_args_list[1]
        self.assertEqual(2, len(ikinci_cagri.kwargs["horizontal_list"]))
        self.assertEqual(2, ikinci_cagri.args[0].ndim)
        self.assertEqual(0.0, ikinci_cagri.kwargs["contrast_ths"])
        self.assertIsNone(ikinci_cagri.kwargs["allowlist"])
        self.assertEqual("0123456789", ikinci_cagri.kwargs["blocklist"])

    def test_yabanci_latin_harfleri_rakam_blocklistiyle_kaybolmaz(self):
        reader = Mock()
        reader.recognize.return_value = [(None, "ÉLODIE O'CONNOR-GARCÍA", 0.91)]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(ad=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("ÉLODIE O'CONNOR-GARCÍA", sonuc["ad"])
        self.assertIsNone(reader.recognize.call_args.kwargs["allowlist"])
        self.assertEqual("0123456789", reader.recognize.call_args.kwargs["blocklist"])

    def test_aksanli_isim_varyant_konsensusunda_korunur(self):
        reader = Mock()
        reader.recognize.side_effect = [
            [],
            [
                (None, "ÉLODIE O'CONNOR-GARCÍA", 0.88),
                (None, "ELODIE O'CONNOR-GARCIA", 0.91),
            ],
        ]

        with (
            patch(
                "metin_ayiklama.easyocr_oku_gocmen",
                return_value=(_gocmen_ocr(ad=False), 0.40),
            ),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("ÉLODIE O'CONNOR-GARCÍA", sonuc["ad"])

    def test_deger_sutunu_mevcut_soyad_kutusundan_ankrajlanir(self):
        reader = Mock()

        def recognize_side_effect(_kart, **kwargs):
            if kwargs["allowlist"] == "0123456789":
                return [(None, "99251771180", 0.91)]
            return [(None, "SOYAM MUSTAFA YOUNUS", 0.92)]

        reader.recognize.side_effect = recognize_side_effect
        ana_ocr = _gocmen_ocr(no=False, ad=False)
        soyad_item = next(item for item in ana_ocr if item["text"] == "KRUT")
        soyad_item["x1"] = 430

        with (
            patch("metin_ayiklama.easyocr_oku_gocmen", return_value=(ana_ocr, 0.40)),
            patch("metin_ayiklama.reader_getir", return_value=reader),
        ):
            sonuc = gocmen_bilgilerini_oku(self.kart)

        self.assertEqual("99251771180", sonuc["tc_no"])
        self.assertEqual("SOYAM MUSTAFA YOUNUS", sonuc["ad"])
        # Mevcut value metni x=430; 1.2% sol payla ortak sutun x=420 olur.
        for cagri in reader.recognize.call_args_list:
            self.assertTrue(all(kutu[0] == 420 for kutu in cagri.kwargs["horizontal_list"]))

    def test_bolunmus_uzun_ad_sutun_baslangicini_saga_kaydirmaz(self):
        ocr = [
            _gocmen_item("ADI", 20, 495, 100, 536),
            _gocmen_item("ILHAM", 376, 495, 445, 536),
            _gocmen_item("MUSTAFA", 455, 495, 540, 536),
            _gocmen_item("YOUNUS", 550, 495, 630, 536),
        ]
        labellar = _gocmen_label_haritasi(ocr)

        x1 = _gocmen_deger_sutunu_sol_siniri(
            self.kart, ocr, {}, labellar,
        )

        self.assertEqual(366, x1)

    def test_tablo_projeksiyonu_cizgileri_hucre_disinda_birakir(self):
        kart = np.full((1100, 800, 3), 255, dtype=np.uint8)
        kart[425:610, 365:368] = 0
        for y in (458, 498, 538, 578):
            kart[y:y + 3, 360:790] = 0
        ocr = [
            _gocmen_item("YABANCI KIMLIK NO", 20, 463, 300, 493),
            _gocmen_item("ADI", 20, 503, 100, 533),
            _gocmen_item("SOYADI", 20, 543, 120, 573),
        ]
        labellar = _gocmen_label_haritasi(ocr)

        x1 = _gocmen_deger_sutunu_sol_siniri(kart, ocr, {}, labellar)
        kutu = _gocmen_hucre_kutusu(
            kart, "ad", labellar["ad"], labellar, x1,
        )

        self.assertGreaterEqual(x1, 368)
        self.assertLessEqual(x1, 371)
        self.assertGreater(kutu[2], 498)
        self.assertLess(kutu[3], 541)

    def test_hucre_onislemi_uzun_cizgiyi_siler_metni_korur(self):
        roi = np.full((40, 120), 255, dtype=np.uint8)
        roi[10, :] = 0
        roi[:, 20] = 0
        roi[22:27, 60:70] = 0

        varyantlar = dict(_gocmen_hucre_onislem_varyantlari(roi))
        kontrast = varyantlar["kontrast"]

        self.assertTrue(np.all(kontrast[20:22, :] == 255))
        self.assertTrue(np.all(kontrast[:, 40:42] == 255))
        self.assertTrue(np.any(kontrast[44:54, 120:140] < 128))

    def test_hucre_onislemi_i_ve_bir_gibi_dikey_karakteri_silmez(self):
        roi = np.full((36, 80), 255, dtype=np.uint8)
        roi[5:31, 30:32] = 0  # yuksekligin %72'si: glyph, tam tablo cizgisi degil

        kontrast = dict(_gocmen_hucre_onislem_varyantlari(roi))["kontrast"]

        self.assertTrue(np.any(kontrast[10:62, 60:64] < 128))

    def test_soyad_hucresi_baba_satirina_tasmaz(self):
        ocr = [
            _gocmen_item("ADI", 20, 495, 100, 536),
            _gocmen_item("SOYADI", 20, 536, 120, 578),
            _gocmen_item("BABA ADI", 20, 578, 140, 620),
        ]
        labellar = _gocmen_label_haritasi(ocr)
        kutu = _gocmen_hucre_kutusu(
            self.kart, "soyad", labellar["soyad"], labellar, int(800 * 0.462),
        )
        soyad_merkez = (536 + 578) / 2
        baba_merkez = (578 + 620) / 2

        self.assertLess(kutu[3], (soyad_merkez + baba_merkez) / 2)


if __name__ == "__main__":
    unittest.main()
