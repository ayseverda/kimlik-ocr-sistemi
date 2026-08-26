import unittest

from kimlikler.kimlikler_eski_tc import (
    eski_tc_bilgilerini_bul,
    eski_tc_no_bul,
    herhangi_bir_label_mi,
    isim_value_adayi_mi,
)


GECERLI_TC = "10000000146"


def ocr_item(text, x, y, width=80, height=20, conf=0.90):
    return {
        "text": text,
        "conf": conf,
        "x1": x,
        "y1": y,
        "x2": x + width,
        "y2": y + height,
    }


class EskiTcNoTestleri(unittest.TestCase):
    def test_checksum_gecmeyen_on_bir_hane_kabul_edilmez(self):
        gecersiz_item = ocr_item("12345678901", 180, 10, width=150)

        tc_no, tc_item = eski_tc_no_bul([gecersiz_item])

        self.assertEqual("Bulunamadi", tc_no)
        self.assertIsNone(tc_item)

    def test_checksum_gecen_numara_kabul_edilir(self):
        gecerli_item = ocr_item(GECERLI_TC, 180, 10, width=150)

        tc_no, tc_item = eski_tc_no_bul([gecerli_item])

        self.assertEqual(GECERLI_TC, tc_no)
        self.assertIs(gecerli_item, tc_item)


class EtiketTestleri(unittest.TestCase):
    def test_etiket_alt_dizisi_iceren_adlar_deger_adayidir(self):
        # HAKAN -> KAN, CANAN -> ANA ve SERIF -> SERI alt dizilerini içerir.
        for ad in ("HAKAN", "CANAN", "SERIF", "ADIL"):
            with self.subTest(ad=ad):
                item = ocr_item(ad, 180, 50)
                self.assertFalse(herhangi_bir_label_mi(ad))
                self.assertTrue(isim_value_adayi_mi(item))

    def test_bilinen_etiketler_hala_taninir(self):
        for etiket in (
            "ADI", "SOYADI", "T.C. KIMLIK NO", "BABA ADI", "ANA ADI",
            "DOGUM TARIHI", "SERI NO", "KAN GRUBU", "ADI HAKAN",
            "SOYADI CANAN", "BABA AHMET",
        ):
            with self.subTest(etiket=etiket):
                self.assertTrue(herhangi_bir_label_mi(etiket))


class AdSoyadAlanTestleri(unittest.TestCase):
    def test_tam_etiketli_kayitta_ad_ve_soyad_dogru_ayrilir(self):
        ocr = [
            ocr_item("T.C. KIMLIK NO", 10, 10, width=130),
            ocr_item(GECERLI_TC, 180, 10, width=150),
            ocr_item("SOYADI", 10, 50, width=70),
            ocr_item("CANAN", 180, 50),
            ocr_item("ADI", 10, 90, width=45),
            ocr_item("HAKAN", 180, 90),
            ocr_item("BABA ADI", 10, 130, width=100),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual(GECERLI_TC, sonuc["tc_no"])
        self.assertEqual("HAKAN", sonuc["ad"])
        self.assertEqual("CANAN", sonuc["soyad"])

    def test_etiketsiz_ilk_iki_deger_satiri_soyad_ve_ad_olur(self):
        ocr = [
            ocr_item(GECERLI_TC, 180, 10, width=150),
            ocr_item("SERIF", 180, 50),
            ocr_item("HAKAN", 180, 90),
            ocr_item("BABA ADI", 10, 130, width=100),
            ocr_item("AHMET", 180, 130),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual("SERIF", sonuc["soyad"])
        self.assertEqual("HAKAN", sonuc["ad"])

    def test_dogru_ad_satiri_eksik_soyada_tasinmaz(self):
        ocr = [
            ocr_item(GECERLI_TC, 180, 10, width=150),
            ocr_item("ADI", 10, 90, width=45),
            ocr_item("HAKAN", 180, 90),
            ocr_item("BABA ADI", 10, 130, width=100),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual("HAKAN", sonuc["ad"])
        self.assertEqual("Bulunamadi", sonuc["soyad"])

    def test_komsu_soyad_etiketi_ad_degerini_sahiplenmez(self):
        ocr = [
            ocr_item(GECERLI_TC, 180, 10, width=150),
            ocr_item("SOYADI", 10, 50, width=70),
            ocr_item("ADI", 10, 76, width=45),
            ocr_item("HAKAN", 180, 76),
            ocr_item("BABA ADI", 10, 120, width=100),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual("HAKAN", sonuc["ad"])
        self.assertEqual("Bulunamadi", sonuc["soyad"])

    def test_baba_adi_eksik_ad_icin_fallback_olmaz(self):
        ocr = [
            ocr_item(GECERLI_TC, 180, 10, width=150),
            ocr_item("SERIF", 180, 50),
            ocr_item("BABA ADI", 10, 90, width=100),
            ocr_item("AHMET", 180, 90),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual("SERIF", sonuc["soyad"])
        self.assertEqual("Bulunamadi", sonuc["ad"])

    def test_bozuk_sol_sutun_etiketi_isimleri_bir_satir_kaydirmiyor(self):
        # Gercek ekran goruntusundeki hata: SOYADI etiketi OCR'da "VIU" olur.
        # Metin olarak etikete benzemeyen VIU solda, gercek degerler ise TC ile
        # ayni sag sutundadir. Eski davranis VIU=soyad, KOCYIGIT=ad seciyordu.
        ocr = [
            ocr_item("T.C. KIMLIK NO", 60, 520, width=150, height=25),
            ocr_item(GECERLI_TC, 250, 520, width=210, height=35),
            ocr_item("VIU", 70, 580, width=70, height=18),
            ocr_item("KOÇYİĞİT", 280, 615, width=190, height=34),
            ocr_item("FERUZE", 300, 690, width=145, height=32),
            ocr_item("BABA ADI", 70, 755, width=130, height=22),
            ocr_item("AHMET", 300, 755, width=110, height=30),
        ]

        sonuc = eski_tc_bilgilerini_bul(ocr)

        self.assertEqual("KOÇYİĞİT", sonuc["soyad"])
        self.assertEqual("FERUZE", sonuc["ad"])
        self.assertGreaterEqual(sonuc["soyad_item"]["x1"], 150)
        self.assertGreaterEqual(sonuc["ad_item"]["x1"], 150)


if __name__ == "__main__":
    unittest.main()
