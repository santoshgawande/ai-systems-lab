import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "01-whisper"))
from whisper_stt import make_sine_wav

class TestAudioTTS(unittest.TestCase):
    def test_make_sine_wav_valid_header(self):
        wav = make_sine_wav(frequency=440.0, duration=0.1, sample_rate=16000)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav)
        self.assertIn(b"fmt ", wav)
        self.assertIn(b"data", wav)
        # 16000 * 0.1 * 2 = 3200 bytes data + 44 bytes header = 3244
        self.assertEqual(len(wav), 44 + 3200)

if __name__ == "__main__":
    unittest.main()
