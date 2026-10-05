"""調音器音高偵測與穩定性測試"""

import unittest
import numpy as np
from src.core.tuner import Tuner


class TestTunerPitch(unittest.TestCase):
    def setUp(self):
        self.tuner = Tuner()
        self.sample_rate = 44100

    def _generate_sine(self, freq: float, duration_samples: int = 4096):
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples, endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32)

    def test_frequencies_accuracy(self):
        # 測試涵蓋低頻 (E2)、中頻 (C4, A4)、高頻 (C6, C7, C8)
        for target_freq in [82.41, 130.81, 261.63, 440.0, 1000.0, 2093.0, 4186.0]:
            sig = self._generate_sine(target_freq)
            det = Tuner.detect_pitch(sig, self.sample_rate)
            self.assertIsNotNone(det, f"Should detect pitch for {target_freq}Hz")
            # 容許誤差在 1% 以內
            self.assertAlmostEqual(det, target_freq, delta=target_freq * 0.01)

    def test_harmonic_stability_c4(self):
        # 測試 261.63 Hz 混入二次諧波 (523.25Hz) 與三次諧波
        t = np.linspace(0, 4096 / self.sample_rate, 4096, endpoint=False)
        f0 = 261.63
        sig = (np.sin(2 * np.pi * f0 * t) + 0.6 * np.sin(2 * np.pi * 2 * f0 * t) + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)).astype(np.float32)
        det = Tuner.detect_pitch(sig, self.sample_rate)
        self.assertIsNotNone(det)
        self.assertAlmostEqual(det, f0, delta=1.5)

    def test_median_filter_smoothing(self):
        # 模擬連續音訊幀傳遞給 _process_audio_frame
        detected_results = []
        def callback(note, octv, cents, freq, in_tune):
            if freq > 0:
                detected_results.append((note, octv, freq, in_tune))

        self.tuner.set_pitch_callback(callback)

        # 模擬 5 幀 440Hz，其中一幀混入諧波跳躍干擾
        s440 = self._generate_sine(440.0)
        s_glitch = self._generate_sine(880.0)  # 突發倍頻干擾

        for i in range(5):
            frame = s_glitch if i == 2 else s440
            self.tuner._process_audio_frame(frame)

        # 中值濾波後，第 5 幀輸出應穩定維持在 A4 (440Hz)，不受第 2 幀的 880Hz 影響
        last_note, last_octv, last_freq, last_in_tune = detected_results[-1]
        self.assertEqual(last_note, "A")
        self.assertEqual(last_octv, 4)
        self.assertAlmostEqual(last_freq, 440.0, delta=2.0)
        self.assertTrue(last_in_tune)


if __name__ == "__main__":
    unittest.main()
