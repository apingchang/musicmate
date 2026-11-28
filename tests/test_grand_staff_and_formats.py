"""測試鋼琴雙手大譜表、300 DPI PDF 高清渲染、調號/低音譜號與 MuseScore 全格式支援 (MusicXML / MSCZ / MIDI / PDF)"""

import os
import tempfile
import unittest
import zipfile

import cv2
import mido
import numpy as np
from PyQt6.QtWidgets import QApplication

from src.core.score_analyzer import ScoreAnalyzer
from src.core.symbolic_score_loader import SymbolicScoreLoader
from src.ui.auto_scroll_page import AutoScrollPage

_app = QApplication.instance() or QApplication([])


class TestGrandStaffAndSymbolicFormats(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_clean_digital_pdf_detection_and_300dpi_render(self):
        """驗證純白向量 PDF 以 300 DPI 清晰渲染且自動跳過翻拍去底色濾鏡"""
        import fitz

        pdf_path = os.path.join(self.tmpdir.name, "clean_vector.pdf")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4 72 DPI pt
        page.insert_text((72, 100), "Sonate No. 14, Moonlight", fontsize=22)
        for i in range(5):
            y = 180 + i * 10
            page.draw_line((72, y), (520, y), color=(0, 0, 0), width=0.8)
        doc.save(pdf_path)
        doc.close()

        ui = AutoScrollPage()
        ok = ui.load_file(pdf_path)
        self.assertTrue(ok)
        self.assertEqual(len(ui._raw_pixmaps), 1)
        # 595 pt @ 300 DPI ≈ 2479 px（遠高於舊版 150 DPI 的 1240 px）
        self.assertGreaterEqual(ui._raw_pixmaps[0].width(), 2200)
        self.assertTrue(ScoreAnalyzer.is_clean_digital_page(ui._raw_rgb_pages[0]))
        # 純白向量 PDF 的增強圖層應直接沿用原抗鋸齒 QPixmap
        self.assertIs(ui._enhanced_pixmaps[0], ui._raw_pixmaps[0])

    def test_musicxml_grand_staff_moonlight_4_sharps(self):
        """驗證 MusicXML (.musicxml) 完整解析 4 個升記號 (C# minor)、雙手大譜表與同拍和弦"""
        xml_path = os.path.join(self.tmpdir.name, "moonlight_m1.musicxml")
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <work><work-title>Moonlight Sonata</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Piano RH</part-name></score-part>
    <score-part id="P2"><part-name>Piano LH</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>6</divisions>
        <key><fifths>4</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction><sound tempo="56"/></direction>
      <note><pitch><step>G</step><alter>1</alter><octave>3</octave></pitch><duration>2</duration><type>eighth</type></note>
      <note><pitch><step>C</step><alter>1</alter><octave>4</octave></pitch><duration>2</duration><type>eighth</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>eighth</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>6</divisions>
        <key><fifths>4</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>F</sign><line>4</line></clef>
      </attributes>
      <note><pitch><step>C</step><alter>1</alter><octave>2</octave></pitch><duration>24</duration><type>whole</type></note>
      <note><chord/><pitch><step>C</step><alter>1</alter><octave>3</octave></pitch><duration>24</duration><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        pages, analysis = SymbolicScoreLoader.load(xml_path)
        self.assertEqual(len(pages), 1)
        self.assertTrue(analysis.is_grand_staff)
        self.assertEqual(analysis.detected_key_fifths, 4)
        self.assertEqual(analysis.detected_time_sig, "4/4")
        self.assertEqual(analysis.detected_bpm, 56)

        # 第 0 拍應同時包含右手 G#3 (MIDI 56) 與左手八度 C#2+C#3 (MIDI 37, 49)
        beat0_events = [e for e in analysis.events if abs(e.beat_offset) < 1e-3]
        self.assertEqual(len(beat0_events), 2)
        rh_ev = [e for e in beat0_events if e.hand == "RH"][0]
        lh_ev = [e for e in beat0_events if e.hand == "LH"][0]
        self.assertEqual(rh_ev.all_midi_notes, [56])
        self.assertEqual(lh_ev.all_midi_notes, [37, 49])
        self.assertIn("C#2", lh_ev.note_name)
        self.assertIn("C#3", lh_ev.note_name)

        # 驗證 UI 可直接載入 MusicXML 並同步拍號與速度
        ui = AutoScrollPage()
        self.assertTrue(ui.load_file(xml_path))
        self.assertEqual(ui._bpm, 56)
        self.assertEqual(ui.time_sig_combo.currentText(), "4/4")

    def test_musescore_mscz_native_loader(self):
        """驗證 MuseScore (.mscz) 壓縮樂譜檔免安裝 MuseScore 即可精準解析雙手與 4♯ 調號"""
        mscz_path = os.path.join(self.tmpdir.name, "moonlight.mscz")
        mscx_xml = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.00">
  <Score>
    <metaTag name="workTitle">Moonlight Test</metaTag>
    <metaTag name="composer">Beethoven</metaTag>
    <Staff id="1">
      <Measure>
        <voice>
          <KeySig><concertKey>4</concertKey></KeySig>
          <TimeSig><sigN>2</sigN><sigD>2</sigD></TimeSig>
          <Tempo><tempo>1.0</tempo></Tempo>
          <Chord><durationType>eighth</durationType><Note><pitch>56</pitch></Note></Chord>
          <Chord><durationType>eighth</durationType><Note><pitch>61</pitch></Note></Chord>
          <Chord><durationType>eighth</durationType><Note><pitch>64</pitch></Note></Chord>
        </voice>
      </Measure>
    </Staff>
    <Staff id="2">
      <Measure>
        <voice>
          <KeySig><concertKey>4</concertKey></KeySig>
          <TimeSig><sigN>2</sigN><sigD>2</sigD></TimeSig>
          <Chord>
            <durationType>whole</durationType>
            <Note><pitch>37</pitch></Note>
            <Note><pitch>49</pitch></Note>
          </Chord>
        </voice>
      </Measure>
    </Staff>
  </Score>
</museScore>
"""
        with zipfile.ZipFile(mscz_path, "w") as zf:
            zf.writestr("score.mscx", mscx_xml)

        pages, analysis = SymbolicScoreLoader.load(mscz_path)
        self.assertEqual(len(pages), 1)
        self.assertTrue(analysis.is_grand_staff)
        self.assertEqual(analysis.detected_key_fifths, 4)
        self.assertEqual(analysis.detected_time_sig, "2/2")
        self.assertEqual(analysis.detected_bpm, 60)

        ui = AutoScrollPage()
        self.assertTrue(ui.load_file(mscz_path))
        self.assertEqual(ui.time_sig_combo.currentText(), "2/2")
        self.assertEqual(ui._bpm, 60)

    def test_midi_dual_track_piano_loader(self):
        """驗證雙軌鋼琴 MIDI (.mid) 自動拆分為雙手大譜表並同步播放"""
        mid_path = os.path.join(self.tmpdir.name, "piano_duet.mid")
        mid = mido.MidiFile(ticks_per_beat=480)
        tr_rh = mido.MidiTrack()
        tr_rh.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(72), time=0))
        tr_rh.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tr_rh.append(mido.MetaMessage("key_signature", key="C#m", time=0))
        for note in (61, 64, 68, 73):
            tr_rh.append(mido.Message("note_on", note=note, velocity=80, time=0))
            tr_rh.append(mido.Message("note_off", note=note, velocity=0, time=480))
        mid.tracks.append(tr_rh)

        tr_lh = mido.MidiTrack()
        tr_lh.append(mido.Message("note_on", note=37, velocity=75, time=0))
        tr_lh.append(mido.Message("note_on", note=49, velocity=75, time=0))
        tr_lh.append(mido.Message("note_off", note=37, velocity=0, time=1920))
        tr_lh.append(mido.Message("note_off", note=49, velocity=0, time=0))
        mid.tracks.append(tr_lh)
        mid.save(mid_path)

        pages, analysis = SymbolicScoreLoader.load(mid_path)
        self.assertEqual(len(pages), 1)
        self.assertTrue(analysis.is_grand_staff)
        self.assertEqual(analysis.detected_key_fifths, 4)
        self.assertEqual(analysis.detected_bpm, 72)
        lh_events = [e for e in analysis.events if e.hand == "LH"]
        self.assertTrue(any(set(e.all_midi_notes) == {37, 49} for e in lh_events))

    def test_key_signature_and_bass_clef_pitch_mapping(self):
        """驗證 4 個升記號 (F#, C#, G#, D#) 在高音譜號與低音譜號之音高換算精準度"""
        sp = 16.0
        y_bot = 200.0
        # 高音譜號：下加一線 C4 (step = -2) 在 4♯ 調號下必為 C#4 (MIDI 61)
        midi_c4s, name_c4s = ScoreAnalyzer._y_to_pitch(y_bot + sp, y_bot, sp, clef="treble", key_fifths=4)
        self.assertEqual((midi_c4s, name_c4s), (61, "C#4"))
        # 高音譜號：第 1 線 E4 (step = 0) 在 4♯ 調號下維持 E4 (MIDI 64)
        midi_e4, name_e4 = ScoreAnalyzer._y_to_pitch(y_bot, y_bot, sp, clef="treble", key_fifths=4)
        self.assertEqual((midi_e4, name_e4), (64, "E4"))
        # 低音譜號：第 2 間 C3 (step = 3) 在 4♯ 調號下必為 C#3 (MIDI 49)
        midi_c3s, name_c3s = ScoreAnalyzer._y_to_pitch(y_bot - 1.5 * sp, y_bot, sp, clef="bass", key_fifths=4)
        self.assertEqual((midi_c3s, name_c3s), (49, "C#3"))

    def test_image_omr_grand_staff_pairing(self):
        """驗證影像 OMR 偵測到左端貫穿兩行的鋼琴大譜表時，自動配對為雙手系統並將下行切換為低音譜號"""
        w, h = 1400, 420
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        sp = 14
        y_rh_top, y_lh_top = 80, 230
        y_rh_bot = y_rh_top + 4 * sp
        y_lh_bot = y_lh_top + 4 * sp
        xl, xr = 90, 1280
        for i in range(5):
            cv2.line(img, (xl, y_rh_top + i * sp), (xr, y_rh_top + i * sp), (25, 25, 25), 2)
            cv2.line(img, (xl, y_lh_top + i * sp), (xr, y_lh_top + i * sp), (25, 25, 25), 2)
        # 左端貫穿雙行的系統直線 + 右端貫穿雙行的小節線
        cv2.line(img, (xl, y_rh_top), (xl, y_lh_bot), (20, 20, 20), 3)
        cv2.line(img, (xr, y_rh_top), (xr, y_lh_bot), (20, 20, 20), 2)
        # 右手 E5 + 左手 C3
        cv2.ellipse(img, (420, int(y_rh_bot - 3.5 * sp)), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (413, int(y_rh_bot - 3.5 * sp)), (413, y_rh_bot + 10), (15, 15, 15), 2)
        cv2.ellipse(img, (420, int(y_lh_bot - 1.5 * sp)), (8, 6), -18, 0, 360, (15, 15, 15), -1)
        cv2.line(img, (413, int(y_lh_bot - 1.5 * sp)), (413, y_lh_bot + 16), (15, 15, 15), 2)

        an = ScoreAnalyzer()
        res = an.analyze_pages([img], beats_per_measure=4.0)
        self.assertTrue(res.is_grand_staff)
        self.assertEqual(len(res.pages[0].staves), 2)
        self.assertEqual(res.pages[0].staves[0].clef, "treble")
        self.assertEqual(res.pages[0].staves[1].clef, "bass")
        self.assertEqual(res.pages[0].staves[0].paired_staff_index, 1)
        self.assertEqual(res.pages[0].staves[1].paired_staff_index, 0)
        lh_notes = [e for e in res.events if e.hand == "LH" and not e.is_rest]
        self.assertTrue(len(lh_notes) >= 1)
        self.assertEqual(lh_notes[0].note_name, "C3")

    def test_pdf_vector_per_measure_segmented_staff_lines(self):
        """驗證 MuseScore PDF 逐小節分段繪製的五線譜線（每小節 < 0.35*pw）能完整串接為整行不漏任何小節"""
        from src.core.pdf_vector_analyzer import PDFVectorAnalyzer

        pw, ph = 595.0, 842.0
        # 模擬 3 小節分段繪製的雙手大譜表（每小節寬度 165pt ≈ 0.27*pw < 0.35*pw）
        h_segs = []
        m_edges = (50.0, 215.0, 380.0, 545.0)
        for y_base in (140.0, 220.0):
            for k in range(5):
                yk = y_base + k * 6.0
                for mi in range(3):
                    h_segs.append((yk, m_edges[mi], m_edges[mi + 1]))

        staves = PDFVectorAnalyzer._group_staff_lines(h_segs, pw, ph)
        self.assertEqual(len(staves), 2)
        self.assertAlmostEqual(staves[0].x_left, 50.0, delta=1.0)
        self.assertAlmostEqual(staves[0].x_right, 545.0, delta=1.0)
        self.assertAlmostEqual(staves[1].x_left, 50.0, delta=1.0)
        self.assertAlmostEqual(staves[1].x_right, 545.0, delta=1.0)

    def test_playhead_monotonic_across_partial_rests_in_measure(self):
        """驗證小節內含部分拍休止符（如四分休止 + 四分休止 + 二分音符）時，紅色游標嚴格由左向右按節拍前進、絕不來回跳動"""
        xml_path = os.path.join(self.tmpdir.name, "partial_rests.musicxml")
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><rest/><duration>4</duration><type>quarter</type></note>
      <note><rest/><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>8</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        ui = AutoScrollPage()
        self.assertTrue(ui.load_file(xml_path))
        ui._is_playing = True
        xs = []
        for step in range(40):
            ui._current_beat = step * 0.1
            ui._sync_playhead_visual(scroll_to_staff=False)
            cx = ui.score_view.page_labels[0]._cursor_x_norm
            self.assertIsNotNone(cx)
            xs.append(cx)
        # 驗證整個小節內 cursor_x 嚴格單調遞增（絕不回彈！）
        for a, b in zip(xs, xs[1:]):
            self.assertGreaterEqual(b, a - 1e-6)

    def test_pdf_vector_long_ledger_stem_not_barline(self):
        """驗證 M3 左手低音 A1+A2 向下突出 3.5*sp 之長符幹絕不被誤判為小節線（不把 M3 拆成 M3+M4）"""
        from src.core.pdf_vector_analyzer import PDFVectorAnalyzer, _VecStaff

        rh = _VecStaff(y_lines=[100.0, 106.0, 112.0, 118.0, 124.0], x_left=50.0, x_right=540.0, sp=6.0, header_end_x=110.0)
        lh = _VecStaff(y_lines=[170.0, 176.0, 182.0, 188.0, 194.0], x_left=50.0, x_right=540.0, sp=6.0, header_end_x=110.0)
        v_segs = [
            (280.0, 100.0, 194.0),  # 真實跨雙行小節線
            (310.0, 170.0, 216.0),  # M3 左手 A1+A2 長符幹（突出底線 22pt ≈ 3.6*sp！）
            (540.0, 100.0, 194.0),  # 行尾貫穿小節線
        ]
        PDFVectorAnalyzer._find_staff_barlines(lh, v_segs, sys_top=100.0, sys_bot=194.0, is_multi_staff=True)
        self.assertEqual(len(lh.barlines_x), 2)
        self.assertNotIn(310.0, lh.barlines_x)

    def test_click_seek_during_playback_no_past_note_burst(self):
        """驗證播放中滑鼠點擊跳轉至後方小節時，只會觸發目標拍點音符，絕不一次爆播前面全部舊音導致破音"""
        xml_path = os.path.join(self.tmpdir.name, "seek_burst.musicxml")
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>5</octave></pitch><duration>8</duration><type>half</type></note>
      <note><pitch><step>A</step><octave>5</octave></pitch><duration>8</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        ui = AutoScrollPage()
        self.assertTrue(ui.load_file(xml_path))
        played = []
        ui._synth.play_note = lambda m, duration_sec=0.35, volume_scale=1.0, timbre=None: played.append(m)
        ui._synth.play_chord = lambda ms, duration_sec=0.35, volume_scale=1.0, timbre=None: played.extend(ms)
        ui._is_playing = True
        ui._last_played_event_idx = 0

        # 點擊第 2 小節最後一顆音 (A5 = MIDI 81)
        target_ev = ui._analysis.events[-1]
        ui._on_score_clicked(target_ev.page_index, target_ev.x_norm, target_ev.y_norm)
        # 只能觸發目標音 A5 (81)，絕不可重播前面的 C5/D5/E5/F5/G5！
        self.assertEqual(played, [81])

    def test_multi_instrument_ensemble_3_parts_and_multiline_status(self):
        """驗證多種樂器重奏總譜（長笛 + 小提琴 + 大提琴 3 聲部）完整解析、對應各自音色並於狀態列顯示 3 行"""
        xml_path = os.path.join(self.tmpdir.name, "trio.musicxml")
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <work><work-title>Chamber Trio</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Flute</part-name></score-part>
    <score-part id="P2"><part-name>Violin</part-name></score-part>
    <score-part id="P3"><part-name>Cello</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>C</step><octave>6</octave></pitch><duration>16</duration><type>whole</type></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>16</duration><type>whole</type></note>
    </measure>
  </part>
  <part id="P3">
    <measure number="1">
      <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>F</sign><line>4</line></clef></attributes>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>16</duration><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        ui = AutoScrollPage()
        self.assertTrue(ui.load_file(xml_path))
        staves = ui._analysis.pages[0].staves
        self.assertEqual(len(staves), 3)
        self.assertEqual([s.instrument_timbre for s in staves], ["長笛", "小提琴", "大提琴"])
        self.assertEqual(staves[0].system_staff_indices, [0, 1, 2])
        # 狀態列應依樂器數量自動顯示 3 行
        lines = [ln for ln in ui.note_detail_label.text().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn("Flute", lines[0])
        self.assertIn("Violin", lines[1])
        self.assertIn("Cello", lines[2])


if __name__ == "__main__":
    unittest.main()
