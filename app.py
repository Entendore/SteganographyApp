#!/usr/bin/env python3
"""
Steganography Learning Application
Learn, practice, and visualize image & text steganography techniques.
Built with PySide6, Pillow, and NumPy.
"""

import sys
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog, QGroupBox,
    QComboBox, QSpinBox, QScrollArea, QFrame, QStackedWidget,
    QGridLayout, QSlider, QSizePolicy, QMessageBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QFont, QColor, QPainter, QPalette
)

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ════════════════════════════════════════════════════════════
# CORE STEGANOGRAPHY FUNCTIONS
# ════════════════════════════════════════════════════════════

DELIMITER = '1111111111111110'  # 16-bit end marker


def text_to_binary(text: str) -> str:
    """Convert text to binary using UTF-8 encoding."""
    return ''.join(format(b, '08b') for b in text.encode('utf-8'))


def binary_to_text(binary_str: str) -> str:
    """Convert binary string back to text using UTF-8 decoding."""
    byte_list = []
    for i in range(0, len(binary_str) - 7, 8):
        byte = binary_str[i:i + 8]
        try:
            byte_list.append(int(byte, 2))
        except ValueError:
            pass
    try:
        return bytes(byte_list).decode('utf-8', errors='replace')
    except Exception:
        return ""


def get_image_capacity(image_path: str, bits_per_channel: int = 1) -> int:
    """Return maximum characters that can be hidden in the image."""
    img = PILImage.open(image_path).convert('RGB')
    w, h = img.size
    total_channels = w * h * 3
    usable_bits = total_channels * bits_per_channel - 16  # minus delimiter
    return max(0, usable_bits // 8)


def encode_lsb(image_path: str, message: str, output_path: str,
               bits_per_channel: int = 1):
    """
    Encode a text message into an image using LSB steganography.
    Returns (total_bits_written, message_chars).
    """
    img = PILImage.open(image_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)

    binary_msg = text_to_binary(message) + DELIMITER

    total_channels = pixels.shape[0] * pixels.shape[1] * 3
    max_bits = total_channels * bits_per_channel

    if len(binary_msg) > max_bits:
        raise ValueError(
            f"Message too long! Need {len(binary_msg)} bits but "
            f"image can only hold {max_bits} bits ({max_bits // 8} chars)."
        )

    flat = pixels.flatten().copy()
    clear_mask = 0xFF & ~((1 << bits_per_channel) - 1)

    bit_idx = 0
    for i in range(len(flat)):
        if bit_idx >= len(binary_msg):
            break
        chunk = binary_msg[bit_idx:bit_idx + bits_per_channel]
        if len(chunk) < bits_per_channel:
            chunk = chunk.ljust(bits_per_channel, '0')
        flat[i] = (flat[i] & clear_mask) | int(chunk, 2)
        bit_idx += bits_per_channel

    encoded = flat.reshape(pixels.shape)
    PILImage.fromarray(encoded.astype(np.uint8)).save(output_path, 'PNG')
    return bit_idx, len(message)


def decode_lsb(image_path: str, bits_per_channel: int = 1) -> str | None:
    """Decode a hidden message from an image. Returns None if no delimiter found."""
    img = PILImage.open(image_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)

    flat = pixels.flatten()
    extract_mask = (1 << bits_per_channel) - 1
    binary_msg = ""

    for i in range(len(flat)):
        extracted = int(flat[i]) & extract_mask
        binary_msg += format(extracted, f'0{bits_per_channel}b')
        if len(binary_msg) >= 16 and binary_msg[-16:] == DELIMITER:
            return binary_to_text(binary_msg[:-16])

    return None


# ── Text steganography with zero-width characters ──

ZW_SPACE = '\u200B'       # represents bit 0
ZW_NON_JOINER = '\u200C'  # represents bit 1


def encode_zero_width(cover_text: str, secret: str) -> str:
    """Hide a secret message inside cover text using zero-width characters."""
    binary_secret = text_to_binary(secret)
    result = []
    bit_idx = 0

    for char in cover_text:
        result.append(char)
        for _ in range(8):
            if bit_idx < len(binary_secret):
                result.append(
                    ZW_SPACE if binary_secret[bit_idx] == '0'
                    else ZW_NON_JOINER
                )
                bit_idx += 1

    # Flush remaining bits
    while bit_idx < len(binary_secret):
        result.append(
            ZW_SPACE if binary_secret[bit_idx] == '0'
            else ZW_NON_JOINER
        )
        bit_idx += 1

    return ''.join(result)


def decode_zero_width(stego_text: str) -> str:
    """Extract a hidden message from text with zero-width characters."""
    binary_msg = ''.join(
        '0' if c == ZW_SPACE else '1'
        for c in stego_text
        if c in (ZW_SPACE, ZW_NON_JOINER)
    )
    return binary_to_text(binary_msg) if binary_msg else ""


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert an H×W×3 uint8 numpy array to QPixmap."""
    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def pil_to_qpixmap(pil_img) -> QPixmap:
    """Convert a PIL Image to QPixmap."""
    return numpy_to_qpixmap(np.array(pil_img.convert('RGB')))


# ════════════════════════════════════════════════════════════
# STYLESHEET (Dark Theme)
# ════════════════════════════════════════════════════════════

STYLESHEET = """
QMainWindow { background-color: #1a1a2e; }

QListWidget {
    background-color: #16213e; border: none; border-radius: 10px;
    padding: 5px; font-size: 14px;
}
QListWidget::item { color: #e0e0e0; padding: 12px 15px; border-radius: 8px; margin: 2px 5px; }
QListWidget::item:selected { background-color: #0f3460; color: #00d2ff; font-weight: bold; }
QListWidget::item:hover   { background-color: #1a1a4e; }

QLabel { color: #e0e0e0; }

QGroupBox {
    color: #00d2ff; border: 1px solid #0f3460; border-radius: 8px;
    margin-top: 12px; padding-top: 15px; font-weight: bold; font-size: 13px;
}
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; }

QPushButton {
    background-color: #0f3460; color: #e0e0e0; border: 1px solid #1a1a5e;
    border-radius: 6px; padding: 8px 18px; font-size: 13px; font-weight: bold;
}
QPushButton:hover   { background-color: #1a4a8a; border-color: #00d2ff; }
QPushButton:pressed { background-color: #00d2ff; color: #1a1a2e; }
QPushButton:disabled { background-color: #2a2a3e; color: #555; border-color: #333; }

QTextEdit {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 6px; padding: 8px; font-size: 13px;
    selection-background-color: #0f3460;
}
QLineEdit {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 6px; padding: 8px; font-size: 13px;
}
QLineEdit:focus { border-color: #00d2ff; }

QComboBox {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 6px; padding: 8px; font-size: 13px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #16213e; color: #e0e0e0;
    selection-background-color: #0f3460;
}

QSpinBox {
    background-color: #16213e; color: #e0e0e0; border: 1px solid #0f3460;
    border-radius: 6px; padding: 5px; font-size: 13px;
}

QSlider::groove:horizontal {
    border: 1px solid #0f3460; height: 6px; background: #16213e; border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00d2ff; border: none; width: 16px; margin: -5px 0; border-radius: 8px;
}

QProgressBar {
    background-color: #16213e; border: 1px solid #0f3460; border-radius: 6px;
    text-align: center; color: #e0e0e0; font-weight: bold;
}
QProgressBar::chunk { background-color: #00d2ff; border-radius: 5px; }

QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical { background: #1a1a2e; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #0f3460; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QTabWidget::pane { border: 1px solid #0f3460; border-radius: 8px; background-color: #1a1a2e; }
QTabBar::tab {
    background-color: #16213e; color: #e0e0e0; padding: 8px 20px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px;
}
QTabBar::tab:selected { background-color: #0f3460; color: #00d2ff; font-weight: bold; }
"""


# ════════════════════════════════════════════════════════════
# REUSABLE CUSTOM WIDGETS
# ════════════════════════════════════════════════════════════

class ImageDropLabel(QLabel):
    """Label that shows an image and accepts drag-and-drop / click-to-browse."""
    imageSelected = Signal(str)

    def __init__(self, placeholder="Drop image here\nor click to browse", parent=None):
        super().__init__(placeholder, parent)
        self._path = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QLabel { background-color: #16213e; border: 2px dashed #0f3460;
                     border-radius: 10px; color: #666; font-size: 14px; padding: 20px; }
            QLabel:hover { border-color: #00d2ff; }
        """)

    @property
    def imagePath(self):
        return self._path

    def _load(self, path):
        self._path = path
        px = QPixmap(path)
        if not px.isNull():
            self.setPixmap(px.scaled(self.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))
            self.imageSelected.emit(path)

    def mousePressEvent(self, ev):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)")
        if p:
            self._load(p)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                self._load(p)
                break

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._path:
            px = QPixmap(self._path)
            if not px.isNull():
                self.setPixmap(px.scaled(self.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation))


class SectionHeader(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "color:#00d2ff; font-size:18px; font-weight:bold;"
            "padding:5px 0; border-bottom:2px solid #0f3460;")


class InfoLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet("color:#b0b0c0; font-size:13px; padding:5px;")


class StepCard(QFrame):
    def __init__(self, number, title, desc, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame{background:#16213e;border:1px solid #0f3460;"
            "border-radius:10px;padding:15px;}")
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        hdr = QHBoxLayout()
        num = QLabel(str(number))
        num.setFixedSize(34, 34)
        num.setAlignment(Qt.AlignCenter)
        num.setStyleSheet(
            "background:#00d2ff;color:#1a1a2e;border-radius:17px;"
            "font-size:15px;font-weight:bold;")
        hdr.addWidget(num)
        t = QLabel(title)
        t.setStyleSheet("color:#fff;font-size:15px;font-weight:bold;")
        hdr.addWidget(t)
        hdr.addStretch()
        lay.addLayout(hdr)
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet("color:#b0b0c0;font-size:13px;padding-left:44px;")
        lay.addWidget(d)


# ════════════════════════════════════════════════════════════
# PAGE 1 — LEARN
# ════════════════════════════════════════════════════════════

class LearnPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(20)
        lay.setContentsMargins(30, 20, 30, 20)

        lay.addWidget(self._title("📚 Learn Steganography"))

        lay.addWidget(InfoLabel(
            "Steganography is the art and science of hiding information in plain sight. "
            "Unlike encryption which scrambles data to make it unreadable, steganography "
            "hides the <i>very existence</i> of the message. The word comes from Greek: "
            "<b>steganos</b> (covered) and <b>graphein</b> (writing)."
        ))

        # ── Comparison cards ──
        lay.addWidget(SectionHeader("🔒 Steganography vs Cryptography"))
        row = QHBoxLayout()
        for title, color, items in [
            ("Steganography", "#00d2ff", [
                "• Hides the existence of a message",
                "• Message is invisible to observers",
                "• Cover medium looks normal",
                "• Detectable with statistical analysis",
                "• Often combined with encryption",
            ]),
            ("Cryptography", "#ff6b6b", [
                "• Scrambles the message content",
                "• Message is visible but unreadable",
                "• Encrypted data looks random",
                "• Breakable with brute force",
                "• Protects message confidentiality",
            ]),
        ]:
            card = QFrame()
            card.setStyleSheet(
                "QFrame{background:#16213e;border:1px solid #0f3460;"
                "border-radius:10px;padding:15px;}")
            cl = QVBoxLayout(card)
            t = QLabel(title)
            t.setStyleSheet(f"color:{color};font-size:16px;font-weight:bold;")
            cl.addWidget(t)
            d = QLabel("\n".join(items))
            d.setStyleSheet("color:#b0b0c0;font-size:13px;")
            cl.addWidget(d)
            row.addWidget(card)
        lay.addLayout(row)

        # ── How LSB works ──
        lay.addWidget(SectionHeader("🔧 How LSB Steganography Works"))
        lay.addWidget(InfoLabel(
            "LSB (Least Significant Bit) steganography is the most common image "
            "hiding technique. It modifies the least-significant bit of each colour "
            "channel value — a change of only 0 or 1, imperceptible to the human eye."
        ))
        steps = [
            ("Convert Message to Binary",
             "The secret text is encoded to a binary string using UTF-8. "
             "For example, 'A' → 01000001."),
            ("Read Pixel Values",
             "Each pixel in an RGB image stores three 8-bit values (0–255) for "
             "Red, Green, and Blue. A pixel might be (154, 203, 87)."),
            ("Replace LSB Bits",
             "The least-significant bit of each channel is replaced with one bit "
             "of the secret. Changing 154→154 or 154→155 differs by at most 1 "
             "— invisible to the naked eye!"),
            ("Add Delimiter",
             "A 16-bit marker (1111111111111110) is appended so the decoder knows "
             "where the hidden data ends."),
            ("Save as PNG",
             "The image must be saved in a <b>lossless</b> format (PNG). JPEG "
             "compression would destroy the hidden bits!"),
        ]
        for i, (t, d) in enumerate(steps, 1):
            lay.addWidget(StepCard(i, t, d))

        # ── Interactive demo ──
        lay.addWidget(SectionHeader("🧪 Interactive Demo: See LSB in Action"))
        demo = QFrame()
        demo.setStyleSheet(
            "QFrame{background:#16213e;border:1px solid #0f3460;"
            "border-radius:10px;padding:20px;}")
        dl = QVBoxLayout(demo)
        dl.addWidget(InfoLabel(
            "Enter a pixel value (0-255) and a secret bit to see how "
            "LSB encoding works:"))
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Pixel value:"))
        self._px = QSpinBox(); self._px.setRange(0, 255); self._px.setValue(154)
        row2.addWidget(self._px)
        row2.addWidget(QLabel("Secret bit:"))
        self._bit = QComboBox(); self._bit.addItems(["0", "1"])
        row2.addWidget(self._bit)
        b = QPushButton("Encode"); b.clicked.connect(self._demo); row2.addWidget(b)
        row2.addStretch()
        dl.addLayout(row2)
        self._result = QLabel()
        self._result.setWordWrap(True)
        self._result.setStyleSheet(
            "color:#e0e0e0;font-size:14px;font-family:monospace;"
            "padding:15px;background:#0f1e3e;border-radius:8px;")
        dl.addWidget(self._result)
        lay.addWidget(demo)

        # ── Types ──
        lay.addWidget(SectionHeader("📂 Types of Steganography"))
        for icon, t, d in [
            ("🖼️", "Image Steganography",
             "Hides data within pixel values. LSB substitution is the most common. "
             "The cover image must be large enough to hold the payload."),
            ("📝", "Text Steganography",
             "Hides data using zero-width characters, whitespace encoding, or "
             "linguistic methods. The text appears identical before and after."),
            ("🎵", "Audio Steganography",
             "Hides data within audio samples in ways inaudible to human ears. "
             "LSB and echo-hiding are common techniques."),
            ("🎬", "Video Steganography",
             "Combines image and audio steganography across video frames, "
             "providing much larger hiding capacity."),
        ]:
            card = QFrame()
            card.setStyleSheet(
                "QFrame{background:#16213e;border:1px solid #0f3460;"
                "border-radius:8px;padding:12px;}")
            cl = QVBoxLayout(card)
            cl.addWidget(QLabel(f"{icon}  {t}"))
            cl.itemAt(0).widget().setStyleSheet(
                "color:#00d2ff;font-size:14px;font-weight:bold;")
            lb = QLabel(d); lb.setWordWrap(True)
            lb.setStyleSheet("color:#b0b0c0;font-size:12px;")
            cl.addWidget(lb)
            lay.addWidget(card)

        # ── Steganalysis ──
        lay.addWidget(SectionHeader("🔍 Steganalysis (Detection)"))
        lay.addWidget(InfoLabel(
            "Steganalysis is the science of detecting steganography:\n\n"
            "• <b>Statistical Analysis</b> — LSB encoding shifts the distribution "
            "of pixel values; histogram analysis can reveal anomalies.\n\n"
            "• <b>Chi-squared Attack</b> — Tests for the pairing of even/odd "
            "values that occurs when LSB substitution is used.\n\n"
            "• <b>RS Analysis</b> — Analyses regular/singular pixel groups to "
            "estimate hidden-message length.\n\n"
            "• <b>Visual Attacks</b> — Examining individual bit planes can reveal "
            "LSB patterns. Try the <b>Bit Visualizer</b> tab!"))

        lay.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._demo()

    # helpers
    @staticmethod
    def _title(text):
        lb = QLabel(text)
        lb.setStyleSheet(
            "color:#00d2ff;font-size:28px;font-weight:bold;padding:10px 0;")
        return lb

    def _demo(self):
        v = self._px.value()
        b = self._bit.currentText()
        nv = (v & 0xFE) | int(b)
        self._result.setText(
            f"Original value:  {v:>3d}   Binary: {format(v, '08b')}\n"
            f"Secret bit:        {b}\n"
            f"New value:       {nv:>3d}   Binary: {format(nv, '08b')}\n"
            f"Difference:      |{v}−{nv}| = {abs(v - nv)}  ← imperceptible! 👁️")


# ════════════════════════════════════════════════════════════
# PAGE 2 — IMAGE STEGANOGRAPHY
# ════════════════════════════════════════════════════════════

class ImageStegoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._src = None
        self._dec_src = None
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(15)
        lay.setContentsMargins(25, 20, 25, 20)

        lay.addWidget(self._heading("🖼️ Image Steganography"))

        tabs = QTabWidget()

        # ── Encode ──
        ew = QWidget(); el = QVBoxLayout(ew); el.setSpacing(12)
        el.addWidget(SectionHeader("Step 1 — Select Cover Image"))
        self._enc_img = ImageDropLabel()
        self._enc_img.imageSelected.connect(self._on_enc_img)
        el.addWidget(self._enc_img)
        self._cap_lbl = InfoLabel("No image selected")
        el.addWidget(self._cap_lbl)

        el.addWidget(SectionHeader("Step 2 — Enter Secret Message"))
        self._enc_msg = QTextEdit()
        self._enc_msg.setPlaceholderText("Type your secret message here…")
        self._enc_msg.setMaximumHeight(110)
        self._enc_msg.textChanged.connect(self._update_stats)
        el.addWidget(self._enc_msg)
        self._stat_lbl = InfoLabel("Characters: 0")
        el.addWidget(self._stat_lbl)

        el.addWidget(SectionHeader("Step 3 — Settings"))
        sr = QHBoxLayout()
        sr.addWidget(QLabel("Bits per channel:"))
        self._bpc = QSpinBox(); self._bpc.setRange(1, 4); self._bpc.setValue(1)
        self._bpc.setToolTip("More bits → more capacity but more visible artefacts")
        self._bpc.valueChanged.connect(self._update_cap)
        sr.addWidget(self._bpc); sr.addStretch()
        el.addLayout(sr)

        self._enc_btn = QPushButton("🔒  Encode Message")
        self._enc_btn.setEnabled(False)
        self._enc_btn.setMinimumHeight(42)
        self._enc_btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#00d2ff;font-size:15px;"
            "font-weight:bold;border:2px solid #00d2ff;border-radius:8px;padding:10px;}"
            "QPushButton:hover{background:#00d2ff;color:#1a1a2e;}"
            "QPushButton:disabled{color:#555;border-color:#333;}")
        self._enc_btn.clicked.connect(self._do_encode)
        el.addWidget(self._enc_btn)
        self._enc_status = QLabel()
        self._enc_status.setWordWrap(True)
        self._enc_status.setStyleSheet("color:#4ade80;font-size:13px;padding:5px;")
        el.addWidget(self._enc_status)
        tabs.addTab(ew, "🔒 Encode")

        # ── Decode ──
        dw = QWidget(); dl = QVBoxLayout(dw); dl.setSpacing(12)
        dl.addWidget(SectionHeader("Step 1 — Select Stego Image"))
        self._dec_img = ImageDropLabel()
        self._dec_img.imageSelected.connect(lambda p: setattr(self, '_dec_src', p))
        dl.addWidget(self._dec_img)

        dl.addWidget(SectionHeader("Step 2 — Settings"))
        dr = QHBoxLayout()
        dr.addWidget(QLabel("Bits per channel:"))
        self._dec_bpc = QSpinBox(); self._dec_bpc.setRange(1, 4); self._dec_bpc.setValue(1)
        dr.addWidget(self._dec_bpc); dr.addStretch()
        dl.addLayout(dr)

        self._dec_btn = QPushButton("🔓  Decode Message")
        self._dec_btn.setEnabled(False)
        self._dec_btn.setMinimumHeight(42)
        self._dec_btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#4ade80;font-size:15px;"
            "font-weight:bold;border:2px solid #4ade80;border-radius:8px;padding:10px;}"
            "QPushButton:hover{background:#4ade80;color:#1a1a2e;}"
            "QPushButton:disabled{color:#555;border-color:#333;}")
        self._dec_btn.clicked.connect(self._do_decode)
        dl.addWidget(self._dec_btn)
        self._dec_img.imageSelected.connect(
            lambda: self._dec_btn.setEnabled(True))

        dl.addWidget(SectionHeader("Decoded Message"))
        self._dec_out = QTextEdit(); self._dec_out.setReadOnly(True)
        self._dec_out.setPlaceholderText("Hidden message will appear here…")
        self._dec_out.setMinimumHeight(140)
        dl.addWidget(self._dec_out)
        self._dec_status = QLabel()
        self._dec_status.setWordWrap(True)
        self._dec_status.setStyleSheet("color:#4ade80;font-size:13px;padding:5px;")
        dl.addWidget(self._dec_status)
        tabs.addTab(dw, "🔓 Decode")

        lay.addWidget(tabs)
        lay.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _heading(text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#00d2ff;font-size:24px;font-weight:bold;padding:5px 0;")
        return lb

    def _on_enc_img(self, path):
        self._src = path
        self._update_cap()
        self._enc_btn.setEnabled(True)

    def _update_cap(self):
        if self._src:
            try:
                cap = get_image_capacity(self._src, self._bpc.value())
                img = PILImage.open(self._src)
                w, h = img.size
                self._cap_lbl.setText(
                    f"✅ Image {w}×{h} — capacity ≈ {cap} chars "
                    f"({self._bpc.value()} bit/channel)")
            except Exception as e:
                self._cap_lbl.setText(f"❌ {e}")

    def _update_stats(self):
        t = self._enc_msg.toPlainText()
        bits = len(text_to_binary(t)) + 16 if t else 0
        self._stat_lbl.setText(
            f"Characters: {len(t)}  |  Binary bits needed: {bits} (incl. delimiter)")

    def _do_encode(self):
        if not self._src:
            QMessageBox.warning(self, "Error", "Select a cover image first!"); return
        msg = self._enc_msg.toPlainText()
        if not msg:
            QMessageBox.warning(self, "Error", "Enter a secret message!"); return
        out, _ = QFileDialog.getSaveFileName(
            self, "Save Encoded Image", "stego_image.png", "PNG (*.png)")
        if not out:
            return
        if not out.lower().endswith('.png'):
            out += '.png'
        try:
            bits_w, chars = encode_lsb(self._src, msg, out, self._bpc.value())
            self._enc_status.setText(
                f"✅ Encoded {chars} chars ({bits_w} bits) → saved to {out}")
            self._enc_status.setStyleSheet("color:#4ade80;font-size:13px;padding:5px;")
        except Exception as e:
            self._enc_status.setText(f"❌ {e}")
            self._enc_status.setStyleSheet("color:#ff6b6b;font-size:13px;padding:5px;")

    def _do_decode(self):
        if not self._dec_src:
            QMessageBox.warning(self, "Error", "Select a stego image first!"); return
        try:
            result = decode_lsb(self._dec_src, self._dec_bpc.value())
            if result is not None:
                self._dec_out.setPlainText(result)
                self._dec_status.setText(f"✅ Decoded {len(result)} characters!")
                self._dec_status.setStyleSheet("color:#4ade80;font-size:13px;padding:5px;")
            else:
                self._dec_out.clear()
                self._dec_status.setText(
                    "❌ No hidden message found. Try different bit settings.")
                self._dec_status.setStyleSheet("color:#ff6b6b;font-size:13px;padding:5px;")
        except Exception as e:
            self._dec_status.setText(f"❌ {e}")
            self._dec_status.setStyleSheet("color:#ff6b6b;font-size:13px;padding:5px;")


# ════════════════════════════════════════════════════════════
# PAGE 3 — TEXT STEGANOGRAPHY
# ════════════════════════════════════════════════════════════

class TextStegoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(15)
        lay.setContentsMargins(25, 20, 25, 20)

        lay.addWidget(self._heading("📝 Text Steganography"))
        lay.addWidget(InfoLabel(
            "Text steganography hides secret messages inside normal-looking text "
            "using <b>zero-width characters</b> — invisible Unicode code-points "
            "inserted between visible characters to encode binary data. "
            "The text looks exactly the same before and after encoding!"))

        # How it works
        hf = QFrame()
        hf.setStyleSheet(
            "QFrame{background:#16213e;border:1px solid #0f3460;"
            "border-radius:10px;padding:15px;}")
        hl = QVBoxLayout(hf)
        hl.addWidget(QLabel("How Zero-Width Encoding Works"))
        hl.itemAt(hl.count() - 1).widget().setStyleSheet(
            "color:#00d2ff;font-size:15px;font-weight:bold;")
        hl.addWidget(InfoLabel(
            "1. Convert secret message → binary (UTF-8)\n"
            "2. For each '0' bit → insert U+200B (Zero-Width Space)\n"
            "3. For each '1' bit → insert U+200C (Zero-Width Non-Joiner)\n"
            "4. Characters are invisible but carry hidden data\n"
            "5. Cover text appears completely unchanged!"))
        lay.addWidget(hf)

        tabs = QTabWidget()

        # ── Encode ──
        ew = QWidget(); el = QVBoxLayout(ew); el.setSpacing(10)
        el.addWidget(QLabel("Cover Text (visible text):"))
        self._cover = QTextEdit()
        self._cover.setPlaceholderText("Enter normal-looking text here…")
        self._cover.setMaximumHeight(90)
        el.addWidget(self._cover)
        el.addWidget(QLabel("Secret Message (to hide):"))
        self._secret = QTextEdit()
        self._secret.setPlaceholderText("Enter your secret here…")
        self._secret.setMaximumHeight(70)
        el.addWidget(self._secret)
        self._tenc_btn = QPushButton("🔒 Hide Message in Text")
        self._tenc_btn.setMinimumHeight(36)
        self._tenc_btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#00d2ff;font-size:14px;"
            "font-weight:bold;border:2px solid #00d2ff;border-radius:8px;padding:8px;}"
            "QPushButton:hover{background:#00d2ff;color:#1a1a2e;}")
        self._tenc_btn.clicked.connect(self._do_tenc)
        el.addWidget(self._tenc_btn)
        el.addWidget(QLabel("Stego Text (contains hidden message — looks identical!):"))
        self._stego_out = QTextEdit(); self._stego_out.setReadOnly(True)
        self._stego_out.setPlaceholderText("Encoded text will appear here…")
        el.addWidget(self._stego_out)
        self._tenc_status = QLabel()
        self._tenc_status.setStyleSheet("color:#4ade80;font-size:13px;")
        el.addWidget(self._tenc_status)
        cpb = QPushButton("📋 Copy Stego Text to Clipboard")
        cpb.clicked.connect(self._copy_stego)
        el.addWidget(cpb)
        tabs.addTab(ew, "🔒 Encode")

        # ── Decode ──
        dw = QWidget(); dl = QVBoxLayout(dw); dl.setSpacing(10)
        dl.addWidget(QLabel("Paste Stego Text (with hidden zero-width chars):"))
        self._stego_in = QTextEdit()
        self._stego_in.setPlaceholderText("Paste text with hidden message here…")
        self._stego_in.setMinimumHeight(100)
        dl.addWidget(self._stego_in)
        self._tdec_btn = QPushButton("🔓 Extract Hidden Message")
        self._tdec_btn.setMinimumHeight(36)
        self._tdec_btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#4ade80;font-size:14px;"
            "font-weight:bold;border:2px solid #4ade80;border-radius:8px;padding:8px;}"
            "QPushButton:hover{background:#4ade80;color:#1a1a2e;}")
        self._tdec_btn.clicked.connect(self._do_tdec)
        dl.addWidget(self._tdec_btn)
        dl.addWidget(QLabel("Decoded Secret Message:"))
        self._decoded = QTextEdit(); self._decoded.setReadOnly(True)
        self._decoded.setPlaceholderText("Hidden message will appear here…")
        dl.addWidget(self._decoded)
        self._tdec_status = QLabel()
        self._tdec_status.setStyleSheet("color:#4ade80;font-size:13px;")
        dl.addWidget(self._tdec_status)
        self._analysis = QLabel()
        self._analysis.setWordWrap(True)
        self._analysis.setStyleSheet("color:#b0b0c0;font-size:12px;")
        dl.addWidget(self._analysis)
        tabs.addTab(dw, "🔓 Decode")

        lay.addWidget(tabs)
        lay.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _heading(text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#00d2ff;font-size:24px;font-weight:bold;padding:5px 0;")
        return lb

    def _do_tenc(self):
        cover, secret = self._cover.toPlainText(), self._secret.toPlainText()
        if not cover:
            QMessageBox.warning(self, "Error", "Enter cover text!"); return
        if not secret:
            QMessageBox.warning(self, "Error", "Enter a secret message!"); return
        try:
            stego = encode_zero_width(cover, secret)
            self._stego_out.setPlainText(stego)
            zw = sum(1 for c in stego if c in (ZW_SPACE, ZW_NON_JOINER))
            self._tenc_status.setText(
                f"✅ Hidden {len(secret)} chars in {len(cover)} cover chars. "
                f"Stego text: {len(stego)} total chars ({zw} invisible).")
            self._tenc_status.setStyleSheet("color:#4ade80;font-size:13px;")
        except Exception as e:
            self._tenc_status.setText(f"❌ {e}")
            self._tenc_status.setStyleSheet("color:#ff6b6b;font-size:13px;")

    def _do_tdec(self):
        stego = self._stego_in.toPlainText()
        if not stego:
            QMessageBox.warning(self, "Error", "Paste stego text first!"); return
        zw = sum(1 for c in stego if c in (ZW_SPACE, ZW_NON_JOINER))
        try:
            msg = decode_zero_width(stego)
            if msg:
                self._decoded.setPlainText(msg)
                self._tdec_status.setText(f"✅ Decoded {len(msg)} characters!")
                self._tdec_status.setStyleSheet("color:#4ade80;font-size:13px;")
            else:
                self._decoded.clear()
                self._tdec_status.setText("❌ No hidden message found.")
                self._tdec_status.setStyleSheet("color:#ff6b6b;font-size:13px;")
        except Exception as e:
            self._tdec_status.setText(f"❌ {e}")
            self._tdec_status.setStyleSheet("color:#ff6b6b;font-size:13px;")
        self._analysis.setText(
            f"📊 Total chars: {len(stego)} | Zero-width: {zw} | "
            f"Visible: {len(stego) - zw} | Potential hidden bits: {zw}")

    def _copy_stego(self):
        t = self._stego_out.toPlainText()
        if t:
            QApplication.clipboard().setText(t)
            QMessageBox.information(self, "Copied", "Stego text copied to clipboard!")


# ════════════════════════════════════════════════════════════
# PAGE 4 — BIT-PLANE VISUALIZER
# ════════════════════════════════════════════════════════════

class BitVisualizerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._src = None
        self._orig = None
        self._mod = None
        self._build()

    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setSpacing(15)
        lay.setContentsMargins(25, 20, 25, 20)

        lay.addWidget(self._heading("🔍 Bit-Plane Visualizer"))
        lay.addWidget(InfoLabel(
            "View individual bit planes of an image to understand how LSB "
            "steganography works. <b>Bit 0 (LSB)</b> is the plane that gets "
            "modified when hiding data — after encoding you'll see patterns "
            "that reveal the hidden message."))

        # Image selection row
        top = QHBoxLayout()
        self._viz_img = ImageDropLabel()
        self._viz_img.imageSelected.connect(self._on_img)
        self._viz_img.setMinimumSize(320, 240)
        top.addWidget(self._viz_img)

        self._orig_lbl = QLabel("Original Preview")
        self._orig_lbl.setAlignment(Qt.AlignCenter)
        self._orig_lbl.setMinimumSize(320, 240)
        self._orig_lbl.setStyleSheet(
            "background:#16213e;border:1px solid #0f3460;"
            "border-radius:10px;color:#555;font-size:13px;")
        top.addWidget(self._orig_lbl)
        lay.addLayout(top)

        # Controls
        ctrl = QFrame()
        ctrl.setStyleSheet(
            "QFrame{background:#16213e;border:1px solid #0f3460;"
            "border-radius:10px;padding:15px;}")
        cl = QHBoxLayout(ctrl)
        cl.addWidget(QLabel("Channel:"))
        self._ch = QComboBox(); self._ch.addItems(["Red", "Green", "Blue"])
        self._ch.currentIndexChanged.connect(self._update_plane)
        cl.addWidget(self._ch)
        cl.addWidget(QLabel("  Bit Plane (0 = LSB, 7 = MSB):"))
        self._bit = QSlider(Qt.Horizontal)
        self._bit.setRange(0, 7); self._bit.setValue(0)
        self._bit.setTickPosition(QSlider.TicksBelow)
        self._bit.setTickInterval(1)
        self._bit.valueChanged.connect(self._update_plane)
        cl.addWidget(self._bit)
        self._bit_lbl = QLabel("Bit 0 (LSB)")
        self._bit_lbl.setStyleSheet("color:#00d2ff;font-weight:bold;min-width:110px;")
        cl.addWidget(self._bit_lbl)
        lay.addWidget(ctrl)

        # Bit-plane + modified + diff display
        disp = QHBoxLayout()
        for group_title, attr in [
            ("Bit Plane (Original)", "_plane_lbl"),
            ("Bit Plane (After LSB Encode)", "_mod_plane_lbl"),
            ("Difference (Changed Pixels)", "_diff_lbl"),
        ]:
            g = QGroupBox(group_title)
            gl = QVBoxLayout(g)
            lb = QLabel("Select an image")
            lb.setAlignment(Qt.AlignCenter)
            lb.setMinimumSize(300, 220)
            lb.setStyleSheet(
                "background:#0f1e3e;border:1px solid #0f3460;"
                "border-radius:8px;color:#555;font-size:12px;")
            gl.addWidget(lb)
            setattr(self, attr, lb)
            disp.addWidget(g)
        lay.addLayout(disp)

        # Test-encode button
        self._test_btn = QPushButton("🧪  Encode Test Message & Compare")
        self._test_btn.setEnabled(False)
        self._test_btn.setMinimumHeight(40)
        self._test_btn.setStyleSheet(
            "QPushButton{background:#0f3460;color:#00d2ff;font-size:14px;"
            "font-weight:bold;border:2px solid #00d2ff;border-radius:8px;padding:8px;}"
            "QPushButton:hover{background:#00d2ff;color:#1a1a2e;}"
            "QPushButton:disabled{color:#555;border-color:#333;}")
        self._test_btn.clicked.connect(self._test_encode)
        lay.addWidget(self._test_btn)

        self._viz_status = QLabel()
        self._viz_status.setWordWrap(True)
        self._viz_status.setStyleSheet("color:#b0b0c0;font-size:12px;")
        lay.addWidget(self._viz_status)
        lay.addStretch()

        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    @staticmethod
    def _heading(text):
        lb = QLabel(text)
        lb.setStyleSheet("color:#00d2ff;font-size:24px;font-weight:bold;padding:5px 0;")
        return lb

    def _on_img(self, path):
        self._src = path
        self._mod = None
        try:
            img = PILImage.open(path).convert('RGB')
            self._orig = np.array(img, dtype=np.uint8)
            px = pil_to_qpixmap(img)
            self._orig_lbl.setPixmap(
                px.scaled(self._orig_lbl.size(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation))
            self._test_btn.setEnabled(True)
            self._update_plane()
            self._viz_status.setText(
                f"✅ Image loaded: {img.size[0]}×{img.size[1]} pixels")
        except Exception as e:
            self._viz_status.setText(f"❌ {e}")

    def _show_on(self, label, arr):
        px = numpy_to_qpixmap(arr)
        label.setPixmap(
            px.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_plane(self):
        if self._orig is None:
            return
        b = self._bit.value()
        self._bit_lbl.setText(
            f"Bit {b} ({'LSB' if b == 0 else 'MSB' if b == 7 else ''})")
        ch = self._ch.currentIndex()

        # Original bit plane
        plane = ((self._orig[:, :, ch] >> b) & 1) * 255
        disp = np.zeros((*plane.shape, 3), dtype=np.uint8)
        disp[:, :, ch] = plane.astype(np.uint8)
        self._show_on(self._plane_lbl, disp)

        # Modified bit plane
        if self._mod is not None:
            mplane = ((self._mod[:, :, ch] >> b) & 1) * 255
            mdisp = np.zeros((*mplane.shape, 3), dtype=np.uint8)
            mdisp[:, :, ch] = mplane.astype(np.uint8)
            self._show_on(self._mod_plane_lbl, mdisp)

            # Difference
            diff = (plane != mplane).astype(np.uint8) * 255
            ddisp = np.zeros((*diff.shape, 3), dtype=np.uint8)
            ddisp[:, :, 0] = diff  # show in red
            self._show_on(self._diff_lbl, ddisp)

    def _test_encode(self):
        if self._orig is None:
            return
        pixels = self._orig.copy()
        msg = "STEGANOGRAPHY_DEMO_" * 20
        binary_msg = text_to_binary(msg) + DELIMITER
        flat = pixels.flatten()
        for i in range(min(len(binary_msg), len(flat))):
            flat[i] = (flat[i] & 0xFE) | int(binary_msg[i])
        self._mod = flat.reshape(pixels.shape)

        changed = int(np.sum(self._orig != self._mod))
        total = self._orig.shape[0] * self._orig.shape[1] * 3
        self._viz_status.setText(
            f"✅ Test message encoded! {changed}/{total} channel values changed "
            f"({changed / total * 100:.2f}%). Switch to Bit 0 to see patterns.")
        self._update_plane()


# ════════════════════════════════════════════════════════════
# MAIN WINDOW
# ════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steganography Learning App")
        self.setMinimumSize(1100, 750)
        self.resize(1350, 870)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Sidebar ──
        sidebar = QFrame()
        sidebar.setFixedWidth(200)
        sidebar.setStyleSheet(
            "QFrame{background:#16213e;border-radius:12px;}")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(8, 15, 8, 15)
        sl.setSpacing(5)

        logo = QLabel("🛡️ StegoLearn")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "color:#00d2ff;font-size:20px;font-weight:bold;"
            "padding:10px;border-bottom:2px solid #0f3460;margin-bottom:10px;")
        sl.addWidget(logo)

        self._nav_btns = []
        for label, idx in [
            ("📚  Learn", 0), ("🖼️  Image Stego", 1),
            ("📝  Text Stego", 2), ("🔍  Bit Visualizer", 3),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(42)
            btn.setStyleSheet(
                "QPushButton{background:transparent;color:#b0b0c0;border:none;"
                "border-radius:8px;text-align:left;padding-left:15px;font-size:14px;}"
                "QPushButton:hover{background:#1a1a4e;color:#e0e0e0;}"
                "QPushButton:checked{background:#0f3460;color:#00d2ff;font-weight:bold;}")
            btn.clicked.connect(lambda _, i=idx: self._switch(i))
            sl.addWidget(btn)
            self._nav_btns.append(btn)

        sl.addStretch()
        ver = QLabel("v1.0")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("color:#555;font-size:11px;")
        sl.addWidget(ver)
        root.addWidget(sidebar)

        # ── Stack ──
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(
            "QStackedWidget{background:#1a1a2e;border-radius:12px;}")
        self._stack.addWidget(LearnPage())
        self._stack.addWidget(ImageStegoPage())
        self._stack.addWidget(TextStegoPage())
        self._stack.addWidget(BitVisualizerPage())
        root.addWidget(self._stack, 1)
                # ── Select first page ──
        self._switch(0)

    # ── Navigation ──
    def _switch(self, idx):
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == idx)


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    if not HAS_PIL:
        print("ERROR: Pillow is required. Install it with:  pip install Pillow")
        sys.exit(1)

    app = QApplication(sys.argv)

    # ── Global font ──
    font = QFont("Segoe UI", 11)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # ── Dark palette as fallback ──
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(26, 26, 46))
    palette.setColor(QPalette.WindowText,      QColor(224, 224, 224))
    palette.setColor(QPalette.Base,            QColor(22, 33, 62))
    palette.setColor(QPalette.AlternateBase,   QColor(26, 26, 46))
    palette.setColor(QPalette.ToolTipBase,     QColor(22, 33, 62))
    palette.setColor(QPalette.ToolTipText,     QColor(224, 224, 224))
    palette.setColor(QPalette.Text,            QColor(224, 224, 224))
    palette.setColor(QPalette.Button,          QColor(15, 52, 96))
    palette.setColor(QPalette.ButtonText,      QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText,      QColor(0, 210, 255))
    palette.setColor(QPalette.Link,            QColor(0, 210, 255))
    palette.setColor(QPalette.Highlight,       QColor(15, 52, 96))
    palette.setColor(QPalette.HighlightedText, QColor(0, 210, 255))
    app.setPalette(palette)

    # ── Stylesheet ──
    app.setStyleSheet(STYLESHEET)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()