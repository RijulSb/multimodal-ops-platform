#======================================================================================
#--------------------------------------------------------------------------------------
#-------------------------RUN TESTS FOR doc_extractor.py-------------------------------
#======================================================================================


#handle edge cases for doc_extractor.py
#check and handle default errors (do not raise them)



"""
tests/unit/test_doc_extractor.py
=================================
Unit tests for doc_extractor.py — Day 2, Step 6.

Coverage targets
----------------
_clean_text               — edge cases (null bytes, CR/LF variants, empty string)
_ocr_image                — confidence scoring, blank-page / zero-confidence path
extract_table_from_page   — None normalisation, extract_tables=False guard,
                            exception handling, multiple tables, empty tables
DocExtractor              — image path (PNG), PDF native path, OCR fallback,
                            MIXED method detection, unsupported extension

Testing approach
----------------
* All Tesseract / pdfplumber / pdf2image calls are MOCKED.
  We test orchestration logic, not third-party libraries.
* Fixtures build minimal in-memory objects so tests run without any files on disk.
* Each test has a single, explicit assertion surface.
"""


from __future__ import annotations
import types
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import io
from PIL import Image


#===================================================================================
#------------------ IMPORT MODULES UNDER TEST --------------------------------------


from tools.doc_extractor import (
    ExtractionConfig,
    ExtractionMethod,
    PageContent,
    TableData,
    _clean_text,
    _ocr_image,
    extract_table_from_page,
    DocExtractor, ExtractionResult,
)


#=====================================================================
#-------------------------FIXTURES -----------------------------------
#---------------------------------------------------------------------


#pytest.fixtures is used to retain context for modules for testing backgrounds


@pytest.fixture()
def _default_config() -> ExtractionConfig:
    """Standard ExtractionConfig with extract_tables enabled"""
    return ExtractionConfig()


@pytest.fixture()
def no_table_cfg() -> ExtractionConfig:
    """Config with table extraction disabled."""
    return ExtractionConfig(extract_tables=False)


@pytest.fixture()
def rgb_image() -> Image.Image:
    """A tiny 10×10 white RGB image — valid Tesseract input."""
    return Image.new("RGB", (10, 10), color=(255, 255, 255))


@pytest.fixture()
def rgba_image() -> Image.Image:
    """A tiny 10×10 RGBA image — needs mode conversion before OCR."""
    return Image.new("RGBA", (10, 10), color=(255, 255, 255, 128))


@pytest.fixture()
def extractor() -> DocExtractor:
    """DocExtractor with tesseract check patched out."""
    with patch.object(DocExtractor, "_check_tesseract"):
        return DocExtractor(ExtractionConfig())


@pytest.fixture()
def minimal_pdf() -> bytes:
    """Smallest byte sequence pdfplumber can open via mock."""
    return b"%PDF-1.4\n%%EOF"


# =============================================================================
# 1  _clean_text
# =============================================================================

class TestCleanText:

    def test_windows_crlf_normalised(self):
        assert _clean_text("a\r\nb") == "a\nb"

    def test_old_mac_cr_normalised(self):
        assert _clean_text("a\rb") == "a\nb"

    def test_null_bytes_removed(self):
        assert _clean_text("hel\x00lo") == "hello"

    def test_all_three_combined(self):
        assert _clean_text("x\r\ny\rz\x00") == "x\ny\nz"

    def test_empty_string_returns_empty(self):
        assert _clean_text("") == ""

    def test_clean_string_is_unchanged(self):
        s = "Hello World\nLine two."
        assert _clean_text(s) == s

    def test_only_null_bytes_returns_empty(self):
        assert _clean_text("\x00\x00") == ""

    def test_multiple_crlf_pairs(self):
        assert _clean_text("a\r\nb\r\nc") == "a\nb\nc"

    def test_leading_whitespace_preserved(self):
        """_clean_text must NOT strip — indentation in code/tables matters."""
        s = "    indented\n"
        assert _clean_text(s) == s

    def test_unicode_content_preserved(self):
        s = "Ödland über Nürnberg\n"
        assert _clean_text(s) == s

    def test_crlf_does_not_become_double_newline(self):
        """\\r\\n must collapse to single \\n, not \\n\\n."""
        assert _clean_text("a\r\nb") == "a\nb"
        assert "\n\n" not in _clean_text("a\r\nb")


# =============================================================================
# 2  _ocr_image
# =============================================================================


class TestOcrImage:
    @patch("doc_extractor.tesseract.image_to_string")
    @patch("doc_extractor.tesseract.image_to_data")
    def test_mean_confidence_correct(self, mock_str, mock_data, rgb_image, cfg):
        mock_data.return_value = {"conf": ["80", "60", "100"], "text": ["a", "b", "c"]}
        mock_str.return_value = "a b c"
        _, conf = _ocr_image(rgb_image, cfg)
        assert conf == pytest.approx(80.0)

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_negative_one_excluded_from_mean(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["-1", "-1", "90"], "text": ["", "", "word"]}
        mock_str.return_value = "word"
        _, conf = _ocr_image(rgb_image, cfg)
        assert conf == pytest.approx(90.0)

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_blank_page_returns_zero_confidence(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["-1", "-1"], "text": ["", ""]}
        mock_str.return_value = ""
        text, conf = _ocr_image(rgb_image, cfg)
        assert conf == 0.0
        assert text == ""

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_nan_entries_excluded(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["nan", "70"], "text": ["?", "word"]}
        mock_str.return_value = "word"
        _, conf = _ocr_image(rgb_image, cfg)
        assert conf == pytest.approx(70.0)

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_ocr_text_passes_through_clean_text(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["90"], "text": ["word"]}
        mock_str.return_value = "hel\x00lo\r\nworld"
        text, _ = _ocr_image(rgb_image, cfg)
        assert "\x00" not in text
        assert "\r" not in text

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_all_zero_confidence_returns_zero(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["0", "0"], "text": ["a", "b"]}
        mock_str.return_value = "a b"
        _, conf = _ocr_image(rgb_image, cfg)
        assert conf == 0.0

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_single_word_confidence(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["55"], "text": ["hello"]}
        mock_str.return_value = "hello"
        _, conf = _ocr_image(rgb_image, cfg)
        assert conf == pytest.approx(55.0)

    @patch("doc_extractor.pytesseract.image_to_string")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_returns_tuple_of_str_and_float(self, mock_data, mock_str, rgb_image, cfg):
        mock_data.return_value = {"conf": ["80"], "text": ["x"]}
        mock_str.return_value = "x"
        result = _ocr_image(rgb_image, cfg)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)

# =============================================================================
# 3  extract_table_from_page
# =============================================================================

class TestExtractTableFromPage:

    @staticmethod
    def _page(tables):
        p = MagicMock()
        p.extract_tables.return_value = tables
        return p

    # ── guard ──────────────────────────────────────────────────────────────

    def test_returns_empty_when_extract_tables_false(self, no_table_cfg):
        page = self._page([[["H"], ["v"]]])
        assert extract_table_from_page(page, 1, no_table_cfg, []) == []

    def test_extract_tables_not_called_when_disabled(self, no_table_cfg):
        page = self._page(None)
        extract_table_from_page(page, 1, no_table_cfg, [])
        page.extract_tables.assert_not_called()

        # ── None normalisation ─────────────────────────────────────────────────

    def test_none_data_cell_becomes_empty_string(self, cfg):
        raw = [["Name", "Value"], [None, "42"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].rows[0][0] == ""

    def test_none_header_cell_gets_fallback_name(self, cfg):
        raw = [[None, "Amount"], ["INV-001", "100"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].headers[0] == "col0"

    def test_blank_string_header_gets_fallback_name(self, cfg):
        raw = [["", "B"], ["1", "2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].headers[0] == "col0"

        # ── empty / all-None tables skipped ───────────────────────────────────

    def test_completely_empty_table_skipped(self, cfg):
        assert extract_table_from_page(self._page([[]]), 1, cfg, []) == []

    def test_all_none_cells_table_skipped(self, cfg):
        raw = [[None, None], [None, None]]
        assert extract_table_from_page(self._page([raw]), 1, cfg, []) == []

    def test_extract_tables_returning_none_treated_as_empty(self, cfg):
        page = MagicMock()
        page.extract_tables.return_value = None
        errors = []
        assert extract_table_from_page(page, 1, cfg, errors) == []
        assert errors == []

        # ── exception handling ─────────────────────────────────────────────────

    def test_exception_does_not_propagate(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = RuntimeError("corrupt stream")
        errors = []
        result = extract_table_from_page(page, 3, cfg, errors)
        assert result == []

    def test_exception_page_number_in_error_message(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = ValueError("bad data")
        errors = []
        extract_table_from_page(page, 5, cfg, errors)
        assert len(errors) == 1
        assert "5" in errors[0]

    def test_exception_original_message_in_error(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = ValueError("bad data")
        errors = []
        extract_table_from_page(page, 1, cfg, errors)
        assert "bad data" in errors[0]

    # ── multiple tables ────────────────────────────────────────────────────

    def test_multiple_tables_all_returned(self, cfg):
        t1 = [["H1"], ["R1"]]
        t2 = [["A", "B"], ["1", "2"]]
        tables = extract_table_from_page(self._page([t1, t2]), 1, cfg, [])
        assert len(tables) == 2

    def test_table_index_is_sequential(self, cfg):
        t1 = [["H1"], ["R1"]]
        t2 = [["H2"], ["R2"]]
        tables = extract_table_from_page(self._page([t1, t2]), 1, cfg, [])
        assert tables[0].table_index == 0
        assert tables[1].table_index == 1

    def test_page_number_stored_on_each_table(self, cfg):
        raw = [["Col"], ["val"]]
        tables = extract_table_from_page(self._page([raw]), 9, cfg, [])
        assert tables[0].page_number == 9

    def test_raw_data_preserved_untouched(self, cfg):
        raw = [["H", None], ["v", "2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].raw == raw

    def test_data_rows_exclude_header_row(self, cfg):
        raw = [["Header"], ["row1"], ["row2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert len(tables[0].rows) == 2
        assert tables[0].rows[0] == ["row1"]


# =============================================================================
# 4  DocExtractor
# =============================================================================

class TestDocExtractor:

    # ── construction ───────────────────────────────────────────────────────

    def test_init_sets_cfg(self):
        with patch.object(DocExtractor, "_check_tesseract"):
            custom = ExtractionConfig(ocr_dpi=150)
            ex = DocExtractor(custom)
            assert ex.cfg.ocr_dpi == 150

    def test_init_defaults_to_extraction_config(self):
        with patch.object(DocExtractor, "_check_tesseract"):
            ex = DocExtractor()
            assert isinstance(ex.cfg, ExtractionConfig)

    def test_init_calls_check_tesseract(self):
        with patch.object(DocExtractor, "_check_tesseract") as mock_check:
            DocExtractor()
            mock_check.assert_called_once()

        # ── unsupported extension ──────────────────────────────────────────────

    def test_unsupported_extension_raises_value_error(self, extractor):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extractor.extract_bytes(b"data", "file.docx")

    def test_unsupported_extension_error_names_the_suffix(self, extractor):
        with pytest.raises(ValueError, match=r"\.xyz"):
            extractor.extract_bytes(b"data", "file.xyz")

    def test_unsupported_extension_case_insensitive(self, extractor):
        """extract_bytes lowercases the suffix before checking."""
        with pytest.raises(ValueError):
            extractor.extract_bytes(b"data", "file.PDF2")

        # ── extract_file raises on missing path ────────────────────────────────

    def test_extract_file_raises_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.extract_file("/tmp/no_such_file_abc123.pdf")

        # ── image path ─────────────────────────────────────────────────────────

    @patch("doc_extractor.pytesseract.image_to_string", return_value="invoice text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_png_routes_to_ocr(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["85"], "text": ["invoice"]}
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "scan.png")
        assert result.method == ExtractionMethod.OCR

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_source_filename_preserved(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "my_scan.png")
        assert result.source_filename == "my_scan.png"

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_source_type_is_string_image(self, mock_data, mock_str, extractor):
        """source_type must be the string 'image', not a PIL object. (B5 fixed)"""
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "scan.png")
        assert result.source_type == "image"
        assert isinstance(result.source_type, str)



# =============================================================================
# 3  extract_table_from_page
# =============================================================================

class TestExtractTableFromPage:

    @staticmethod
    def _page(tables):
        p = MagicMock()
        p.extract_tables.return_value = tables
        return p

        # ── guard ──────────────────────────────────────────────────────────────

    def test_returns_empty_when_extract_tables_false(self, no_table_cfg):
        page = self._page([[["H"], ["v"]]])
        assert extract_table_from_page(page, 1, no_table_cfg, []) == []

    def test_extract_tables_not_called_when_disabled(self, no_table_cfg):
        page = self._page(None)
        extract_table_from_page(page, 1, no_table_cfg, [])
        page.extract_tables.assert_not_called()

    # ── None normalisation ─────────────────────────────────────────────────


    def test_none_data_cell_becomes_empty(self, cfg):
        raw = [["Name", "Value"], [None, 42]]
        tables = extract_table_from_page(self._page(raw), 1, cfg, [])

        assert tables[0].rows[0][0] == ""

    def test_none_header_cell_gets_fallback_name(self, cfg):
        raw = [["Name", "Amount"], ["INV-001", "100"]]
        tables = extract_table_from_page(self._page(raw), 1, cfg, [])

        assert tables[0].headers[0] == "col0"

    def test_blank_string_header_gets_fallback_name(self, cfg):
        raw = [["", "B"], ["1", "2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].headers[0] == "col0"

        # ── empty / all-None tables skipped ───────────────────────────────────

    def test_completely_empty_table_skipped(self, cfg):
        assert extract_table_from_page(self._page([[]]), 1, cfg, []) == []

    def test_all_none_cells_table_skipped(self, cfg):
        raw = [[None, None], [None, None]]
        assert extract_table_from_page(self._page([raw]), 1, cfg, []) == []

    def test_extract_tables_returning_none_treated_as_empty(self, cfg):
        page = MagicMock()
        page.extract_tables.return_value = None
        errors = []
        assert extract_table_from_page(page, 1, cfg, errors) == []
        assert errors == []

        # ── exception handling ─────────────────────────────────────────────────

    def test_exception_does_not_propagate(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = RuntimeError("corrupt stream")
        errors = []
        result = extract_table_from_page(page, 3, cfg, errors)
        assert result == []

    def test_exception_page_number_in_error_message(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = ValueError("bad data")
        errors = []
        extract_table_from_page(page, 5, cfg, errors)
        assert len(errors) == 1
        assert "5" in errors[0]

    def test_exception_original_message_in_error(self, cfg):
        page = MagicMock()
        page.extract_tables.side_effect = ValueError("bad data")
        errors = []
        extract_table_from_page(page, 1, cfg, errors)
        assert "bad data" in errors[0]

        # ── multiple tables ────────────────────────────────────────────────────

    def test_multiple_tables_all_returned(self, cfg):
        t1 = [["H1"], ["R1"]]
        t2 = [["A", "B"], ["1", "2"]]
        tables = extract_table_from_page(self._page([t1, t2]), 1, cfg, [])
        assert len(tables) == 2

    def test_table_index_is_sequential(self, cfg):
        t1 = [["H1"], ["R1"]]
        t2 = [["H2"], ["R2"]]
        tables = extract_table_from_page(self._page([t1, t2]), 1, cfg, [])
        assert tables[0].table_index == 0
        assert tables[1].table_index == 1

    def test_page_number_stored_on_each_table(self, cfg):
        raw = [["Col"], ["val"]]
        tables = extract_table_from_page(self._page([raw]), 9, cfg, [])
        assert tables[0].page_number == 9

    def test_raw_data_preserved_untouched(self, cfg):
        raw = [["H", None], ["v", "2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert tables[0].raw == raw

    def test_data_rows_exclude_header_row(self, cfg):
        raw = [["Header"], ["row1"], ["row2"]]
        tables = extract_table_from_page(self._page([raw]), 1, cfg, [])
        assert len(tables[0].rows) == 2
        assert tables[0].rows[0] == ["row1"]

# =============================================================================
# 4  DocExtractor
# =============================================================================

class TestDocExtractor:

    # ── construction ───────────────────────────────────────────────────────

    def test_init_sets_cfg(self):
        with patch.object(DocExtractor, "_check_tesseract"):
            custom = ExtractionConfig(ocr_dpi=150)
            ex = DocExtractor(custom)
            assert ex.cfg.ocr_dpi == 150

    def test_init_defaults_to_extraction_config(self):
        with patch.object(DocExtractor, "_check_tesseract"):
            ex = DocExtractor()
            assert isinstance(ex.cfg, ExtractionConfig)

    def test_init_calls_check_tesseract(self):
        with patch.object(DocExtractor, "_check_tesseract") as mock_check:
            DocExtractor()
            mock_check.assert_called_once()

    # ── unsupported extension ──────────────────────────────────────────────

    def unsupported_extension_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):

            extractor.extract_bytes(b"data", "file.docx")

    def test_unsupported_extension_error_names_the_suffix(self, extractor):
        with pytest.raises(ValueError, match=r"\.xyz"):
            extractor.extract_bytes(b"data", "file.xyz")

    def test_unsupported_extension_case_insensitive(self, extractor):
        """extract_bytes lowercases the suffix before checking."""
        with pytest.raises(ValueError):
            extractor.extract_bytes(b"data", "file.PDF2")

        # ── extract_file raises on missing path ────────────────────────────────

    def test_extract_file_raises_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.extract_file("/tmp/no_such_file_abc123.pdf")

        # ── image path ─────────────────────────────────────────────────────────

    @patch("doc_extractor.pytesseract.image_to_string", return_value="invoice text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_png_routes_to_ocr(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["85"], "text": ["invoice"]}
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "scan.png")
        assert result.method == ExtractionMethod.OCR

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_source_filename_preserved(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "my_scan.png")
        assert result.source_filename == "my_scan.png"

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_source_type_is_string_image(self, mock_data, mock_str, extractor):
        """source_type must be the string 'image', not a PIL object. (B5 fixed)"""
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "scan.png")
        assert result.source_type == "image"
        assert isinstance(result.source_type, str)

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_total_pages_is_one(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "p.png")
        assert result.total_pages == 1

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_rgba_image_converted_before_ocr(self, mock_data, mock_str, extractor, rgba_image):
        """RGBA images must be converted to RGB; no crash on mode check. (B2 fixed)"""
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        rgba_image.save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "rgba.png")
        assert result.method == ExtractionMethod.OCR

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_image_metadata_contains_dimensions(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (320, 240)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "img.png")
        assert result.metadata["width"] == 320
        assert result.metadata["height"] == 240

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_jpeg_extension_accepted(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["70"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        result = extractor.extract_bytes(buf.getvalue(), "photo.jpg")
        assert result.source_filename == "photo.jpg"

    @patch("doc_extractor.pytesseract.image_to_string", return_value="text")
    @patch("doc_extractor.pytesseract.image_to_data")
    def test_elapsed_seconds_set_for_image(self, mock_data, mock_str, extractor):
        mock_data.return_value = {"conf": ["80"], "text": ["text"]}
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        result = extractor.extract_bytes(buf.getvalue(), "img.png")
        assert result.elapsed_seconds >= 0.0

    # ── PDF path — metadata ────────────────────────────────────────────────

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_pdf_source_type_is_pdf(self, mock_open, mock_reader, extractor, minimal_pdf):
        self._setup_plumber(mock_open, [self._make_page("A" * 200)])
        self._setup_reader(mock_reader, {"/Title": "T"})
        with patch.object(extractor, "_process_page", side_effect=self._native_page):
            result = extractor.extract_bytes(minimal_pdf, "doc.pdf")
        assert result.source_type == "pdf"

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_pdf_metadata_title_extracted(self, mock_open, mock_reader, extractor, minimal_pdf):
        self._setup_plumber(mock_open, [self._make_page("A" * 200)])
        self._setup_reader(mock_reader, {"/Title": "Annual Report", "/Author": "Acme"})
        with patch.object(extractor, "_process_page", side_effect=self._native_page):
            result = extractor.extract_bytes(minimal_pdf, "doc.pdf")
        assert result.metadata.get("title") == "Annual Report"
        assert result.metadata.get("author") == "Acme"

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_metadata_failure_goes_to_errors_not_crash(self, mock_open, mock_reader, extractor, minimal_pdf):
        mock_reader.side_effect = Exception("bad header")
        self._setup_plumber(mock_open, [self._make_page("A" * 200)])
        with patch.object(extractor, "_process_page", side_effect=self._native_page):
            result = extractor.extract_bytes(minimal_pdf, "doc.pdf")
        assert any("metadata" in e.lower() or "Could not" in e for e in result.errors)

    # ── PDF path — native text ─────────────────────────────────────────────

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_native_pdf_method_is_native_text(self, mock_open, mock_reader, extractor, minimal_pdf):
        """
        When _process_page returns a PageContent with NATIVE_TEXT,
        the overall result method must be NATIVE_TEXT.
        We inject PageContent directly to bypass RB1 (tuple return bug).
        """
        self._setup_plumber(mock_open, [self._make_page("A" * 200)])
        self._setup_reader(mock_reader, {})
        with patch.object(extractor, "_process_page", side_effect=self._native_page):
            result = extractor.extract_bytes(minimal_pdf, "native.pdf")
        assert result.method == ExtractionMethod.NATIVE_TEXT

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_native_pdf_page_count(self, mock_open, mock_reader, extractor, minimal_pdf):
        p1 = self._make_page("A" * 200, page_number=1)
        p2 = self._make_page("B" * 200, page_number=2)
        self._setup_plumber(mock_open, [p1, p2])
        self._setup_reader(mock_reader, {})
        pages = [
            PageContent(page_number=1, text="A" * 200, method=ExtractionMethod.NATIVE_TEXT),
            PageContent(page_number=2, text="B" * 200, method=ExtractionMethod.NATIVE_TEXT),
        ]
        with patch.object(extractor, "_process_page", side_effect=pages):
            result = extractor.extract_bytes(minimal_pdf, "two_pages.pdf")
        assert result.total_pages == 2



    #------------ PDF Path -- OCR Fallback --------------------------------------

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_sparse_page_triggers_ocr_(self, mock_open, extractor, mock_reader, minimal_pdf):
        """
                Page with < min_chars_for_native characters must trigger OCR.
                We inject an OCR PageContent to test overall method resolution.
        """

        self._setup_plumber(mock_open, [self._make_page("Hi",page_number=1)])
        self._setup_reader(mock_reader, {})

        ocr_page = PageContent(
            page_number=1, text="scanned text",
            method=ExtractionMethod.OCR, ocr_confidence=72.0
        )

        with patch.object(extractor, "_process_page", return_value=ocr_page):
            result = extractor.extract_bytes(minimal_pdf, "scan.pdf")

        assert result.method == ExtractionMethod.OCR
        assert result.pages[0].method == ExtractionMethod.OCR

    @patch("doc_extractor.convert_from_bytes")
    @patch("doc_extractor.pytesseract.image_to_string", return_value="ocr result")
    @patch("doc_extractor.pytesseract.image_to_data")
    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_ocr_confidence_none_due_to_rb3(
            self, mock_open, mock_reader, mock_data, mock_str, mock_convert, extractor, minimal_pdf
    ):
        """
                RB3: after _ocr_image returns `confidence`, it is stored in the local
                variable `confidence` but PageContent is constructed with `conf`
                (which was initialised to None and never updated).
                Result: ocr_confidence is always None on OCR pages from PDF path.

                This test documents the CURRENT (buggy) behaviour.
                When RB3 is fixed, change the assertion to: assert ... is not None
        """

        mock_data.return_value = {"conf": ["80"], "text": ["word"]}
        sparse_page = self._make_page("Hi", page_number=1)
        self._setup_plumber(mock_open, [sparse_page])
        self._setup_reader(mock_reader, {})
        ocr_img = Image.new("RGB", (100, 100))
        mock_convert.return_value = [ocr_img]

        result = extractor.extract_bytes(minimal_pdf, "scan.pdf")

        # RB3: remove this assertion and replace with `is not None` when fixed
        assert result.pages[0].ocr_confidence is None


    #------------------------------------------------------------------------------
    #====================== MIXED METHOD DETECTION ================================
    #------------------------------------------------------------------------------

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_mixed_method_when_pages_differ(self, mock_open, mock_reader, extractor, minimal_pdf):
        p1 = self._make_page("A" * 200, page_number = 1)
        p2 = self._make_page("Hi", page_number = 2)
        self._setup_plumber(mock_open, [p1, p2])
        self._setup_reader(mock_reader, {})

        native = PageContent(page_number=1, text="A" * 200, method=ExtractionMethod.NATIVE_TEXT)
        ocr = PageContent(page_number=2, text="scanned", method=ExtractionMethod.OCR, ocr_confidence=60.0)

        side_effect = [native, ocr]
        with patch.object(extractor, "_process_page", side_effect=side_effect):

            result = extractor.extract_bytes(minimal_pdf, "mixed.pdf")

        assert result.method == ExtractionMethod.MIXED

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_all_ocr_pages_gives_ocr_method(self, mock_open, mock_reader, extractor, minimal_pdf):
        self._setup_plumber(mock_open, [self._make_page("Hi", page_number=1)])
        self._setup_reader(mock_reader, {})
        ocr = PageContent(page_number=1, text="scan", method=ExtractionMethod.OCR, ocr_confidence=55.0)
        with patch.object(extractor, "_process_page", return_value=ocr):
            result = extractor.extract_bytes(minimal_pdf, "scan.pdf")
        assert result.method == ExtractionMethod.OCR

        # ── errors collected, not raised ──────────────────────────────────────

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_pdfplumber_crash_goes_to_errors(self, mock_open, mock_reader, extractor, minimal_pdf):
        mock_open.side_effect = Exception("Pdfplumber exploded")
        self._setup_reader(mock_reader, {})
        result = extractor.extract_bytes(minimal_pdf, "bad.pdf")
        assert any("pdfplumber" in e.lower() for e in result.errors)

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_pdfplumber_crash_gives_zero_pages(self, mock_open, mock_reader, extractor, minimal_pdf):
        mock_open.side_effect = Exception("gone")
        self._setup_reader(mock_reader, {})
        result = extractor.extract_bytes(minimal_pdf, "bad.pdf")
        assert result.total_pages == 0

        # ── max_pages config respected ────────────────────────────────────────

    @patch("doc_extractor.PdfReader")
    @patch("doc_extractor.pdfplumber.open")
    def test_max_pages_limits_pages_processed(self, mock_open, mock_reader, extractor, minimal_pdf):
        extractor.cfg.max_pages = 1
        pages = [self._make_page("A" * 200, i) for i in range(1, 4)]
        self._setup_plumber(mock_open, pages)
        self._setup_reader(mock_reader, {})
        native = PageContent(page_number=1, text="A" * 200, method=ExtractionMethod.NATIVE_TEXT)
        with patch.object(extractor, "_process_page", return_value=native):
            result = extractor.extract_bytes(minimal_pdf, "long.pdf")
        assert result.total_pages == 1



#===================================================================================
#-------------------- helper functions ---------------------------------------------
#===================================================================================


    @staticmethod
    def _make_page(text: str, page_number: int  = 1):
        p = MagicMock()
        p.page_number=page_number,
        p.extract_tables.return_value = [],
        p.extract_text.return_text = text
        return p

    @staticmethod
    def _setup_plumber(mock_open, pages):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.pages = pages
        mock_open.return_value = ctx

    @staticmethod
    def _setup_reader(mock_reader, meta: dict):
        inst = MagicMock()
        inst.metadata = meta
        inst.pages = [MagicMock()]
        mock_reader.return_value = inst

    @staticmethod
    def _native_page(plumber_page, page_number, raw_pdf_bytes, errors):
        return PageContent(
            page_number=page_number,
            text="A" * 200,
            method=ExtractionMethod.NATIVE_TEXT,
        )


# =============================================================================
# 5  ExtractionResult — computed properties and summary
# =============================================================================

class TestExtractionResult:

    @staticmethod
    def _result(**kwargs):
        defaults = dict(
            source_filename="test.pdf",
            source_type="pdf",
            total_pages=1,
            pages=[],
            method=ExtractionMethod.NATIVE_TEXT,
        )
        defaults.update(kwargs)
        return ExtractionResult(**defaults)

    def test_full_text_joins_with_double_newline(self):
        p1 = PageContent(page_number=1, text="Page one", method=ExtractionMethod.NATIVE_TEXT)
        p2 = PageContent(page_number=2, text="Page two", method=ExtractionMethod.NATIVE_TEXT)
        r = self._result(pages=[p1, p2])
        assert r.full_text == "Page one\n\nPage two"

    def test_full_text_skips_blank_pages(self):
        p1 = PageContent(page_number=1, text="", method=ExtractionMethod.NATIVE_TEXT)
        p2 = PageContent(page_number=2, text="Content", method=ExtractionMethod.NATIVE_TEXT)
        r = self._result(pages=[p1, p2])
        assert r.full_text == "Content"

    def test_all_tables_flattened_across_pages(self):
        t1 = TableData(page_number=1, table_index=0, headers=["H"], rows=[], raw=[])
        t2 = TableData(page_number=2, table_index=0, headers=["X"], rows=[], raw=[])
        p1 = PageContent(page_number=1, text="a", method=ExtractionMethod.NATIVE_TEXT, tables=[t1])
        p2 = PageContent(page_number=2, text="b", method=ExtractionMethod.NATIVE_TEXT, tables=[t2])

        r = self._result(pages=[p1, p2])
        assert len(r.all_tables) == 2
        assert r.all_tables[0].page_number == 1
        assert r.all_tables[1].page_number == 2

    def test_summary_does_not_contain_full_text(self):
        p1 = PageContent(page_number=1, text="secret", method=ExtractionMethod.NATIVE_TEXT)
        r = self._result(pages=[p1])
        s = r.summary()
        assert "full_text" not in s
        assert "secret" not in str(s)

    def test_summary_has_errors_flag_true(self):
        r = self._result(errors=["something failed"])
        assert r.summary()["has_errors"] is True

    def test_summary_has_errors_flag_false_when_no_errors(self):
        r = self._result(errors=[])
        assert r.summary()["has_errors"] is False

    def test_summary_total_words_aggregated(self):
        p1 = PageContent(page_number=1, text="one two three", method=ExtractionMethod.NATIVE_TEXT)
        p2 = PageContent(page_number=2, text="four five", method=ExtractionMethod.NATIVE_TEXT)
        r = self._result(pages=[p1, p2])
        assert r.summary()["total_words"] == 5

    def test_summary_total_tables_aggregated(self):
        t = TableData(page_number=1, table_index=0, headers=["H"], rows=[], raw=[])
        p = PageContent(page_number=1, text="x", method=ExtractionMethod.NATIVE_TEXT, tables=[t])
        r = self._result(pages=[p])
        assert r.summary()["total_tables"] == 1


# =============================================================================
# 6  TableData.to_markdown
# =============================================================================

class TestTableDataToMarkdown:

    def test_header_row_rendered(self):
        t = TableData(
            page_number=1, table_index=0,
            headers=["Name", "Amount"],
            rows=[["Alice", "100"]],
            raw=[]
        )
        md = t.to_markdown()
        assert "| Name | Amount |" in md

    def test_separator_row_rendered(self):
        t = TableData(
            page_number=1, table_index=0,
            headers=["A", "B"],
            rows=[],
            raw=[]
        )
        assert "| --- | --- |" in t.to_markdown()

    def test_data_rows_rendered(self):
        t = TableData(
            page_number=1, table_index=0,
            headers=["A"],
            rows=[["val1"], ["val2"]],
            raw=[]
        )
        md = t.to_markdown()
        assert "| val1 |" in md
        assert "| val2 |" in md

    def test_empty_headers_returns_empty_string(self):
        t = TableData(page_number=1, table_index=0, headers=[], rows=[], raw=[])
        assert t.to_markdown() == ""

    def test_none_cell_in_row_rendered_as_empty(self):
        t = TableData(
            page_number=1, table_index=0,
            headers=["A", "B"],
            rows=[["val", None]],
            raw=[]
        )
        md = t.to_markdown()
        assert "| val |  |" in md


# =============================================================================
# 7  PageContent — auto-computed fields
# =============================================================================

class TestPageContent:

    def test_word_count_computed(self):
        p = PageContent(page_number=1, text="hello world foo", method=ExtractionMethod.NATIVE_TEXT)
        assert p.word_count == 3

    def test_char_count_computed(self):
        p = PageContent(page_number=1, text="hello", method=ExtractionMethod.NATIVE_TEXT)
        assert p.char_count == 5

    def test_empty_text_gives_zero_counts(self):
        p = PageContent(page_number=1, text="", method=ExtractionMethod.NATIVE_TEXT)
        assert p.word_count == 0
        assert p.char_count == 0

    def test_word_count_not_accepted_as_constructor_arg(self):
        """field(init=False) means word_count cannot be passed in."""
        with pytest.raises(TypeError):
            PageContent(
                page_number=1, text="hello",
                method=ExtractionMethod.NATIVE_TEXT,
                word_count=99,  # must raise
            )

    def test_ocr_confidence_defaults_to_none_for_native(self):
        p = PageContent(page_number=1, text="text", method=ExtractionMethod.NATIVE_TEXT)
        assert p.ocr_confidence is None













