
"""Completes the data extraction contract.
Basically shaping the extracted data from the source text using 5 Data Models.

We define exactly what extracted data looks like before writing
any algorithm. Every function we write later builds toward these shapes.

Build order:
  1. ExtractionMethod  — enum of 3 valid extraction states
  2. TableData         — one extracted table + markdown renderer
  3. ExtractorConfig   — all tunable parameters in one place
  4. PageContent       — extracted content of one page
  5. ExtractionResult  — the top-level output everything works with

"""


from __future__ import annotations

import io
import time
from enum import Enum
from dataclasses import dataclass, field
from pydoc import pager
from time import perf_counter
from typing import Any
from pathlib import Path

import pdfplumber
import pytesseract

import logging #a logger to display warnings when Table extraction fails

from pdf2image import convert_from_bytes
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


#MODEL-1 ------------ ExtractionMethod -----------------

#Uses Enum to avoid typos, make invalid strings unrepresentable
#Enum also stores provenance tags (stored on every page) for downstream agents to know how much trust should
#be applied to every piece of text.


class ExtractionMethod(str, Enum):

    NATIVE_TEXT = "native_text" #pdfplumber pulls actual chars: HIGH TRUST

    OCR = "ocr" #tesseract pulls pixels: MEDIUM TRUST

    MIXED = "mixed" #some pages native, some varied : TRUST VARIES



#MODEL 2 ------ TABLE DATA -------------

#table extraction from different pages, also have page counts, headers and table index

#rows -> cleaned strings from extracted data
#raws -> raw data placed as None, good for debugging issues

#to_markdown() because GPT-4 uses md files better compared to list of strings.


@dataclass
class TableData:

    page_number: int # which page the table came from (1-based)

    table_index: int # nth table of the page (0-based)

    headers: list[str]  # first row, cleaned — "Invoice #", "Amount"

    rows: list[list[str]]  # data rows, None cells replaced with ""

    raw: list[list[str | None]]  # original pdfplumber output, untouched


    #convert to markdown for OpenAI, if using Groq then alter if needed

    def to_markdown(self) -> str:
        """
                Render the table as a Markdown string.

                Example output:
                    | Invoice # | Amount | Date   |
                    | ---       | ---    | ---    |
                    | INV-001   | $2,400 | Jan 12 |

                Returns empty string if there are no headers (malformed table).
                """
        if not self.headers:
            return ""

        header_row = "| " + " | ".join(self.headers) + " |"
        separator = "| " + " | ".join(["---"] * len(self.headers)) + " |"

        data_rows = [
            "| " + " | ".join(
                str(cell) if cell is not None else ""
                for cell in row
            ) + " |"
            for row in self.rows
        ]

        return "\n".join([header_row, separator, *data_rows])



#---------MODEL 3: ExtractionConfig --------------

#a configuration file for OCR scanning
#adjusting dpi for speed or accuracy.
#max_pages: during development to avoid waiting on 200-page PDFs
#min_chars_for_native: tune if you see native pages being OCR'd wrongly

#every parameter is tunable.
#Grouping them here means:
#   1. You can override per-environment without touching logic code
#   2. Tests can pass a custom config to control behaviour
#   3. There is one place to look when tuning extraction quality

@dataclass
class ExtractionConfig:
    # ── OCR settings ──────────────────────────────────────────────────────
    # ocr_dpi: resolution to rasterize PDF pages before OCR.
    # 300 is the industry standard minimum for reliable text recognition.
    # Below 200 DPI, small fonts (10-11pt) don't have enough pixels.

    ocr_dpi: int = 300

    # tesseract_lang: language model tesseract uses.
    # "eng" for English. Multi-language: "eng+fra" for English + French.

    tesseract_lang: str = "eng"

    # tesseract_config: flags passed directly to the tesseract binary.
    # --oem 3 → use LSTM neural net engine (most accurate)
    # --psm 6 → assume a single uniform block of text (correct for docs)

    tesseract_config: str = "--oem 3 --psm 6"

    # ── Strategy thresholds ───────────────────────────────────────────────
    # min_chars_for_native: if pdfplumber extracts fewer characters than
    # this from a page, we treat it as scanned and fall back to OCR.
    # 50 chars ≈ roughly one short sentence. A real native PDF page will
    # always exceed this. An image-only page returns ~0.

    min_chars_for_native: int = 50

    # max_pages: stop after this many pages. None = no limit.
    # Useful during development to avoid waiting on 200-page documents.
    max_pages: int  | None = None

    # ── Table extraction ──────────────────────────────────────────────────
    # extract_tables: set False to skip table detection entirely.
    # Table detection adds ~20% overhead — disable if you only need text.
    extract_tables: bool = True

    # table_strategy: pdfplumber line-detection strategy.
    # "lines_strict" = only use explicitly drawn PDF border lines.
    # Avoids false positives on regular paragraph text spacing.
    # We cover this in full detail in Step 4.
    table_strategy: str = "lines_strict"

    # ── Supported formats ─────────────────────────────────────────────────
    # frozenset = immutable, hashable — correct type for a set of constants.
    supported_image_exts: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    )


#====================================================================================
# ------------ MODEL-4 PageContent ----------------------------

#Why do we have per-page content instead of just one big text blob?

#Source citation: - keeps a track of the source of the text of every page.

#OCR confidence doesn't go down much if a single page is blurry and the previous ones are clean

#Page-trust ensures that whether page 2 is native text or mixed. (native -> high trust/ mixed -> low trust)


@dataclass
class PageContent:
    page_number: int #1-based, page from which text was extracted

    text: str #extracted text, whitespace normalized(cleaning inconsistent white spaces)

    method: ExtractionMethod #method through which text was extracted

    # Tables found on this page. Empty list if none found or extract_tables=False.
    tables: list[TableData] = field(default_factory=list)

    # OCR confidence: mean tesseract word-level confidence score, 0–100.

    # None for native-text pages (no OCR was run, so no confidence to report).

    # Score below ~60 = extraction quality is poor, candidate for human review.

    ocr_confidence: float | None = None

    # Auto-calculated — do NOT pass these as constructor arguments
    #they are derived values and are already calculated

    word_count: int = field(default=0, init=False)
    char_count: int = field(default=0, init=False)

    def __post_init__(self):
        """
                Called automatically by Python after __init__ completes.
                Derives word_count and char_count from the text field.

                field(init=False) means these are never accepted as constructor args —
                they are always computed here, never set externally.
                This prevents word_count from ever being out of sync with text.
        """

        self.word_count = len(self.text.split())
        self.char_count = len(self.text)

# =============================================================================
# MODEL 5 — ExtractionResult
# =============================================================================

# The top-level output. This is what POST /ingest returns (via IngestResponse)

# and what agents in Day 3 receive as their document context.
#
# Two computed properties keep callers clean:
#   full_text   — joins all page text for LLM context window consumption
#   all_tables  — flattens tables across all pages into one list
#
# summary() returns a lightweight dict safe for logging — it does NOT

# include full_text or page content, so it never dumps thousands of tokens
# of document content into your log stream.
#
# errors is a list, not a raised exception. A single bad page (e.g. a

# corrupted scan) should not abort extraction of the other 19 pages.

# We collect errors and let the caller decide what to do with them.
# =============================================================================


@dataclass
class ExtractionResult:
    source_filename: str #original filename  e.g. "invoice_jan_2024.pdf"
    source_type: str #pdf or image
    total_pages: int
    pages: list[PageContent]
    method: ExtractionMethod #common method to extract data, MIXED if pages differ

    # Document-level metadata from pypdf:
    # title, author, creator, creation date, page count

    metadata: dict[str, Any] = field(default_factory=dict)

    # Extraction wall-clock time in seconds.
    elapsed_seconds: float = 0.0

    # Non-fatal errors encountered during extraction.
    # A single bad page does not abort the whole document — its error
    # is appended here and extraction continues with the remaining pages.

    errors: list[str] = field(default_factory=list)


#===========================================
#_-----------------------Computed Properties----------------------

#@property lets you access a method without using () [without calling it]

    @property
    def full_text(self) -> str:
        """
                All page text concatenated with double newlines between pages.

                This is what gets passed as document context in LLM prompts.
                Double newline is intentional — it signals a section break to
                the LLM, which helps it understand page boundaries.

                Only includes pages that have actual content (skips blank pages).
        """

        return "\n\n".join(
            p.text for p in self.pages
            if p.text.strip()
        )

    @property
    def all_tables(self) -> list[TableData]:
        """
                All tables across all pages, flattened into one list.
                Each TableData carries its page_number so source is never lost.

                Usage:
                    for table in result.all_tables:
                        print(f"Page {table.page_number}: {table.to_markdown()}")
        """

        return [
            table
            for page in self.pages
            for table in page.tables
        ]


    def summary(self) -> dict[str, Any]:
        """
                Lightweight summary dict — safe for logging and monitoring.

                Does NOT include full_text, page text, or table content.
                Safe to write to application logs, MLflow, or W&B without
                accidentally dumping the entire document into your log stream.
                """
        return {
            "source": self.source_filename,
            "source_type": self.source_type,
            "pages": self.total_pages,
            "method": self.method.value,
            "total_words": sum(p.word_count for p in self.pages),
            "total_chars": sum(p.char_count for p in self.pages),
            "total_tables": len(self.all_tables),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "has_errors": len(self.errors) > 0,
            "errors": self.errors,
        }

# =============================================================================
# STEP 3 — Helper functions
# =============================================================================
# These are module-level functions, not methods on a class.
# Reason: they do one pure thing each — clean text, run OCR.
# Pure functions are easier to test in isolation and easier to reason about.
# DocExtractor will call these in Step 5.
# =============================================================================


def _clean_text(raw: str) -> str:



    """
    Normalise raw text from pdfplumber or Tesseract before storing.

    Why this is necessary:
      - pdfplumber sometimes emits \x00 null bytes from malformed PDFs.
        Null bytes silently corrupt JSON serialisation and LLM prompts.
      - Tesseract on Windows PDFs produces \r\n line endings.
        Double newlines (\r\n → \n\n) create spurious blank lines in prompts.
      - \r carriage returns without \n (old Mac format) need normalising too.

    This is intentionally minimal — we only fix what causes downstream bugs.
    We do NOT strip whitespace aggressively because leading spaces in code
    blocks or indented tables carry meaning.

    Args:
        raw: raw string from pdfplumber.extract_text() or pytesseract

    Returns:
        Cleaned string safe for JSON, logging, and LLM consumption
    """

    return (
        raw
        .replace("\r\n", "\n") #remove old Windows line endings -> Unix
        .replace("\r", "\n") #remove old Mac line endings -> Unix
        .replace("\x00", "") #remove null bytes
    )


from PIL import Image
def _ocr_image(img: Image.Image, cfg: "ExtractionConfig") -> tuple[str, float]:
    """
        Run Tesseract OCR on a PIL Image and return text + mean confidence.

        Why image_to_data() instead of image_to_string():
            image_to_string() gives you the text only.
            image_to_data() gives you the text AND per-word confidence scores.
            We need confidence scores to be stored as PageContent.ocr_confidence
            so downstream agents can decide how much to trust this page's text.
            One extra call, but it unlocks the entire quality-signalling system.

        Why we filter conf >= 0:
            Tesseract returns -1 for non-text regions (whitespace, separators).
            Including -1 values in the average would drag the score down
            and misrepresent the actual recognition quality on real words.

        Why str(c).isdigit() check:
            Tesseract occasionally returns the string "nan" or empty strings
            in the conf list on malformed inputs. The isdigit() guard prevents
            a ValueError when casting to int.

        Args:
            img: PIL Image in RGB or L (grayscale) mode — NEVER RGBA or palette.
                 Caller is responsible for converting before calling this function.
            cfg: ExtractorConfig carrying tesseract_lang and tesseract_config

        Returns:
            (text, mean_confidence) where:
                text            — cleaned OCR string
                mean_confidence — float 0–100, average word-level confidence
                                  0.0 if no valid confidence scores were returned
        """
    # image_to_data returns a dict with these keys:
    #   level, page_num, block_num, par_num, line_num, word_num,
    #   left, top, width, height, conf, text
    # We care about "conf" and "text".

    data = pytesseract.image_to_data(
        img,
        lang=cfg.tesseract_lang,
        config=cfg.tesseract_config,
        output_type=pytesseract.Output.DICT
    )

    # Extract valid word-level confidence scores
    #based on no. of words extracted

    # -1 = non-text region (whitespace, decorators) — exclude from average
    confidences: list[int] =[
        int(c)
        for c in data['conf']
        if str(c).isdigit() and int(c) >= 0
    ]

    # Mean confidence across all recognised words on this image
    # Returns 0.0 if no words were found (blank page or total OCR failure)

    mean_confidence: float = (
        sum(confidences)/ len(confidences)
        if confidences
        else 0.0
    )

    # Get the full text string — reuse the same Tesseract call result
    # via image_to_string for clean paragraph formatting

    raw_text = pytesseract.image_to_string(
        img,
        lang=cfg.tesseract_lang,
        config=cfg.tesseract_config
    )

    return _clean_text(raw_text), mean_confidence




# =============================================================================
# STEP 4 — Table extraction helper
# =============================================================================

def extract_table_from_page(
        plumber_page: Any,
        page_number: int,
        cfg: "ExtractionConfig",
        errors: list[str],

) -> list["TableData"]:
    """
        Extract all tables from a single pdfplumber page object.

        Why lines_strict:
            Only trusts explicitly drawn PDF border lines — no guessing from
            text spacing. This eliminates false positives on paragraph text
            while correctly capturing every bordered table in ops documents
            (invoices, reports, contracts always have drawn borders).

        Why we normalise headers:
            pdfplumber returns None for empty cells. An empty header is
            unusable downstream — LLMs and agents need a name for every column.
            We synthesise "Col0", "Col1" as fallbacks so downstream code
            never receives None or "" as a column identifier.

        Why errors is a parameter (not raised):
            A single malformed table should not abort extraction of the rest
            of the page. We log, append to the shared errors list, and return
            whatever tables we successfully extracted before the failure.

        Args:
            plumber_page:  a pdfplumber Page object (duck-typed as Any to avoid
                           importing pdfplumber at the module level just for typing)
            page_number:   1-based page number, stored on each TableData for citation
            cfg:           ExtractorConfig — we read extract_tables and table_strategy
            errors:        shared error list from ExtractionResult — appended to in place

        Returns:
            List of TableData objects. Empty list if:
              - cfg.extract_tables is False
              - no tables found on this page
              - pdfplumber raised an exception
        """

    if not cfg.extract_tables:
        return []

    raw_tables: list[list[list[str | None]]] = []

    try:
        raw_tables = plumber_page.extract_tables(
            table_settings = {
                #both axes use the same strategy
                # "lines_strict": only explicit PDF drawing commands count.
                "horizontal_strategy": cfg.table_strategy,
                "vertical_strategy": cfg.table_strategy
            }

        ) or [] #extract_tables can return None on pages with no tables at all.

    except Exception as exc:
        #use logger to display a warning about failed table extraction, not an error

        logger.warning(f"Table extraction failed on Page Number %d: %s", page_number, exc)
        errors.append(f"Page Number {page_number} table extraction failed: {exc}")
        return []

    #if nothing is found so far return an empty TableData lists objects

    tables: list[TableData] = []

    for table_index, raw in enumerate(raw_tables):

        #skip if table is empty, or contains None values
        # -- outer any --  checks if there is any row with real data
        # -- inner any -- checks if there even a single word or char in that cell
        if not raw or not any(any(cell for cell in row) for row in raw):
            continue #skip the table and continue

        # ── Header normalisation ──────────────────────────────────────────
        # Rule: first row is always treated as the header row.
        # None or blank cells get a synthesised fallback name (Col0, Col1…)
        # str() cast handles the rare case where pdfplumber returns an int.

        header_row = raw[0] if raw else []
        headers: list[str] = [
            str(cell).strip() if raw and str(cell).strip() else f"col{i}"
            for i, cell in enumerate(header_row)
        ]

        # ── Data row normalisation ────────────────────────────────────────
        # Rule: None → empty string. Every cell cast to str and stripped.
        # We never want None values in rows — they break string operations
        # in prompt builders and serialisers downstream.

        rows: list[list[str]] = [
            [
                str(cell).strip() if cell is not None else ""
                for cell in row
            ]
            for row in raw[1:] #skip header row
        ]

        tables.append(
            TableData(
                page_number=page_number,
                table_index=table_index,
                headers=headers,
                rows=rows,
                raw=raw #keep original for debugging
            )

        )

        #a developer breadcrumb appended to logger to show what was extracted from each page
        logger.debug(
            "Page %d table %d: %d cols x %d rows",
            page_number, table_index, len(headers), len(rows)
        )

    return tables


# =============================================================================
# STEP 5 — DocExtractor class
# =============================================================================
#This is a public orchestrator which allows users to interact with the API interface and upload files
#using UploadFile of FastAPI.
# Everything built in Steps 2-4 is wired together here.
#
# Design principles:
#   1. Two public methods (extract_file --- (development path for local testing and debugging),
#   (extract_bytes --- production path for live API interaction) — one for disk paths,
#      one for raw bytes. Callers don't care which internal path runs.
#   2. All private methods prefixed with _ — callers only see the public API.
#   3. Errors are collected, never raised mid-extraction. A bad page does
#      not abort the document.
#   4. elapsed_seconds measured with perf_counter — wall clock time,
#      used by MLOps monitoring in Day 5.
# =============================================================================

class DocExtractor:
    """
    Multimodal document extractor

    Supports 4 kinds of documents to extract:-

        Native-text PDFs --> pdfplumber (text + tables)
        Scanned PDFs --> pdf2image + pytesseract(OCR fallback per page when chars < 50)

        Images            → pytesseract OCR
        Mixed PDFs        → per-page auto-detection

    #usage: -

        extractor = DocExtractor()

        # From a file path
        result = extractor.extract_files("invoice.pdf")

        #from raw bytes, UploadFile from FastAPI
        result = extractor.extract_bytes(raw_bytes, "invoice.pdf")


        #access results
        print(result.full_text)
        for table in result.all_tables:
            print(table.to_markdown())
        print(result.summary())
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        """
               Initialise with an optional config.
               Defaults to ExtractorConfig() if not provided.
               Also verifies the tesseract binary is reachable at startup —
               better to fail now than silently during extraction.
        """

        self.cfg = config or ExtractionConfig()
        self._check_tesseract()


    # =========================================================================
    # PUBLIC API
    # ========================================================================

    def extract_file(self, path: str | Path) -> ExtractionResult:
        """
                Extract content from a file on disk.

                Args:
                    path: absolute or relative path to a PDF or image file

                Returns:
                    ExtractionResult

                Raises:
                    FileNotFoundError: if the path does not exist
                    ValueError: if the file extension is not supported
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = path.read_bytes()
        return self.extract_bytes(raw, filename=path.name)


    def extract_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        """
                Extract content from raw bytes.
                Used by POST /ingest — FastAPI gives us bytes directly from UploadFile,
                no temp file needed.

                Args:
                    data:     raw file bytes
                    filename: original filename — used ONLY to determine file type
                              from extension. Does not need to exist on disk.

                Returns:
                    ExtractionResult

                Raises:
                    ValueError: if the file extension is not supported
        """

        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf_bytes(data, filename)
        if suffix in self.cfg.supported_image_exts:
            img = Image.open(io.BytesIO(data))
            return self._extract_pil_image(img, filename)
        else:
            raise ValueError(f"Unsupported file type: {suffix}"
                            f"Supported file types: .pdf, {(", ".join(sorted(self.cfg.supported_image_exts)))}")

    # =========================================================================
    # PRIVATE — PDF PATH
    # =========================================================================

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        """Read PDF from disk → bytes → delegate to _extract_pdf_bytes."""

        with open(path, "rb") as f:
            data = f.read()
        return self._extract_pdf_bytes(data, path.name)

    def _extract_pdf_bytes(self, data: bytes, filename: str) -> ExtractionResult:

        """
                Full PDF extraction pipeline:
                  1. Read document metadata via pypdf
                  2. Open with pdfplumber, iterate pages
                  3. Per page: try native text, fall back to OCR if sparse
                  4. Determine overall ExtractionMethod
                  5. Return assembled ExtractionResult

                Why pypdf for metadata and pdfplumber for text?
                    pypdf exposes the XMP/DocInfo metadata dict cleanly.
                    pdfplumber exposes character positions needed for tables.
                    Each library does one thing well — we use both.
        """

        start = time.perf_counter()
        errors: list[str] = []
        pages: list[PageContent] = []
        metadata: dict[str, Any] = {}


            #----- STEP 1: READ DOCUMENT METADATA ---------

            #threshold to prevent abortion of entire document if metadata extraction from
            # a single page fails
        try:
            reader = PdfReader(io.BytesIO(data))
            raw_metadata = reader.metadata or {}
            metadata = {
                "title": raw_metadata.get("/Title", ""),
                "author": raw_metadata.get("/Author", ""),
                "creator": raw_metadata.get("/Creator", ""),
                "producer": raw_metadata.get("/Producer", ""),
                "pages": len(reader.pages)
            }
        except Exception as e:
            errors.append(f"Could not read or extract metadata: {e}")
            logger.warning("Metadata extraction failed", filename, e)


        #------------------STEP 2: EXTRACT TEXT AND TABLES PER PAGE---------------------

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                total_pages = len(pdf.pages)
                limit = self.cfg.max_pages or total_pages #maximum pages per session





                for plumber_page in pdf.pages[:limit]:
                    page_num = plumber_page.page_number  # pdfplumber is 1-based
                    page = self._process_page(plumber_page, page_num, data, errors)
                    pages.append(page)




        except Exception as e:
            errors.append(f"pdfplumber failed to extract : {e}")
            logger.error(f"failed to extract pages via pdfplumber on %s: %s", e, filename, exc_info=True)

        #---------------------------STEP 3: DETERMINE OVERALL EXTRACTION METHOD ----------------------

        ########## --------- if pages used multiple methods ---------- #############

        methods_used = {p.method for p in pages}
        if len(methods_used) > 1:
            overall_method = ExtractionMethod.MIXED
        elif ExtractionMethod.OCR in methods_used:
            overall_method = ExtractionMethod.OCR
        else:
            overall_method = ExtractionMethod.NATIVE_TEXT

        return ExtractionResult(
            source_filename=filename,
            source_type="pdf",
            total_pages=len(pages),
            pages=pages,
            method=overall_method,
            metadata=metadata,
            elapsed_seconds=time.perf_counter() - start,
            errors=errors,
        )

    def _process_page(self,
          plumber_page:Any,
          page_number:int,
          raw_pdf_bytes:bytes,
          errors:list[str]
          ) -> PageContent:

        """
                Decide the extraction strategy for a single PDF page.

                Decision logic:
                    1. Try pdfplumber native text extraction
                    2. Count characters in result
                    3. If chars >= min_chars_for_native → use native text + extract tables
                    4. If chars < min_chars_for_native  → rasterize page → run OCR

                This runs independently for every page, so a document can have
                NATIVE_TEXT pages and OCR pages in the same file.

                Returns:
                    (method, text, ocr_confidence, tables)
                    ocr_confidence is None for native text pages.
        """

        native_text = ""
        tables: list[TableData] = []
        conf: float | None = None

        #-------------Attempt native text extraction -----------------

        try:
            raw = plumber_page.extract_text() or ""
            text = _clean_text(raw)

        except Exception as exc:
            errors.append(f"Page {page_number} native text failed: {exc}")
            logger.warning("Native text failed on page %d: %s", page_number, exc)

            text = ""
            # ── Enough text? Use native path ──────────────────────────────────
        if len(text) >= self.cfg.min_chars_for_native:
            tables = extract_table_from_page(
                plumber_page=plumber_page,
                page_number=page_number,
                cfg=self.cfg,
                errors=errors,
            )
            return ExtractionMethod.NATIVE_TEXT, text, None, tables

        # ── Sparse text → OCR fallback ────────────────────────────────────
        logger.debug(
            "Page %d: only %d chars from native extraction, falling back to OCR",
            page_number, len(text),
        )

        ocr_text = native_text  # keep whatever we got as fallback
        # Native text was sparse — this page is likely a scan.
        # Rasterize just this page at ocr_dpi, then run Tesseract.

        try:
            #rasterize the page at a certain DPI
            images = convert_from_bytes(
                raw_pdf_bytes,
                dpi=self.cfg.ocr_dpi,
                first_page=page_number,
                last_page=page_number
            )

            if images:
                ocr_text, confidence = _ocr_image(images[0], self.cfg)



        except Exception as e:
            errors.append(f"Page {page_number} OCR failed: {e}")
            logger.warning("OCR failed on page %d: %s", page_number, e)

            # Both native and OCR failed — return whatever native gave us
        return PageContent(
            page_number=page_number,
            text=ocr_text,
            method=ExtractionMethod.OCR,
            tables=[],  # no table extraction on OCR pages
            ocr_confidence=conf,
        )


        # =========================================================================
        # PRIVATE — IMAGE PATH
        # =========================================================================


    def _extract_image(self, path: Path):
            """Open image from disk path → delegate to _extract_pil_image."""

            img = Image.open(path)
            return self._extract_pil_image(img, path.name)

    def _extract_pil_image(self, img: Image.Image, filename: str) -> ExtractionResult:
        """
                OCR a PIL image.

                Why we convert image mode before OCR:
                    Tesseract only handles RGB and L (grayscale).
                    PNG uploads often have RGBA (alpha channel).
                    Palette images (mode P) use a colour lookup table.
                    Pillow's convert("RGB") normalises all of these safely.
                """
        start = time.perf_counter()
        errors: list[str] = []

        #normalize image
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        text = ""
        confidence: float | None = None

        try:
            text, confidence = _ocr_image(img, self.cfg)

        except Exception as e:
            errors.append(f"OCR failed: {e}")
            logger.error("OCR failed for %s: %s", filename, e, exc_info=True)


        page = PageContent(
            page_number=1,
            text=text,
            method=ExtractionMethod.OCR,
            ocr_confidence=confidence
        )

        return ExtractionResult(
            source_filename=filename,
            source_type="image",
            total_pages=1,
            pages=[page],
            method=ExtractionMethod.OCR,
            metadata= {
                "width": img.width,
                "height": img.height,
                "mode": img.mode
            },
            elapsed_seconds=time.perf_counter() - start,
            errors=errors


        )

    @staticmethod
    def _check_tesseract() -> None:
        """
        Verify the tesseract binary is accessible at class initialisation.

        Why at __init__ and not at OCR time?
            Fail early — if tesseract is missing, we want a clear error
            when the server starts, not a cryptic failure on the first
            document upload hours later in production.

        Only logs a warning (does not raise) so the server can still
        start and serve non-OCR requests even without tesseract.
        """

        try:
            pytesseract.get_tesseract_version()

        except  pytesseract.TesseractNotFoundError: logger.warning(
                "Tesseract binary not found. OCR will fail at runtime. "
                "Install with: sudo apt-get install -y tesseract-ocr  "
                "or: brew install tesseract"
            )
























