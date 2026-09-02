"""Format-specific parsers. One registry, keyed by the catalog's ``format``."""

from app.ingest.parsers.base import Parser, TabularParser
from app.ingest.parsers.docx_parser import ProposalDocxParser
from app.ingest.parsers.extraction import ExtractionResult, extract_attachment
from app.ingest.parsers.json_parser import BudgetJsonParser
from app.ingest.parsers.pdf_parser import ContractPdfParser
from app.ingest.parsers.txt_parser import TranscriptTxtParser
from app.ingest.parsers.xlsx_parser import RateCardXlsxParser

DEFAULT_PARSERS: dict[str, Parser] = {
    "json": BudgetJsonParser(),
    "txt": TranscriptTxtParser(),
    "docx": ProposalDocxParser(),
    "pdf": ContractPdfParser(),
    "xlsx": RateCardXlsxParser(),
}

__all__ = [
    "BudgetJsonParser",
    "ContractPdfParser",
    "DEFAULT_PARSERS",
    "ExtractionResult",
    "Parser",
    "ProposalDocxParser",
    "RateCardXlsxParser",
    "TabularParser",
    "TranscriptTxtParser",
    "extract_attachment",
]
