"""Physical access to sources: how we reach the bytes, never what is in them.

Keeping storage out of the parsers is what stops a PDF needing three
parsers because it lives in three places. Adding Drive or S3 means
implementing ``list_files``/``read`` and changing nothing downstream.
"""

from app.ingest.loaders.filesystem import FileRef, FilesystemLoader, Loader

__all__ = ["FileRef", "FilesystemLoader", "Loader"]
