"""Silencing native library output.

The xtb library writes progress banners, topology tables and per-atom listings
straight to file descriptor 1 from its Fortran layer.  ``contextlib.redirect_stdout``
rebinds ``sys.stdout`` inside Python and does not touch that descriptor, so it
has no effect here: a single energy evaluation would otherwise dump a hundred
lines into the middle of a report.

Redirecting the descriptor itself is the only thing that works.  The captured
text is kept and returned, because a backend failure often explains itself in
exactly the output we are suppressing.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from typing import Iterator


@contextlib.contextmanager
def suppress_native_output(capture: bool = True) -> Iterator[dict[str, str]]:
    """Silence writes to stdout and stderr, including from native code.

    Yields a dict that is populated with the captured text on exit, so a caller
    can attach it to a diagnostic when something goes wrong.
    """
    captured: dict[str, str] = {"stdout": "", "stderr": ""}

    # Flush Python-level buffers first, or their contents land in the temp file.
    sys.stdout.flush()
    sys.stderr.flush()

    saved_out, saved_err = os.dup(1), os.dup(2)
    sink = tempfile.TemporaryFile(mode="w+b") if capture else open(os.devnull, "wb")
    try:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            if capture:
                try:
                    sink.seek(0)
                    text = sink.read().decode("utf-8", errors="replace")
                    captured["stdout"] = text
                except Exception:
                    pass
    finally:
        os.close(saved_out)
        os.close(saved_err)
        sink.close()
