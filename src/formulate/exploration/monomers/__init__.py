"""Generated repeat-unit libraries, one module per polymerisation chemistry.

Each module exposes ``units() -> list[tuple[str, str]]``: (name, repeat-unit
SMILES) pairs, deduplicated, every SMILES carrying exactly two ``[*]``.  The
name carries the availability of the polymer, because a search that returns a
repeat unit nobody sells has answered a different question than the one asked
and the ranking has to be able to tell the two apart.

Nothing is imported here at package level on purpose: a library is a *list of
strings*, and an explorer should be able to take one without paying for the
others.
"""

from __future__ import annotations
