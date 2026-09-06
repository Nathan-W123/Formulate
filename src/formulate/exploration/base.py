"""The exploration interface.

Specification section 3: exploration generates a diverse candidate pool, and
"exploration and evaluation remain separate".  An explorer proposes; it never
scores, ranks, or decides.  It may *read* scores from a previous iteration to
steer itself, which is what :meth:`Explorer.propose` receives ``scored`` for.
"""

from __future__ import annotations

import abc
from typing import Sequence

from formulate.core.candidate import Candidate
from formulate.targets.spec import TargetSpec


class Explorer(abc.ABC):
    """Proposes candidates for evaluation."""

    #: Identifier recorded as the candidate's generation strategy.
    id: str = "explorer"
    version: str = "0"

    @abc.abstractmethod
    def propose(
        self,
        spec: TargetSpec,
        count: int,
        *,
        scored: Sequence[Candidate] = (),
        seed: int = 0,
    ) -> list[Candidate]:
        """Return up to ``count`` candidates.

        ``scored`` carries candidates already evaluated this run, so that
        population-based strategies can build on them.  An explorer that
        ignores it is a valid fixed generator, which is the research baseline
        of section 1.
        """

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} id={self.id!r}>"
