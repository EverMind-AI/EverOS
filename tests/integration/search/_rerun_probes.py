"""Re-run probes against an existing corpus + regenerate the report.

Reuses everything from :mod:`_run_full_report` except the ingest step —
points at the already-populated ``~/.everos-report-corpus`` and only
re-runs the search probes + report rendering. Useful when the corpus
is already there from a previous run and you just want to refresh the
retrieval section without paying for LLM ingestion again.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


from _run_full_report import (  # noqa: E402
    CONVERSATION,
    CORPUS_ROOT,
    REPORT_PATH,
    inspect_artifacts,
    render_report,
    run_probes,
)


async def main() -> None:
    if not (CORPUS_ROOT / "users").is_dir():
        raise SystemExit(f"{CORPUS_ROOT} not populated — run _run_full_report.py first")
    os.environ["EVEROS_MEMORY__ROOT"] = str(CORPUS_ROOT)
    from everos.config import load_settings

    load_settings.cache_clear()

    print(f"[1/3] using corpus at {CORPUS_ROOT}")

    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        print("[2/3] inspecting artifacts + running probes ...")
        artifacts = await inspect_artifacts(CORPUS_ROOT)
        probes = await run_probes(client)

    print("[3/3] re-rendering report ...")
    md = render_report(
        memory_root=CORPUS_ROOT,
        ingest_summary={
            "batches": [
                {
                    "idx": i,
                    "msg_count": len(b),
                    "status": "extracted (cached)",
                    "returned_count": len(b),
                }
                for i, b in enumerate(CONVERSATION)
            ],
            "flush_status": "extracted (cached)",
        },
        cascade_summary={
            "note": "cascade was force-completed via _rerun_probes.py "
            "after initial run; counts below are post-completion."
        },
        artifacts=artifacts,
        probes=probes,
    )
    REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"      → {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
