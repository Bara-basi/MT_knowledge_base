from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from uuid import uuid4

from app.api.v1.feishu import _answer_feishu_message
from app.core.config import settings
from app.db.postgres import claim_feishu_answer_job, finish_feishu_answer_job
from app.services.privacy import decrypt_chat_text
from app.services.harness import close_local_harnesses


logger = logging.getLogger(__name__)


async def _run_slot(slot: int) -> None:
    owner = f"{socket.gethostname()}:{os.getpid()}:{slot}:{uuid4().hex[:8]}"
    while True:
        job = await asyncio.to_thread(
            claim_feishu_answer_job,
            lease_owner=owner,
            lease_seconds=settings.answer_job_lease_seconds,
        )
        if not job:
            await asyncio.sleep(max(0.1, settings.answer_worker_poll_seconds))
            continue
        job_id = str(job["job_id"])
        logger.info(
            "Claimed Feishu answer job %s (slot=%s, attempt=%s/%s)",
            job_id,
            slot,
            job.get("attempts"),
            job.get("max_attempts"),
        )
        try:
            payload = json.loads(decrypt_chat_text(str(job["payload"])))
            if not isinstance(payload, dict):
                raise ValueError("answer job payload must be a JSON object")
            payload["job_id"] = job_id
            await _answer_feishu_message(**payload)
        except asyncio.CancelledError:
            await asyncio.to_thread(
                finish_feishu_answer_job,
                job_id=job_id,
                success=False,
                error="answer worker stopped while job was running",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - the durable queue owns retries.
            logger.exception("Feishu answer job %s failed", job_id)
            await asyncio.to_thread(
                finish_feishu_answer_job,
                job_id=job_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            await asyncio.to_thread(
                finish_feishu_answer_job,
                job_id=job_id,
                success=True,
            )
            logger.info("Completed Feishu answer job %s", job_id)


async def run_worker() -> None:
    concurrency = max(1, settings.answer_worker_concurrency)
    logger.info("Starting durable answer worker with concurrency=%s", concurrency)
    try:
        await asyncio.gather(*(_run_slot(slot) for slot in range(concurrency)))
    finally:
        await asyncio.to_thread(close_local_harnesses)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
