"""Concurrency test for the event logger (E1)."""

import threading

from logs.event_logger import EventLogger, validate_entry


def test_concurrent_producers_no_lost_events(tmp_path):
    logger = EventLogger(log_dir=tmp_path, flush_interval_s=0.01)
    n_threads, per = 8, 100

    def worker(wid):
        for i in range(per):
            logger.log("comp", "evt", {"w": wid, "i": i}, latency_ms=float(wid))

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.flush()
    logger.close()

    entries = logger.read_entries()
    assert len(entries) == n_threads * per
    for e in entries:
        validate_entry(e)


def test_concurrent_mixed_with_rotation(tmp_path):
    logger = EventLogger(log_dir=tmp_path, max_bytes=500, flush_interval_s=0.01)

    def worker(wid):
        for i in range(60):
            logger.log("c", "e", {"w": wid, "i": i})

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.flush()
    logger.close()

    total = len(logger.read_entries())
    import gzip

    for a in (tmp_path / "archive").glob("*.ndjson.gz"):
        with gzip.open(a, "rt", encoding="utf-8") as f:
            total += sum(1 for _ in f)
    assert total == 4 * 60