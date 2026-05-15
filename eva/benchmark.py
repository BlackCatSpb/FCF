"""
Benchmark — end-to-end тест всей системы FCF.

Загружает чекпоинт, прогоняет разнородные запросы через FCFSystem,
сохраняет отчёт с метриками: confidence, SRG+, время, домен, KCA.
"""

import sys, os, json, time, torch
sys.path.insert(0, '.')
from loguru import logger
from typing import Dict, Any


def run_benchmark(checkpoint_path: str = None,
                  output_path: str = "benchmark.json",
                  num_queries: int = 12) -> Dict[str, Any]:
    """
    Полный бенчмарк системы.

    Возвращает отчёт:
      - per_query: метрики по каждому запросу
      - summary: сводка (средний confidence, время, домены)
      - system: состояние системы после теста
    """
    from eva.fcf_system import FCFSystem

    logger.info("=" * 60)
    logger.info("FCF Benchmark")
    logger.info("=" * 60)

    fcf = FCFSystem()

    fcf.bootstrap()

    if checkpoint_path and os.path.exists(checkpoint_path):
        from eva.utils import load_primordial_layer
        from eva.primordial_layer import PrimordialLayer
        fcf.layer = load_primordial_layer(checkpoint_path, PrimordialLayer)
        logger.info(f"Loaded: {checkpoint_path} (snapshots={len(fcf.layer.state_storage)})")

    queries = [
        {"text": "Что такое история?", "domain": "humanities"},
        {"text": "Объясни что изучает математика", "domain": "science"},
        {"text": "Как работают компьютеры?", "domain": "tech"},
        {"text": "Расскажи о природе Земли", "domain": "nature"},
        {"text": "Что такое программирование?", "domain": "tech"},
        {"text": "Какие законы физики самые важные?", "domain": "science"},
        {"text": "Чем отличается человек от животных?", "domain": "biology"},
        {"text": "Что такое философия?", "domain": "humanities"},
        {"text": "Как устроена экономика?", "domain": "social"},
        {"text": "Что делает медицина?", "domain": "health"},
        {"text": "Расскажи о искусственном интеллекте", "domain": "tech"},
        {"text": "В чем смысл жизни?", "domain": "humanities"},
    ]

    per_query = []
    total_time = 0.0
    total_confidence = 0.0
    domains_found = 0
    kca_triggered = 0
    anomalies = 0

    for i, q in enumerate(queries[:num_queries]):
        logger.info(f"[{i + 1}/{num_queries}] {q['text'][:60]}...")
        t0 = time.time()

        try:
            result = fcf.query(q["text"], max_tokens=64)
        except Exception as e:
            result = {"response": str(e), "confidence": 0.0, "error": str(e)}

        elapsed = time.time() - t0
        total_time += elapsed

        confidence = result.get("confidence", 0.0)
        total_confidence += confidence

        has_domain = bool(result.get("domain_id") or
                          len(fcf.hnsw.level0) > 0 if fcf.hnsw else False)
        domains_found += int(has_domain)

        kca_used = result.get("kca_applied", False)
        kca_triggered += int(kca_used)

        entry = {
            "id": i + 1,
            "query": q["text"],
            "domain": q["domain"],
            "response": result.get("response", "")[:200],
            "confidence": round(float(confidence), 3),
            "similarity": round(float(result.get("similarity", 0)), 3),
            "ethics": round(float(result.get("ethics_score", 0)), 3),
            "time_s": round(elapsed, 2),
            "domain_found": has_domain,
            "kca_used": kca_used,
        }
        per_query.append(entry)

        if i == 0:
            logger.info(f"  Response: {result.get('response', '')[:80]}...")
            logger.info(f"  conf={confidence:.3f}, time={elapsed:.2f}s")

    system_state = fcf.stats()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path or "fresh",
        "per_query": per_query,
        "summary": {
            "queries": len(per_query),
            "avg_confidence": round(total_confidence / max(len(per_query), 1), 3),
            "avg_time_s": round(total_time / max(len(per_query), 1), 2),
            "total_time_s": round(total_time, 2),
            "domains_found": f"{domains_found}/{len(per_query)}",
            "kca_triggered": kca_triggered,
            "errors": sum(1 for q in per_query if q["confidence"] == 0.0),
        },
        "system": system_state,
        "queries": queries[:num_queries],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Benchmark saved: {output_path}")
    logger.info(
        f"Summary: avg_conf={report['summary']['avg_confidence']}, "
        f"avg_time={report['summary']['avg_time_s']}s, "
        f"domains={report['summary']['domains_found']}, "
        f"kca={kca_triggered}"
    )

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/language/step_023000")
    parser.add_argument("--output", type=str, default="benchmark.json")
    parser.add_argument("--queries", type=int, default=12)
    args = parser.parse_args()
    run_benchmark(args.checkpoint, args.output, args.queries)
