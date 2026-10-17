"""Measured lexical regression checks, not a claim of semantic or safety accuracy."""
from statistics import median
import platform
import time

from knowledge_base.core import retrieve_with_diagnostics, RETRIEVAL_VERSION
from backend.persistence import now

CASES = [
    ("lidar scan dropout missing range returns", ["FLT-LIDAR", "HAZ-LIDAR", "EVD-LIDAR"]),
    ("proximity sensor stuck clear close range", ["FLT-PROXIMITY", "HAZ-PROXIMITY", "EVD-PROXIMITY"]),
    ("low sun glare saturated camera", ["FLT-GLARE", "COND-GLARE", "HAZ-GLARE"]),
    ("GNSS outage urban canyon", ["FLT-GNSS", "COND-GNSS", "HAZ-GNSS"]),
    ("stale map construction barrier", ["FLT-MAP", "COND-MAP", "HAZ-MAP"]),
    ("brake command acknowledgement lost", ["FLT-BRAKE", "COND-BRAKE", "HAZ-BRAKE"]),
    ("wheel encoder stuck odometry", ["FLT-WHEEL", "EVD-WHEEL", "HAZ-WHEEL"]),
    ("battery undervoltage brownout reset", ["FLT-BATTERY", "HAZ-BATTERY", "EVD-BATTERY"]),
    ("timestamp synchronization camera lidar clocks", ["FLT-TIME", "COND-TIME", "HAZ-TIME"]),
    ("wireless packet delay jitter", ["FLT-NETWORK", "EVD-NETWORK", "HAZ-NETWORK"]),
    ("replayed stale control message freshness", ["FLT-REPLAY", "HAZ-REPLAY", "MIT-REPLAY"]),
    ("cargo payload latch retention", ["FLT-CARGO", "HAZ-CARGO", "MIT-CARGO"]),
    ("crowd occlusion child wheelchair", ["FLT-CROWD", "COND-CROWD", "HAZ-CROWD"]),
    ("icy surface wheel traction slip", ["FLT-ICE", "COND-ICE", "HAZ-ICE"]),
    ("steering angle feedback mismatch", ["FLT-STEERING", "COND-STEERING", "HAZ-STEERING"]),
    ("thermal throttling compute backlog", ["FLT-THERMAL", "EVD-THERMAL", "HAZ-THERMAL"]),
]


def benchmark_retrieval(dataset, cases=None, repeats=3):
    if not 1 <= repeats <= 20:
        raise ValueError("Choose 1-20 timing repetitions")
    cases = CASES if cases is None else cases
    ids = {r.id for r in dataset.records}
    applicable = [(q, expected) for q, expected in cases if set(expected) <= ids]
    if not applicable:
        raise ValueError("No bundled lexical labels match this dataset; use the retrieval inspector for custom queries")
    rows, timings, total_start = [], [], time.perf_counter()
    for query, expected in applicable:
        results, diagnostics = retrieve_with_diagnostics(dataset, query, limit=5, threshold=0)
        direct = [r["record"]["id"] for r in results if r["retrieval"] == "tfidf"]
        ranks = [i + 1 for i, ref in enumerate(direct) if ref in expected]
        rows.append({"query": query, "expected_relevant_ids": expected, "direct_top5": direct,
                     "hit_at_5": bool(ranks), "recall_at_5": len(set(direct) & set(expected)) / len(expected),
                     "reciprocal_rank": 1 / min(ranks) if ranks else 0})
        for _ in range(repeats):
            _, sample = retrieve_with_diagnostics(dataset, query, limit=5, threshold=0)
            timings.append(sample["total_ms"])
    timings.sort()
    return {"created": now(), "dataset_version": dataset.version, "retrieval_version": RETRIEVAL_VERSION,
            "label_set": "bundled-lexical-regression-16-v1", "case_count": len(rows), "timing_samples": len(timings),
            "hit_at_5": sum(r["hit_at_5"] for r in rows)/len(rows),
            "mean_recall_at_5": sum(r["recall_at_5"] for r in rows)/len(rows),
            "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in rows)/len(rows),
            "warm_p50_total_ms": median(timings), "warm_p95_total_ms": timings[max(0, (95 * len(timings) + 99)//100 - 1)],
            "index_build_ms": diagnostics["index_build_ms"], "indexed_records": len(dataset.records),
            "duration_ms": round((time.perf_counter()-total_start)*1000, 3), "python": platform.python_version(),
            "machine": platform.platform(), "cases": rows,
            "limitations": "Small curated lexical document-matching regression set, not independently reviewed semantic relevance or safety accuracy. Direct top-5 only; relationship expansion cannot inflate scores. Timings describe this machine and dataset."}


def benchmark_scaling(dataset, counts=(181, 1000, 5000), repeats=10):
    """Synthetic index-load diagnostics only; never activate this data as knowledge."""
    from knowledge_base.core import Dataset, Record
    rows = []
    for count in counts:
        if not 1 <= count <= 5000 or not 1 <= repeats <= 100:
            raise ValueError("Stress workload exceeds dataset/repetition limits")
        stress = Dataset(schema_version="1.0", dataset_id=f"synthetic-index-load-{count}", dataset_name="Synthetic index-load fixture, not engineering evidence",
                         records=[Record(id=f"LOAD-{i}", type="evidence", title=f"Load fixture {i}: {dataset.records[i % len(dataset.records)].title}",
                                         description=dataset.records[i % len(dataset.records)].description) for i in range(count)])
        _, first = retrieve_with_diagnostics(stress, "lidar scan dropout missing range returns", limit=5)
        durations = [retrieve_with_diagnostics(stress, "lidar scan dropout missing range returns", limit=5)[1]["total_ms"] for _ in range(repeats)]
        durations.sort()
        rows.append({"dataset_kind":"synthetic_index_load", "record_count":count,"first_request_total_ms":first["total_ms"],
                     "index_build_ms":first["index_build_ms"],"warm_p50_total_ms":median(durations),
                     "warm_p95_total_ms":durations[max(0,(95*len(durations)+99)//100-1)],"timing_samples":repeats})
    return rows


if __name__ == "__main__":
    import argparse
    import json
    from backend.services import Services
    parser = argparse.ArgumentParser(description="Measure current active KB lexical retrieval; no model inference")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="Save actual measurements in local history")
    parser.add_argument("--scale", action="store_true", help="Also measure separate synthetic index loads up to 5000 records")
    arguments = parser.parse_args()
    service = Services()
    measured = benchmark_retrieval(service.dataset(), repeats=arguments.repeats)
    if arguments.scale:
        measured["synthetic_load_scaling"] = benchmark_scaling(service.dataset())
    if arguments.save:
        service.store.save("retrieval_benchmark", measured)
    print(json.dumps(measured, indent=2))
