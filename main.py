import bisect
import heapq
import json
import math
import os
import random
from datetime import datetime, timezone


INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

PATHS = 4000


def parse_hours(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() / 3600.0


def log_cap(row):
    lamports = int(str(row["market_cap_lamports"]))
    if lamports <= 0:
        raise ValueError("market cap must be positive")
    return math.log10(lamports) - 9.0


def default_edges():
    return [2.0 + 0.05 * i for i in range(81)]


def load_task():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as handle:
        task = json.load(handle)
    if not isinstance(task, dict):
        raise ValueError("task must be an object")
    return task


def load_marks():
    marks = []
    path = os.path.join(INPUT_DIR, "history.ndjson")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                mark = json.loads(line)
            except Exception:
                continue
            if isinstance(mark, dict) and isinstance(mark.get("rows"), list):
                marks.append(mark)
    if not marks:
        raise ValueError("history has no marks")
    return marks


def rounded_to_cap(rows, cap, exact=False):
    total = math.fsum(value for _, value in rows)
    if total > 0.0 and (exact or total > cap):
        factor = cap / total
    else:
        factor = 1.0
    result = [(key, round(value * factor, 6)) for key, value in rows]
    result = [(key, min(1.0, max(0.0, value))) for key, value in result]

    excess = math.fsum(value for _, value in result) - cap
    if excess > 0.0:
        largest = max(range(len(result)), key=lambda index: result[index][1])
        key, value = result[largest]
        result[largest] = (key, round(max(0.0, value - excess), 6))
    return result


def compute(task, marks):
    places = int(task.get("places", 5))
    edges = [float(edge) for edge in task["size"]["edges"]]
    reads = list(task.get("reads") or [])

    series = {}
    for mark in marks:
        at = parse_hours(mark["at"])
        for row in mark["rows"]:
            mint = row["mint"]
            if not isinstance(mint, str) or not mint:
                continue
            series.setdefault(mint, []).append((at, log_cap(row)))

    coins = []
    residuals = []
    for mint, observations in series.items():
        observations.sort(key=lambda item: item[0])
        first_t, first_x = observations[0]
        last_t, last_x = observations[-1]
        dt = last_t - first_t
        if len(observations) >= 2 and dt > 0.0:
            mu = (last_x - first_x) / dt
            mu = min(0.35, max(-0.25, mu))
        else:
            mu = 0.03

        if len(observations) >= 2:
            for (t1, x1), (t2, x2) in zip(observations, observations[1:]):
                step = t2 - t1
                if step > 0.0:
                    residuals.append(((x2 - x1) - mu * step) ** 2 / step)

        coins.append((mint, last_x, mu))

    if residuals:
        sigma = math.sqrt(math.fsum(residuals) / len(residuals))
        sigma = min(0.60, max(0.02, sigma))
    else:
        sigma = 0.12

    last_time = parse_hours(marks[-1]["at"])
    hours = []
    for read in reads:
        delta = parse_hours(read) - last_time
        if delta > 0.0:
            hours.append(delta)
    if not hours:
        hours = [1.0] * len(reads)
    if not hours:
        raise ValueError("task has no reads")

    rng = random.Random(12345)
    place_counts = {place: {} for place in range(1, places + 1)}
    place_bins = {place: {} for place in range(1, places + 1)}

    for _ in range(PATHS):
        for hours_ahead in hours:
            root_hours = math.sqrt(hours_ahead)
            simulated = []
            for mint, start, mu in coins:
                x = start + mu * hours_ahead + sigma * root_hours * rng.gauss(0.0, 1.0)
                x = min(7.0, max(1.0, x))
                simulated.append((mint, x))
            ranked = heapq.nsmallest(
                places, simulated, key=lambda item: (-item[1], item[0])
            )
            for place, (mint, cap) in enumerate(ranked, start=1):
                counts = place_counts[place]
                counts[mint] = counts.get(mint, 0) + 1
                bin_number = bisect.bisect_right(edges, cap)
                bins = place_bins[place]
                bins[bin_number] = bins.get(bin_number, 0) + 1

    probabilities = []
    for place in range(1, places + 1):
        kept = [
            (mint, count / PATHS)
            for mint, count in place_counts[place].items()
            if count / PATHS >= 0.002
        ]
        kept.sort(key=lambda item: (-item[1], item[0]))
        kept = kept[:60]
        for mint, probability in rounded_to_cap(kept, 0.85):
            probabilities.append({"mint": mint, "place": place, "p": probability})

    total_samples = PATHS * len(hours)
    sizes = []
    for place in range(1, places + 1):
        kept = [
            (bin_number, count / total_samples)
            for bin_number, count in place_bins[place].items()
            if count / total_samples >= 0.002
        ]
        kept.sort(key=lambda item: (-item[1], item[0]))
        kept = kept[:60]
        for bin_number, probability in rounded_to_cap(kept, 0.90):
            sizes.append({"place": place, "bin": bin_number, "p": probability})

    mint_totals = {}
    mint_best = {}
    for row in probabilities:
        mint = row["mint"]
        place = row["place"]
        probability = row["p"]
        mint_totals[mint] = mint_totals.get(mint, 0.0) + probability
        previous = mint_best.get(mint)
        if previous is None or place < previous[0] or (place == previous[0] and probability > previous[1]):
            mint_best[mint] = (place, probability)

    leaders = sorted(mint_totals, key=lambda mint: (-mint_totals[mint], mint))[:5]
    if len(leaders) < 5:
        for mint, _, _ in sorted(coins, key=lambda item: (-item[1], item[0])):
            if mint not in mint_best:
                mint_best[mint] = (places + 1, 0.0)
                mint_totals[mint] = 0.0
                leaders.append(mint)
            if len(leaders) == 5:
                break
    leaders.sort(
        key=lambda mint: (
            mint_best[mint][0],
            -mint_best[mint][1],
            -mint_totals[mint],
            mint,
        )
    )

    return {"probabilities": probabilities, "top5": leaders, "sizes": sizes}


def fallback(task, marks):
    edges = default_edges()
    if isinstance(task, dict):
        try:
            edges = [float(edge) for edge in task["size"]["edges"]]
        except Exception:
            pass

    chart = []
    if marks:
        rows = [
            row
            for row in marks[-1].get("rows", [])
            if isinstance(row, dict)
            and isinstance(row.get("mint"), str)
            and row.get("mint")
        ]
        rows.sort(key=lambda row: (int(row.get("rank", 10**9)), str(row.get("mint", ""))))
        chart = rows

    mints = [row.get("mint") for row in chart if isinstance(row.get("mint"), str) and row.get("mint")]
    top5 = mints[:5]

    probabilities = []
    sizes = []
    for place in range(1, 6):
        entries = {}
        if place <= len(chart):
            entries[chart[place - 1]["mint"]] = 0.60
        if 1 <= place - 1 <= len(chart):
            mint = chart[place - 2]["mint"]
            entries[mint] = entries.get(mint, 0.0) + 0.15
        if place + 1 <= len(chart):
            mint = chart[place]["mint"]
            entries[mint] = entries.get(mint, 0.0) + 0.15

        rows = sorted(entries.items(), key=lambda item: (-item[1], item[0]))
        for mint, probability in rounded_to_cap(rows, 0.85, exact=True):
            probabilities.append({"mint": mint, "place": place, "p": probability})

        if place <= len(chart):
            try:
                bin_number = bisect.bisect_right(edges, log_cap(chart[place - 1]))
                sizes.append({"place": place, "bin": bin_number, "p": 0.9})
            except Exception:
                pass

    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task = None
    marks = []
    try:
        task = load_task()
        marks = load_marks()
        forecast = compute(task, marks)
    except Exception:
        try:
            if task is None:
                task = load_task()
            if not marks:
                marks = load_marks()
            forecast = fallback(task, marks)
        except Exception:
            forecast = {"probabilities": [], "top5": [], "sizes": []}

    with open(os.path.join(OUTPUT_DIR, "forecast.json"), "w", encoding="utf-8") as handle:
        json.dump(forecast, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
