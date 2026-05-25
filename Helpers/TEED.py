from __future__ import annotations

import argparse
import math
import re


# Edit these once to match your standard setting.
BASE_AMP_UA = 25.0
BASE_FREQ_HZ = 135.0
BASE_PW_US = 100.0


def teed_constant(amp_uA: float, freq_hz: float, pw_us: float) -> float:
    return float(amp_uA) ** 2 * float(freq_hz) * float(pw_us)


def baseline_teed() -> float:
    return teed_constant(BASE_AMP_UA, BASE_FREQ_HZ, BASE_PW_US)


def solve_missing(target: str, amp_uA: float, freq_hz: float, pw_us: float, teed: float) -> float:
    if target == "amp":
        return math.sqrt(teed / (float(freq_hz) * float(pw_us)))
    if target == "freq":
        return teed / (float(amp_uA) ** 2 * float(pw_us))
    if target == "pw":
        return teed / (float(amp_uA) ** 2 * float(freq_hz))
    raise ValueError(f"Unknown target: {target}")


def normalize_name(raw: str) -> str:
    token = raw.strip().lower()
    token = re.sub(r"^(and|,)\s*", "", token)
    token = token.replace(" ", "")
    aliases = {
        "amp": "amp",
        "amplitude": "amp",
        "voltage": "amp",
        "v": "amp",
        "frq": "freq",
        "freq": "freq",
        "frequency": "freq",
        "f": "freq",
        "pw": "pw",
        "pulsewidth": "pw",
        "pulse_width": "pw",
        "pulse": "pw",
    }
    if token not in aliases:
        raise ValueError(f"Unknown variable: {raw}")
    return aliases[token]


def parse_query(query: str) -> tuple[str, dict[str, float]]:
    text = query.strip().lower()
    target_match = re.search(r"new\s+([a-z_ ]+?)\s+if", text)
    if not target_match:
        raise ValueError("Use a query like: new amp if frq = 40")
    target = normalize_name(target_match.group(1))

    assignments: dict[str, float] = {}
    condition_text = text.split("if", 1)[1]
    for raw_name, raw_value in re.findall(r"([a-z_ ]+?)\s*=\s*([-+]?\d*\.?\d+)", condition_text):
        key = normalize_name(raw_name)
        assignments[key] = float(raw_value)

    if target in assignments:
        raise ValueError("Target variable cannot also be fixed in the same query.")
    return target, assignments


def solve_query(query: str) -> tuple[str, float]:
    target, assignments = parse_query(query)
    amp_uA = assignments.get("amp", BASE_AMP_UA)
    freq_hz = assignments.get("freq", BASE_FREQ_HZ)
    pw_us = assignments.get("pw", BASE_PW_US)
    value = solve_missing(target, amp_uA, freq_hz, pw_us, baseline_teed())
    return target, value


def print_baseline() -> None:
    print(
        "Baseline: "
        f"amp={BASE_AMP_UA:.3f} uA, "
        f"freq={BASE_FREQ_HZ:.3f} Hz, "
        f"pw={BASE_PW_US:.3f} us, "
        f"balance={baseline_teed():.6g} uA^2*Hz*us"
    )


def format_result(target: str, value: float) -> str:
    unit = {"amp": "uA", "freq": "Hz", "pw": "us"}[target]
    label = {"amp": "amp", "freq": "freq", "pw": "pw"}[target]
    return f"new {label} = {value:.2f} {unit}"


def repl() -> None:
    print_baseline()
    print("Type queries like: new amp if frq = 40")
    print("You can also do: new amp if frq = 40 and pw = 60")
    print("Type 'baseline' to show the baseline again, or 'quit' to exit.")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        lower = raw.lower()
        if lower in {"quit", "exit", "q"}:
            return
        if lower in {"baseline", "base"}:
            print_baseline()
            continue
        try:
            target, value = solve_query(raw)
            print(format_result(target, value))
        except Exception as exc:
            print(f"Could not parse query: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constant-TEED rebalance helper around one baseline setting."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Example: new amp if frq = 40",
    )
    parser.add_argument("--show-baseline", action="store_true")
    args = parser.parse_args()

    if args.show_baseline:
        print_baseline()
        return

    if args.query:
        query = " ".join(args.query)
        target, value = solve_query(query)
        print_baseline()
        print(format_result(target, value))
        return

    repl()


if __name__ == "__main__":
    main()
