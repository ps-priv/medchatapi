"""Wspólne helpery dla testów integracyjnych — urllib, bez zewnętrznych zależności."""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8000"


def post(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status_code": e.code}


def get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(), "status_code": e.code}


def fmt_conviction(c: dict | None) -> str:
    if not c:
        return "  (brak conviction)"
    return (
        f"  interest={c.get('interest_level', 0):.2f} | "
        f"trust={c.get('trust_in_rep', 0):.2f} | "
        f"clinical={c.get('clinical_confidence', 0):.2f} | "
        f"fit={c.get('perceived_fit', 0):.2f} | "
        f"readiness={c.get('decision_readiness', 0):.2f}"
    )


def fmt_metrics(tm: dict) -> str:
    return (
        f"  metrics: topic={tm.get('topic_adherence', 0):.2f} | "
        f"clinical={tm.get('clinical_precision', 0):.2f} | "
        f"ethics={tm.get('ethics', 0):.2f} | "
        f"frustration={tm.get('frustration', 0):.2f}"
    )


def run_turns(session_id: str, turns: list[str], label: str = "") -> tuple[dict | None, bool]:
    """Przeprowadza kolejne tury. Zwraca (ostatnia odpowiedź, czy przerwana wcześniej)."""
    last_resp = None
    terminated_early = False
    for i, msg in enumerate(turns, 1):
        prefix = f"[{label} " if label else "["
        print(f"{prefix}Tura {i}] Przedstawiciel: {msg}")
        resp = post("/message", {"session_id": session_id, "message": msg})
        if "error" in resp:
            print(f"  BŁĄD: {resp['error']}")
            return last_resp, True
        print(
            f"{prefix}Tura {i}] Lekarz "
            f"({resp.get('doctor_attitude', '?')} | {resp.get('doctor_decision', '?')}):"
        )
        print(f"  {resp.get('doctor_message', '')}")
        print(fmt_conviction(resp.get("conviction")))
        print(fmt_metrics(resp.get("turn_metrics", {})))
        print()
        last_resp = resp
        if resp.get("is_terminated"):
            print(f"  *** Rozmowa zakończona: {resp.get('termination_reason', '')} ***")
            terminated_early = True
            break
    return last_resp, terminated_early


def assert_pass(condition: bool, msg: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_separator() -> None:
    print("─" * 60)
