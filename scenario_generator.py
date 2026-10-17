"""
scenario_generator.py — Exact Real Data Scenarios for Streamlit Demo

This module loads exactly one RAW test window for each attack class, extracted
directly from the dataset by `extract_scenarios.py`. It guarantees 100% fidelity
with the training patterns, avoiding the pitfalls of synthetic generation.
"""

import json
import os
import numpy as np

# Column indices
KPI_NAMES = [
    "dlBytes", "dlMcs", "dlBler", "ulBytes", "ulMcs", "ulBler",
    "ri", "phr", "pcmax", "rsrq", "sinr", "rsrp", "rssi",
    "cqi", "pucchSnr", "puschSnr",
]
N_KPIS = len(KPI_NAMES)

_demo_scenarios = {}
_loaded = False

def load_demo_scenarios(scenario_file: str = "demo_scenarios.json"):
    """Load the raw test windows extracted from the DB/CSV dataset."""
    global _demo_scenarios, _loaded
    if _loaded:
        return

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), scenario_file)
    if not os.path.exists(path):
        for candidate in ["demo_scenarios.json", os.path.join("demo_old", scenario_file), os.path.join("demo", scenario_file)]:
            if os.path.exists(candidate):
                path = candidate
                break

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                _demo_scenarios[k] = np.array(v, dtype=np.float32)
        print(f"[scenario_generator] Loaded {_demo_scenarios.keys()} exact real scenarios.")
    except Exception as e:
        print(f"[scenario_generator] WARNING: Could not load demo_scenarios.json: {e}")
        print("Please run `python extract_scenarios.py` first!")
    _loaded = True

def _get_exact_window(category: str, subcategory: str, noise: float = 0.0) -> np.ndarray:
    """Return the exact raw array from the test set, optionally with some relative noise."""
    key = f"{category}/{subcategory}"
    if key not in _demo_scenarios:
        print(f"Warning: {key} not found. Returning zeros.")
        return np.zeros((24, N_KPIS), dtype=np.float32)

    base = _demo_scenarios[key]
    if noise > 0.0:
        # Add relative noise so we don't completely destroy the patterns
        rng = np.random.default_rng(42)
        noise_arr = rng.normal(0, noise, base.shape).astype(np.float32)
        return base * (1.0 + noise_arr)
    return base

# =============================================================================
# Exact Scenario Wrappers
# =============================================================================

def generate_benign(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Benign", "Benign", noise)

# Probe
def generate_probe_portscan(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Probe", "Probe_PortScan", noise)
def generate_probe_osfingerprinting(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Probe", "Probe_OsFingerprinting", noise)

# BruteForce
def generate_bruteforce_ssh(noise: float = 0.0) -> np.ndarray: return _get_exact_window("BruteForce", "BruteForce_SSH", noise)
def generate_bruteforce_ftp(noise: float = 0.0) -> np.ndarray: return _get_exact_window("BruteForce", "BruteForce_FTP", noise)

# Web Attacks
def generate_web_http_bruteforce(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Web Attacks", "Web_HTTP_Bruteforce", noise)
def generate_web_directory_bruteforce(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Web Attacks", "Web_Directory_Bruteforce", noise)
def generate_web_sql_injection(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Web Attacks", "Web_SQL_Injection", noise)
def generate_web_xss(noise: float = 0.0) -> np.ndarray: return _get_exact_window("Web Attacks", "Web_XSS", noise)

# DDoS
def generate_ddos_slowloris(noise: float = 0.0) -> np.ndarray: return _get_exact_window("DDoS", "DDoS_Slowloris", noise)
def generate_ddos_syn(noise: float = 0.0) -> np.ndarray: return _get_exact_window("DDoS", "DDoS_SYN", noise)
def generate_ddos_udp(noise: float = 0.0) -> np.ndarray: return _get_exact_window("DDoS", "DDoS_UDP", noise)
def generate_ddos_ack(noise: float = 0.0) -> np.ndarray: return _get_exact_window("DDoS", "DDoS_ACK", noise)
def generate_ddos_icmp(noise: float = 0.0) -> np.ndarray: return _get_exact_window("DDoS", "DDoS_ICMP", noise)

# =============================================================================
# Registry
# =============================================================================

def _make_entry(fn, desc, color):
    return {
        "fn": fn,
        "color": color,
        "description": desc,
    }

SCENARIO_REGISTRY = {
    "Benign": _make_entry(generate_benign, "Exact Benign window from the test dataset.", "#22c55e"),

    "Probe / Probe_PortScan": _make_entry(generate_probe_portscan, "Exact PortScan window.", "#8b5cf6"),
    "Probe / Probe_OsFingerprinting": _make_entry(generate_probe_osfingerprinting, "Exact OsFingerprinting window.", "#a78bfa"),

    "BruteForce / BruteForce_SSH": _make_entry(generate_bruteforce_ssh, "Exact SSH BruteForce window.", "#06b6d4"),
    "BruteForce / BruteForce_FTP": _make_entry(generate_bruteforce_ftp, "Exact FTP BruteForce window.", "#22d3ee"),

    "Web Attacks / Web_HTTP_Bruteforce": _make_entry(generate_web_http_bruteforce, "Exact HTTP BruteForce window.", "#ec4899"),
    "Web Attacks / Web_Directory_Bruteforce": _make_entry(generate_web_directory_bruteforce, "Exact Directory BruteForce window.", "#f472b6"),
    "Web Attacks / Web_SQL_Injection": _make_entry(generate_web_sql_injection, "Exact SQL Injection window.", "#db2777"),
    "Web Attacks / Web_XSS": _make_entry(generate_web_xss, "Exact XSS window.", "#be185d"),

    "DDoS / DDoS_Slowloris": _make_entry(generate_ddos_slowloris, "Exact Slowloris window.", "#eab308"),
    "DDoS / DDoS_SYN": _make_entry(generate_ddos_syn, "Exact SYN flood window.", "#f97316"),
    "DDoS / DDoS_UDP": _make_entry(generate_ddos_udp, "Exact UDP flood window.", "#ef4444"),
    "DDoS / DDoS_ACK": _make_entry(generate_ddos_ack, "Exact ACK flood window.", "#fb923c"),
    "DDoS / DDoS_ICMP": _make_entry(generate_ddos_icmp, "Exact ICMP flood window.", "#dc2626"),
}
