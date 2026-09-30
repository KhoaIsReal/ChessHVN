"""Utility script to download Grandmaster PGN databases and benchmark FEN datasets.

Usage:
  uv run python scripts/download_data.py --all
  uv run python scripts/download_data.py --gm fischer kasparov carlsen
  uv run python scripts/download_data.py --benchmarks
"""

import argparse
import io
from pathlib import Path
import urllib.request
import zipfile
import chess

PGN_MENTOR_URLS = {
    "fischer": "https://www.pgnmentor.com/players/Fischer.zip",
    "kasparov": "https://www.pgnmentor.com/players/Kasparov.zip",
    "carlsen": "https://www.pgnmentor.com/players/Carlsen.zip",
    "tal": "https://www.pgnmentor.com/players/Tal.zip",
    "karpov": "https://www.pgnmentor.com/players/Karpov.zip",
    "capablanca": "https://www.pgnmentor.com/players/Capablanca.zip",
}

# Classic Bratko-Kopec Test (BKT) - 24 standard benchmark positions
BRATKO_KOPEC_FENS = [
    ("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", -0.9, "BKT 01"),
    ("3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1", 0.35, "BKT 02"),
    ("2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1", 0.0, "BKT 03"),
    ("rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1", 0.85, "BKT 04"),
    ("r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1", 0.45, "BKT 05"),
    ("2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1", 0.6, "BKT 06"),
    ("1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P4Q/P1P2RPP/R3KB2 w Q - 0 1", 0.5, "BKT 07"),
    ("4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1", 0.2, "BKT 08"),
    ("2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1", 0.55, "BKT 09"),
    ("3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1", -0.4, "BKT 10"),
    ("2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1", 0.3, "BKT 11"),
    ("r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1", -0.8, "BKT 12"),
    ("r2q1rk1/4bppp/p2p4/2pPn3/3p4/3B3P/PPP2PP1/R1BQR1K1 w - - 0 1", 0.25, "BKT 13"),
    ("rnbqk2r/2ppbppp/p7/1p1QP3/4n3/1B6/PPP2PPP/RNB1K1NR b KQkq - 0 1", -1.0, "BKT 14"),
    ("r1b2rk1/1pp1qppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPPBQPPP/R4RK1 w - - 0 1", 0.1, "BKT 15"),
    ("rnbqkb1r/pppp1ppp/8/4P3/6n1/7P/PPPNPPP1/R1BQKBNR b KQkq - 0 1", -0.95, "BKT 16"),
    ("r1b1k2r/1p1nbppp/pq1p4/3P4/3NP3/1BN5/PP4PP/R2Q1RK1 b kq - 0 1", -0.3, "BKT 17"),
    ("2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1", 0.0, "BKT 18"),
    ("r1bqr1k1/pp1n1ppp/2p5/3p4/3P1Pn1/2NBB3/PPP3PP/R2QR1K1 w - - 0 1", -0.7, "BKT 19"),
    ("r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1", 0.0, "BKT 20"),
    ("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 1", 0.3, "BKT 21"),
    ("r1b2rk1/pp1n1ppp/4p3/2qpP3/5P2/3B1N2/PPP3PP/R2QK2R w KQ - 0 1", 0.8, "BKT 22"),
    ("r2q1rk1/ppp2ppp/2n1b3/2bnp3/2B5/2PP1N2/PP1N1PPP/R1BQR1K1 b - - 0 1", -0.1, "BKT 23"),
    ("r1bqk2r/pp2bppp/2n1p3/2ppP3/3P4/2PB1N2/P1P2PPP/R1BQK2R w KQkq - 0 1", 0.2, "BKT 24"),
]

# Strategic endgame and tactical benchmark positions
TACTICAL_BENCHMARKS = [
    ("8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1", 1.0, "K+Q vs K White Win"),
    ("8/8/8/4k3/8/8/4K3/4q3 b - - 0 1", -1.0, "K+q vs K Black Win"),
    ("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1", 0.9, "K+R vs K White Win"),
    ("8/8/8/4k3/8/8/4K3/8 w - - 0 1", 0.0, "Bare Kings Draw"),
    ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", 0.6, "King and Pawn vs King"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 0.15, "Ruy Lopez / Open"),
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2", 0.1, "Sicilian Defense"),
    ("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2", 0.15, "Indian Defense"),
    ("r1b1k2r/pppp1ppp/8/8/1b1qn3/2N5/PPPB1PPP/R2QKB1R w KQkq - 0 8", -0.9, "Black Queen Attack"),
    ("r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", 1.0, "Scholar's Mate White Win"),
]


def download_url(url: str) -> bytes:
    """Download content with browser User-Agent header."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChessResearchBot/1.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def download_pgn_player(player_key: str, dest_dir: Path) -> Path:
    """Download and extract a Grandmaster PGN database."""
    url = PGN_MENTOR_URLS.get(player_key.lower())
    if not url:
        raise ValueError(f"Unknown player '{player_key}'. Available: {list(PGN_MENTOR_URLS.keys())}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {player_key.title()}'s games from {url}...")

    raw_zip = download_url(url)
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        z.extractall(dest_dir)

    pgn_files = list(dest_dir.glob(f"*{player_key}*.pgn")) + list(dest_dir.glob(f"*{player_key.title()}*.pgn"))
    if not pgn_files:
        pgn_files = list(dest_dir.glob("*.pgn"))

    out_file = pgn_files[0] if pgn_files else dest_dir
    print(f"  -> Extracted to: {out_file}")
    return out_file


def save_fen_benchmarks(dest_file: Path) -> None:
    """Save Bratko-Kopec and Tactical benchmark positions in standard annotated FEN format."""
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    all_positions = BRATKO_KOPEC_FENS + TACTICAL_BENCHMARKS
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write("# Chess AI Benchmark Positions (Bratko-Kopec Test + Tactical Endgames)\n")
        f.write("# Format: <FEN>; <Score in [-1.0, 1.0] from White perspective>; <Description>\n\n")
        for fen, score, desc in all_positions:
            f.write(f"{fen}; {score:.2f}; {desc}\n")
    print(f"Saved {len(all_positions)} benchmark positions to {dest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download chess grandmaster PGNs and benchmark FEN datasets")
    parser.add_argument("--all", action="store_true", help="Download all GM databases (Fischer, Kasparov, Carlsen, Tal) and benchmark FENs")
    parser.add_argument("--gm", nargs="+", choices=list(PGN_MENTOR_URLS.keys()), help="Download specific grandmaster PGNs")
    parser.add_argument("--benchmarks", action="store_true", help="Generate benchmark positions (Bratko-Kopec + Tactics)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")

    args = parser.parse_args()
    data_dir = Path(args.output_dir)

    if not args.all and not args.gm and not args.benchmarks:
        parser.print_help()
        print("\nExamples:")
        print("  uv run python scripts/download_data.py --all")
        print("  uv run python scripts/download_data.py --gm fischer kasparov carlsen")
        print("  uv run python scripts/download_data.py --benchmarks")
        return

    # 1. Download GM databases
    gm_list = list(PGN_MENTOR_URLS.keys()) if args.all else (args.gm or [])
    if gm_list:
        gm_dir = data_dir / "grandmaster"
        print(f"\n[1/2] Fetching Grandmaster PGN databases to {gm_dir}...")
        for p in gm_list:
            try:
                download_pgn_player(p, gm_dir)
            except Exception as e:
                print(f"  [ERROR] Could not download {p}: {e}")

    # 2. Generate benchmark FENs
    if args.all or args.benchmarks:
        print(f"\n[2/2] Generating benchmark FEN positions...")
        pos_file = data_dir / "benchmark_positions.fen"
        save_fen_benchmarks(pos_file)

    print("\nDone! Data is ready for multi-source training.")


if __name__ == "__main__":
    main()
