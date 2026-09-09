"""Download the REASSEMBLE benchmark-split demos (h5 files) into data/reassemble/.

Streams individual .h5 entries out of the remote data.zip via HTTP range requests
(remotezip) instead of downloading the full ~59GB archive to disk first. Only pulls
files listed in splits/{train,test}_split1.txt — the official benchmark split used
for the F1@50 numbers referenced in the design spec.

The host throttles per connection (~1.7MB/s each, measured), not per IP — downloading
several files concurrently gets a roughly proportional aggregate speedup (measured ~2.5-3x
at 4 concurrent connections, some diminishing returns). Each worker opens its own RemoteZip
connection since a single instance isn't safe to share across threads.

Source: https://researchdata.tuwien.ac.at/records/0ewrv-8cb44 (CC-BY-4.0)
"""

import argparse
import concurrent.futures
from pathlib import Path

from remotezip import RemoteZip
from tqdm import tqdm

DATA_ZIP_URL = "https://researchdata.tuwien.ac.at/records/0ewrv-8cb44/files/data.zip?download=1"
DATA_DIR = Path(__file__).parent
SPLITS_DIR = DATA_DIR / "splits_inspect"
OUT_DIR = DATA_DIR / "reassemble"


def load_split_stems() -> set[str]:
    stems: set[str] = set()
    for name in ("train_split1.txt", "test_split1.txt"):
        with open(SPLITS_DIR / name) as f:
            stems.update(line.strip() for line in f if line.strip())
    return stems


def _download_one(filename: str, file_size: int) -> str:
    dest = OUT_DIR / Path(filename).name
    if dest.exists() and dest.stat().st_size == file_size:
        return f"skip {dest.name} (already complete)"

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with RemoteZip(DATA_ZIP_URL) as z:
        with z.open(filename) as src, open(tmp_dest, "wb") as out:
            while chunk := src.read(8 * 1024 * 1024):
                out.write(chunk)
    tmp_dest.rename(dest)
    return f"done {dest.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Only download the first N files (for a quick smoke test)."
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of files to download concurrently."
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted_stems = load_split_stems()

    with RemoteZip(DATA_ZIP_URL) as z:
        entries = [n for n in z.infolist() if n.filename.endswith(".h5")]
        to_fetch = [n for n in entries if Path(n.filename).stem in wanted_stems]
        to_fetch.sort(key=lambda n: n.filename)
        if args.limit:
            to_fetch = to_fetch[: args.limit]

    print(f"{len(to_fetch)} files to fetch with {args.workers} parallel workers -> {OUT_DIR}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_download_one, n.filename, n.file_size) for n in to_fetch]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="file"):
            print(future.result())

    print("done")


if __name__ == "__main__":
    main()
