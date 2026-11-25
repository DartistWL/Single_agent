import argparse
import os
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


COLLECTION_ID = 6829134  # Figshare collection id for StoreNet Ireland dataset
API_BASE = "https://api.figshare.com/v2"


def _safe_name(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    return name


def _download(
    session: requests.Session,
    url: str,
    out_path: Path,
    chunk_size: int = 1024 * 1024,
    max_tries: int = 8,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    last_err = None
    for i in range(max_tries):
        try:
            with session.get(url, stream=True, timeout=180, headers=headers) as r:
                # 206 = partial content (resume), 200 = full download
                if r.status_code not in (200, 206):
                    r.raise_for_status()

                total = r.headers.get("Content-Length")
                total = int(total) + existing if total and r.status_code == 206 else int(total or 0)
                done = existing

                mode = "ab" if existing > 0 and r.status_code == 206 else "wb"
                if mode == "wb":
                    existing = 0
                    done = 0
                    headers = {}  # no range for clean download

                with tmp_path.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100.0 / total
                            print(f"  {out_path.name}: {pct:6.2f}% ({done}/{total} bytes)", end="\r")

            tmp_path.replace(out_path)
            print(f"  Saved: {out_path}")
            return
        except Exception as e:
            last_err = e
            wait = min(60, 2**i)
            print(f"\n  Warning: download failed ({type(e).__name__}). retry in {wait}s")
            time.sleep(wait)
            # refresh resume offset
            existing = tmp_path.stat().st_size if tmp_path.exists() else 0
            if existing > 0:
                headers = {"Range": f"bytes={existing}-"}
            else:
                headers = {}

    raise last_err


def _build_session() -> requests.Session:
    """
    Build a requests session with retries.
    Note: trust_env=True (default) respects system proxies if present.
    """
    s = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "HEMS_project/StoreNet-downloader"})
    return s


def _get_json(session: requests.Session, url: str, timeout: int = 60, max_tries: int = 6):
    last_err = None
    for i in range(max_tries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = min(20, 2**i)
            print(f"  Warning: request failed ({type(e).__name__}). retry in {wait}s: {url}")
            time.sleep(wait)
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/storenet_raw",
        help="下载输出目录（会创建并保存 Figshare 原始文件）",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="可选：只下载匹配关键字的文件（逗号分隔），例如 'H1_W,Weather'。为空表示全下。",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = [s.strip() for s in args.only.split(",") if s.strip()]

    session = _build_session()

    print(f"Listing collection articles: {COLLECTION_ID}")
    # Figshare API paginates collection articles. Fetch all pages.
    articles = []
    page = 1
    page_size = 100
    while True:
        batch = _get_json(
            session,
            f"{API_BASE}/collections/{COLLECTION_ID}/articles?page={page}&page_size={page_size}",
            timeout=60,
        )
        if not isinstance(batch, list):
            raise SystemExit("Failed to list collection articles (unexpected response).")
        if not batch:
            break
        articles.extend(batch)
        page += 1

    if not articles:
        raise SystemExit("No articles returned for this collection.")

    # For each article, query its files and download
    for a in articles:
        article_id = a.get("id")
        title = a.get("title") or str(article_id)
        print(f"\nArticle {article_id}: {title}")
        try:
            article = _get_json(session, f"{API_BASE}/articles/{article_id}", timeout=60)
        except Exception as e:
            print(f"  Error: skip article due to repeated failures: {type(e).__name__}: {e}")
            continue
        files = article.get("files", [])
        if not files:
            print("  No files in this article, skip.")
            continue

        for f in files:
            name = _safe_name(f.get("name") or "file")
            download_url = f.get("download_url")
            if not download_url:
                continue

            if only and not any(k.lower() in name.lower() for k in only):
                continue

            # Keep extension from name; store under a folder by article title
            article_folder = out_dir / _safe_name(title)
            out_path = article_folder / name

            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"  Exists, skip: {out_path}")
                continue

            print(f"  Downloading: {name}")
            try:
                _download(session, download_url, out_path)
            except Exception as e:
                print(f"  Error: failed to download {name}. skip. ({type(e).__name__}: {e})")
                continue

    print("\nAll done.")


if __name__ == "__main__":
    # requests can be noisy on some environments; keep output clean
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()

