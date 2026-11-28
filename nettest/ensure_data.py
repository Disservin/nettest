from .utils import execute, BUCKET_PREFIX
from pathlib import Path
from time import sleep
from huggingface_hub import snapshot_download
import threading
import zstandard as zstd
import gzip
import shutil
import os


def decompress_file_zstd(file_path):
    output_path = file_path[:-4]  # Remove .zst extension
    try:
        print(f"Decompressing: {file_path}")
        with open(file_path, "rb") as compressed, open(
            output_path, "wb"
        ) as decompressed:
            dctx = zstd.ZstdDecompressor()
            dctx.copy_stream(compressed, decompressed)
        os.remove(file_path)
        print(f"Decompressed and removed: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error decompressing {file_path} : {e}") from e


def decompress_file_gz(file_path):
    output_path = file_path[:-3]  # Remove .gz extension
    try:
        print(f"Decompressing: {file_path}")
        with open(output_path, "wb") as decompressed:
            with gzip.open(file_path, "rb") as gzfile:
                shutil.copyfileobj(gzfile, decompressed)
        os.remove(file_path)
        print(f"Decompressed and removed: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error decompressing {file_path} : {e}") from e


def decompress_files_in_threads(file_list, decompress_file):
    threads = []
    for file_path in file_list:
        t = threading.Thread(target=decompress_file, args=(file_path,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# compressed variants that may be stored instead of the plain file, in order of preference
COMPRESSED_SUFFIXES = (".zst", ".gz")


def with_retries(what, f, n_repeats=3, delay=30):
    """try a couple of times, since we might be overloading hf"""
    while True:
        try:
            return f()
        except Exception as e:
            print(f"Error during {what}: {e}")
            n_repeats -= 1
            if n_repeats > 0:
                print(f"Retrying in {delay} seconds... ({n_repeats} attempts left)")
                sleep(delay)
            else:
                raise RuntimeError(f"Failed {what} after multiple attempts.") from e


def download_from_dataset(owner: str, repo: str, filenames: list[str], repo_dir: Path):
    # download the filename or e.g. compressed files like *.zst
    pattern_filenames = []
    for filename in filenames:
        pattern_filenames.append(filename)
        pattern_filenames.extend(filename + suffix for suffix in COMPRESSED_SUFFIXES)

    with_retries(
        f"update of dataset {owner}/{repo}",
        lambda: snapshot_download(
            repo_id=f"{owner}/{repo}",
            repo_type="dataset",
            allow_patterns=pattern_filenames,
            cache_dir=repo_dir,
            local_dir=repo_dir,
            etag_timeout=600,
        ),
    )


def download_from_bucket(owner: str, bucket: str, filenames: list[str], repo_dir: Path):
    from huggingface_hub import download_bucket_files, get_bucket_paths_info

    bucket_id = f"{owner}/{bucket}"

    # find out which variant (plain or compressed) is present in the bucket
    candidates = []
    for filename in filenames:
        candidates.append(filename)
        candidates.extend(filename + suffix for suffix in COMPRESSED_SUFFIXES)
    available = with_retries(
        f"listing of bucket {bucket_id}",
        lambda: {
            info.path: info for info in get_bucket_paths_info(bucket_id, candidates)
        },
    )

    files = []
    for filename in filenames:
        for remote in [filename] + [filename + s for s in COMPRESSED_SUFFIXES]:
            if remote in available:
                files.append((available[remote], repo_dir / remote))
                break
        else:
            raise FileNotFoundError(
                f"File {filename} (or compressed variant) not found in bucket {bucket_id}."
            )

    # download to a temporary name, and only move into place once complete,
    # so that an interrupted download is not mistaken for a valid file
    for remote_file, local_path in files:
        tmp_path = local_path.with_name(local_path.name + ".incomplete")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {bucket_id}/{remote_file.path} ({remote_file.size} bytes)")
        with_retries(
            f"download of {bucket_id}/{remote_file.path}",
            lambda: download_bucket_files(
                bucket_id, [(remote_file, tmp_path)], raise_on_missing_files=True
            ),
        )
        os.replace(tmp_path, local_path)


def run_data_update(
    owner: str, repo: str, filenames: list[str], source: str = "dataset"
) -> None:
    if source == "dataset":
        repo_dir = Path().cwd() / "data" / owner / repo
    elif source == "bucket":
        repo_dir = Path().cwd() / "data" / BUCKET_PREFIX / owner / repo
    else:
        raise ValueError(f"Unknown data source {source}, expected dataset or bucket")
    repo_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Ensuring data for {source} {owner}/{repo} in {repo_dir} : "
        + " ".join(filenames)
    )

    missing = [f for f in filenames if not (repo_dir / f).exists()]
    if missing:
        if source == "dataset":
            download_from_dataset(owner, repo, missing, repo_dir)
        else:
            download_from_bucket(owner, repo, missing, repo_dir)
        print("", flush=True)
        execute("Repo disk usage: ", ["du", "-sh", "."], repo_dir, True)

    # collect the compressed files that need decompression
    for suffix, decompress_file in (
        (".zst", decompress_file_zstd),
        (".gz", decompress_file_gz),
    ):
        compressed_files = []
        for filename in filenames:
            compressed_file = repo_dir / f"{filename}{suffix}"
            std_file = repo_dir / filename
            if compressed_file.exists() and not std_file.exists():
                compressed_files.append(str(compressed_file))
        decompress_files_in_threads(compressed_files, decompress_file)

    execute("Repo disk usage: ", ["du", "-sh", "."], repo_dir, True)

    # final check
    for filename in filenames:
        stored_file = repo_dir / filename
        if not stored_file.exists():
            raise FileNotFoundError(
                f"File {filename} not found in {source} {owner}/{repo} after update."
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensure HF data is present in cwd/data for a given owner, and repo."
    )
    parser.add_argument(
        "--bucket",
        action="store_true",
        help="Download from a HF bucket (owner/bucket) instead of a HF dataset repo",
    )
    parser.add_argument("owner", help="Repository / bucket owner")
    parser.add_argument("repo", help="Repository / bucket name")
    parser.add_argument(
        "filenames",
        nargs="+",
        help="Optional list of filenames (at least one required)",
    )
    args = parser.parse_args()

    run_data_update(
        args.owner,
        args.repo,
        args.filenames,
        source="bucket" if args.bucket else "dataset",
    )
