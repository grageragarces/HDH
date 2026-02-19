"""
Step 1: Reorganize MQTBench Random Circuits
============================================
Moves all random_* pkl files from Database/HDHs/Circuit/MQTBench/pkl/
into a new Database/HDHs/Circuit/MQTBenchRandom/pkl/ folder.

RUN:
    python 1_reorganize_mqtbench_random.py --database-root ../database
    python 1_reorganize_mqtbench_random.py --database-root ../database --dry-run
    python 1_reorganize_mqtbench_random.py --database-root ../database --copy  # keep originals
"""

import argparse
import shutil
from pathlib import Path


def reorganize_mqtbench_random(
    database_root: Path,
    dry_run: bool = False,
    copy_mode: bool = False,
) -> None:
    src_dir = database_root / "HDHs" / "Circuit" / "MQTBench" / "pkl"
    dst_dir = database_root / "HDHs" / "Circuit" / "MQTBenchRandom" / "pkl"

    if not src_dir.exists():
        print(f"✗ Source directory not found: {src_dir}")
        return

    random_files = sorted(src_dir.glob("random_*.pkl"))

    if not random_files:
        print(f"✗ No random_*.pkl files found in {src_dir}")
        return

    print(f"Found {len(random_files)} random_*.pkl file(s) in MQTBench")
    print(f"  Source : {src_dir}")
    print(f"  Dest   : {dst_dir}")
    print(f"  Mode   : {'DRY RUN' if dry_run else ('COPY' if copy_mode else 'MOVE')}")
    print()

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0

    for src_file in random_files:
        dst_file = dst_dir / src_file.name
        status = ""

        if dst_file.exists():
            print(f"  [SKIP]  {src_file.name}  (already exists at destination)")
            skipped += 1
            continue

        if dry_run:
            action = "COPY" if copy_mode else "MOVE"
            print(f"  [DRY {action}]  {src_file.name}")
        else:
            if copy_mode:
                shutil.copy2(src_file, dst_file)
                status = "copied"
            else:
                shutil.move(str(src_file), str(dst_file))
                status = "moved"
            print(f"  [OK]  {src_file.name}  → {dst_file.relative_to(database_root)}  ({status})")
            moved += 1

    print()
    if dry_run:
        print(f"Dry run complete — {len(random_files)} file(s) would be processed, {skipped} skipped.")
    else:
        action_word = "copied" if copy_mode else "moved"
        print(f"Done — {moved} file(s) {action_word}, {skipped} skipped.")
        print(f"New folder: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Move MQTBench random_* pkl files to a dedicated MQTBenchRandom folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--database-root", type=Path, required=True,
                        help="Root directory of the database (contains HDHs/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would happen without making changes")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them (keeps originals in MQTBench)")
    args = parser.parse_args()

    reorganize_mqtbench_random(
        database_root=args.database_root,
        dry_run=args.dry_run,
        copy_mode=args.copy,
    )


if __name__ == "__main__":
    main()
