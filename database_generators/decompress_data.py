import gzip
import shutil
from pathlib import Path
import argparse

def decompress_file(gz_path: Path):
    """Decompress a .gz file."""
    if not gz_path.exists():
        print(f"Error: {gz_path} not found")
        return False
    
    # Output path without .gz
    output_path = gz_path.with_suffix('')
    
    print(f"Decompressing {gz_path.name}...")
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✓ Created {output_path.name}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Decompress database files')
    parser.add_argument('--database-root', type=Path, required=True,
                       help='Root directory of the database')
    parser.add_argument('--keep-compressed', action='store_true',
                       help='Keep the .gz file after decompression')
    
    args = parser.parse_args()
    
    # Find all .csv.gz files
    gz_files = list(args.database_root.rglob('*.csv.gz'))
    
    if not gz_files:
        print("No compressed CSV files found")
        return
    
    print(f"Found {len(gz_files)} compressed file(s)")
    
    for gz_file in gz_files:
        success = decompress_file(gz_file)
        if success and not args.keep_compressed:
            gz_file.unlink()
            print(f"✓ Removed {gz_file.name}")
    
    print("\n✓ All files decompressed!")

if __name__ == '__main__':
    main()