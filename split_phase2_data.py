"""Create a persistent, seeded SmolTalk train/validation split."""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=108)
    args = parser.parse_args()

    source = pq.ParquetFile(args.source)
    num_rows = source.metadata.num_rows
    if not 0 < args.validation_rows < num_rows:
        parser.error("--validation-rows must be between 1 and the source row count")

    validation_indices = sorted(
        random.Random(args.seed).sample(range(num_rows), args.validation_rows)
    )
    validation_set = set(validation_indices)
    train_path = args.output_dir / "smol-smoltalk" / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    manifest_path = args.output_dir / "split_manifest.json"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temporary_train = train_path.with_suffix(".parquet.tmp")
    temporary_validation = validation_path.with_suffix(".parquet.tmp")
    train_writer = pq.ParquetWriter(temporary_train, source.schema_arrow, compression="zstd")
    validation_writer = pq.ParquetWriter(temporary_validation, source.schema_arrow, compression="zstd")
    offset = 0
    train_rows = validation_rows = 0
    try:
        for batch in source.iter_batches(batch_size=8192):
            table = pa.Table.from_batches([batch])
            is_validation = pa.array(
                [offset + index in validation_set for index in range(batch.num_rows)]
            )
            validation_table = table.filter(is_validation)
            train_table = table.filter(pc.invert(is_validation))
            if train_table.num_rows:
                train_writer.write_table(train_table)
                train_rows += train_table.num_rows
            if validation_table.num_rows:
                validation_writer.write_table(validation_table)
                validation_rows += validation_table.num_rows
            offset += batch.num_rows
    finally:
        train_writer.close()
        validation_writer.close()

    temporary_train.replace(train_path)
    temporary_validation.replace(validation_path)
    manifest = {
        "source": str(args.source.resolve()),
        "source_rows": num_rows,
        "seed": args.seed,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "validation_indices": validation_indices,
        "train_file": str(train_path.resolve()),
        "validation_file": str(validation_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "validation_indices"}, indent=2))


if __name__ == "__main__":
    main()
