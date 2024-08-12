import polars as pl
import argparse


def calculate_global_mean(input_file, chunk_size):
    total_sum = 0
    total_count = 0

    reader = pl.read_csv_batched(
        input_file,
        separator="\t",
        batch_size=chunk_size,
    )

    batches = reader.next_batches(100)
    counter = 0
    while batches:
        df_chunk = pl.concat(batches)
        df_chunk = df_chunk.with_columns(pl.col("event_stdv").cast(pl.Float64))
        # Filter rows where event_stdv is less than or equal to 10.0
        df_chunk = df_chunk.filter(pl.col("event_stdv") <= 10.0)

        total_sum += df_chunk["event_stdv"].sum()
        total_count += df_chunk.shape[0]

        counter += 1
        batches = reader.next_batches(100)

    return total_sum / total_count


def standardize_and_write_chunks(input_file, output_file, global_mean, chunk_size):
    reader = pl.read_csv_batched(
        input_file,
        separator="\t",
        batch_size=chunk_size,
    )

    with open(output_file, "w") as f:
        header_written = False

        batches = reader.next_batches(100)
        while batches:
            df_chunk = pl.concat(batches)
            df_chunk = df_chunk.with_columns(pl.col("event_stdv").cast(pl.Float64))
            # Filter rows where event_stdv is less than or equal to 10.0
            df_chunk = df_chunk.filter(pl.col("event_stdv") <= 10.0)

            # Standardize the event_stdv column using the global mean
            df_chunk = df_chunk.with_columns(
                (pl.col("event_stdv") / global_mean).alias("event_stdv")
            )

            # Write the chunk to the output file
            if not header_written:
                df_chunk.write_csv(f, separator="\t", has_header=True)
                header_written = True
            else:
                df_chunk.write_csv(f, separator="\t", has_header=False)

            batches = reader.next_batches(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize the event_stdv column of a large TSV file."
    )
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument(
        "output_file", type=str, help="Path to the output standardized TSV file"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000000,
        help="Number of rows per chunk for processing",
    )

    args = parser.parse_args()

    # First pass: Calculate the global mean of the event_stdv column
    # global_mean = calculate_global_mean(args.input_file)
    global_mean = calculate_global_mean(args.input_file, args.chunk_size)

    # Second pass: Standardize the event_stdv column using the global mean
    standardize_and_write_chunks(
        args.input_file, args.output_file, global_mean, args.chunk_size
    )
