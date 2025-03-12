import polars as pl
import argparse
import os

def parse_sigtk_file(sigtk_file):
    """Parse the sigtk file to extract pa_mean and pa_std."""
    if not sigtk_file or not os.path.exists(sigtk_file):
        return 0, 0  # Default values if file doesn't exist
    with open(sigtk_file, 'r') as f:
        line = f.readline().strip()
        parts = line.split()
        pa_mean = float(parts[2])
        pa_std = float(parts[5])
    return pa_mean, pa_std

def calculate_global_mean(input_file, chunk_size, filter_length):
    """Calculate global mean using streaming lazy execution with projection"""
    dtypes = {
        "start_idx": pl.Int64,
        "end_idx": pl.Int64,
        "event_stdv": pl.Float64,
    }
    
    q = (
        pl.scan_csv(
            input_file,
            separator="\t",
            schema_overrides=dtypes,
        )
        .select(["start_idx", "end_idx", "event_stdv"]) # Project only needed columns
        .filter((pl.col("end_idx") - pl.col("start_idx")) <= filter_length)
        .select(
            total_sum=pl.sum("event_stdv").cast(pl.Float64),
            total_count=pl.len().cast(pl.UInt64),
        )
    )
    result = q.collect(streaming=True)
    return result["total_sum"][0] / result["total_count"][0]

def standardize_and_write_chunks(input_file, output_file, chunk_size, pa_mean, pa_std, filter_length=70):
    dtypes = {
        "start_idx": pl.Int64,
        "end_idx": pl.Int64,
        "event_stdv": pl.Float64,
        "samples": pl.String,
    }
    
    # Base query with filtering and event_stdv normalization
    q = (
        pl.scan_csv(
            input_file,
            separator="\t",
            schema_overrides=dtypes,
        )
        .filter((pl.col("end_idx") - pl.col("start_idx")) <= filter_length)
    )
    
    # Add samples transformation if needed
    if pa_mean != 0 and pa_std != 0:
        q = q.with_columns(
            pl.col("samples")
            .str.split(",")
            .cast(pl.List(pl.Float64))
            .list.eval(pl.element() * pa_std + pa_mean)
            .cast(pl.List(pl.String))
            .list.join(",")
            .alias("samples")
        )

        q = q.with_columns(
            # Compute event_stdv as the standard deviation of denormalized samples
            pl.col("samples")
            .str.split(",")
            .cast(pl.List(pl.Float64))
            .list.std()
            .alias("event_stdv")
        )
    
    # Stream results directly to CSV
    q.sink_csv(
        output_file,
        separator="\t",
        include_header=True,
        batch_size=chunk_size,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standardize the event_stdv column of a large TSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input TSV file')
    parser.add_argument('output_file', type=str, help='Path to the output standardized TSV file')
    parser.add_argument('--sigtk', type=str, help='Path to the sigtk file containing pa_mean and pa_std')
    parser.add_argument('--chunk_size', type=int, default=25000, help='Number of rows per chunk for processing')
    parser.add_argument('--filter_length', type=int, default=70, help='Maximal length of an event. Longer events will be filtered out.')
    args = parser.parse_args()

    # Parse the sigtk file for pa_mean and pa_std
    pa_mean, pa_std = parse_sigtk_file(args.sigtk)

    # Standardize and write the processed data
    standardize_and_write_chunks(
        args.input_file,
        args.output_file,
        args.chunk_size,
        pa_mean,
        pa_std,
        args.filter_length
    )