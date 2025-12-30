"""
Example: Plot Size Distribution from CSV

This simple example demonstrates how to use the SizeDistributionPlotter
to generate particle size distribution histograms from a detected shapes CSV file.

Usage:
    python plot_size_distribution.py <csv_path> [--output OUTPUT_DIR]

Example:
    python plot_size_distribution.py results/5_shapes/detected_shapes_original.csv
    python plot_size_distribution.py results/5_shapes/detected_shapes_original.csv --output charts
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nanorange.image_analyzer.plotter.stats_plotter import (
    SizeDistributionPlotter,
    SizeDistributionConfig,
    plot_size_distribution
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot particle size distribution from a detected shapes CSV file"
    )
    parser.add_argument(
        "csv_path", 
        type=str, 
        help="Path to the detected shapes CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory (default: same as CSV file location)"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20)"
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_base = os.path.join(args.output, "size_distribution")
    else:
        csv_dir = os.path.dirname(args.csv_path) or "."
        output_base = os.path.join(csv_dir, "size_distribution")
    
    print("\n" + "=" * 50)
    print("📊 SIZE DISTRIBUTION PLOTTER")
    print("=" * 50)
    print(f"   Input CSV:  {args.csv_path}")
    print(f"   Output:     {output_base}.[png|html]")
    print(f"   Bins:       {args.num_bins}")
    print(f"   IQR mult:   {args.iqr_multiplier}")
    print("=" * 50 + "\n")
    
    # Configure and create plotter
    config = SizeDistributionConfig(
        num_bins=args.num_bins,
        outlier_iqr_multiplier=args.iqr_multiplier
    )
    
    plotter = SizeDistributionPlotter(config)
    
    # Load data and show statistics
    diameters = plotter.load_diameters_from_csv(args.csv_path)
    
    if not diameters:
        print("❌ No valid diameter data found in CSV")
        sys.exit(1)
    
    filtered, lower, upper = plotter.filter_outliers_iqr(diameters)
    n_outliers = len(diameters) - len(filtered)
    
    print(f"📈 Statistics:")
    print(f"   Total particles:    {len(diameters)}")
    print(f"   Diameter range:     {min(diameters):.1f} - {max(diameters):.1f} pixels")
    print(f"   Mean diameter:      {sum(diameters)/len(diameters):.1f} pixels")
    print(f"   Outliers excluded:  {n_outliers}")
    print(f"   Outlier bounds:     {lower:.1f} - {upper:.1f} pixels")
    
    if filtered:
        print(f"\n📈 After filtering:")
        print(f"   Particles:          {len(filtered)}")
        print(f"   Diameter range:     {min(filtered):.1f} - {max(filtered):.1f} pixels")
        print(f"   Mean diameter:      {sum(filtered)/len(filtered):.1f} pixels")
    
    # Generate plot
    print("\n🎨 Generating plots...")
    saved_files = plotter.plot_from_csv(args.csv_path, output_base)
    
    if saved_files:
        print("\n✅ Files saved:")
        for file_type, path in saved_files.items():
            print(f"   {file_type}: {path}")
        print("\n✅ Done!")
    else:
        print("\n❌ Failed to generate plots")
        sys.exit(1)


if __name__ == "__main__":
    main()

