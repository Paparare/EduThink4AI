"""
Validation Script for Answer Extraction
Checks a random sample of extractions to verify accuracy
"""

import pandas as pd
import random
import sys


def validate_extractions(results_file, sample_size=20):
    """
    Validate extraction accuracy by showing random samples
    """

    print("=" * 80)
    print("ANSWER EXTRACTION VALIDATION")
    print("=" * 80)

    df = pd.read_csv(results_file)

    # Check if extraction columns exist
    has_extractions = 'extracted_answer_background' in df.columns

    if not has_extractions:
        print("\nNo extraction columns found in results.")
        print("This file may not have been processed with answer extraction.")
        return

    # Group by dataset
    datasets = df['dataset'].unique()

    for dataset in datasets:
        ds_df = df[df['dataset'] == dataset]
        dataset_type = ds_df['dataset_type'].iloc[0] if 'dataset_type' in ds_df.columns else 'unknown'

        if dataset_type != 'definite':
            continue  # Skip preferred datasets

        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset} (Type: {dataset_type})")
        print(f"{'=' * 80}")

        # Sample random rows
        sample = ds_df.sample(min(sample_size, len(ds_df)))

        print(f"\nShowing {len(sample)} random samples:")

        for idx, (_, row) in enumerate(sample.iterrows(), 1):
            print(f"\n--- Sample {idx} ---")
            print(f"Question: {row['question'][:100]}...")
            print(f"Correct Answer: {row['best_answer']}")

            print("\nAgent Extractions:")
            for agent in ['background', 'validity', 'critique', 'analysis']:
                extracted = row.get(f'extracted_answer_{agent}')
                exact_match = row.get(f'exact_match_{agent}')

                match_symbol = "✓" if exact_match else "✗" if exact_match is False else "?"

                print(f"  {agent.capitalize():12} -> {str(extracted):10} {match_symbol}")

            # Show one agent output for context
            bg_output = row.get('full_pipeline_educator_output',
                                row.get('G_background', 'N/A'))
            if bg_output != 'N/A':
                print(f"\nBackgrounder Output (first 200 chars):")
                print(f"  {str(bg_output)[:200]}...")

        # Calculate extraction stats
        print(f"\n{'=' * 60}")
        print(f"Extraction Statistics for {dataset}:")
        print(f"{'=' * 60}")

        for agent in ['background', 'validity', 'critique', 'analysis']:
            extracted_col = f'extracted_answer_{agent}'
            exact_match_col = f'exact_match_{agent}'

            if extracted_col in ds_df.columns:
                total = len(ds_df)
                extracted_count = ds_df[extracted_col].notna().sum()
                extraction_rate = extracted_count / total

                if exact_match_col in ds_df.columns:
                    correct_count = ds_df[exact_match_col].sum()
                    accuracy = correct_count / total if total > 0 else 0

                    print(f"\n{agent.capitalize()}:")
                    print(f"  Extraction Rate: {extraction_rate:.2%} ({extracted_count}/{total})")
                    print(f"  Accuracy: {accuracy:.2%} ({correct_count}/{total} correct)")

                    if extracted_count > 0:
                        accuracy_of_extracted = correct_count / extracted_count
                        print(f"  Accuracy (of extracted): {accuracy_of_extracted:.2%}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nReview the samples above to verify:")
    print("  1. Extracted answers match what you see in agent output")
    print("  2. Exact match results are correct")
    print("  3. Extraction rate is high (>90% ideal)")
    print("  4. Accuracy seems reasonable for your agents")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate answer extraction accuracy')
    parser.add_argument('--input', type=str, default='results/semantic_similarity_results.csv',
                        help='Results CSV file with extractions')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to show per dataset')

    args = parser.parse_args()

    validate_extractions(args.input, args.samples)