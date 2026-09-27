"""
Analyze Answer Types Across Datasets
Identifies which datasets have definite answers vs preferred/open-ended answers
"""

import pandas as pd
import json
import os
from collections import Counter
import re

# Dataset configurations
DATASETS = {
    'truthfulqa': {
        'file': 'results/truthfulqa_ablation_results.csv',
        'answer_col': 'best_answer',
        'name': 'TruthfulQA'
    },
    'bbq': {
        'file': 'results/bbq_ablation_results.csv',
        'answer_col': 'correct_answer',
        'name': 'BBQ'
    },
    'mmlu': {
        'file': 'results/mmlu_ablation_results.csv',
        'answer_col': 'correct_answer',
        'name': 'MMLU'
    },
    'crowspairs': {
        'file': 'results/crowspairs_ablation_results.csv',
        'answer_col': 'correct_answer',
        'name': 'CrowS-Pairs'
    },
    'creak': {
        'file': 'results/creak_ablation_results.csv',
        'answer_col': 'correct_label',
        'name': 'CREAK'
    },
    'ciar': {
        'file': 'results/ciar_ablation_results.csv',
        'answer_col': 'correct_answers',
        'name': 'CIAR'
    }
}


def analyze_answer_type(answer_str):
    """Classify answer type"""
    if pd.isna(answer_str) or str(answer_str).strip() == '':
        return 'empty'

    answer = str(answer_str).strip()

    # Check for JSON list
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            return 'json_list', len(parsed)
    except:
        pass

    # Check for single letter (A/B/C/D)
    if len(answer) == 1 and answer.upper() in 'ABCDEFGHIJ':
        return 'single_letter', answer.upper()

    # Check for True/False
    if answer.lower() in ['true', 'false']:
        return 'boolean', answer.lower()

    # Check for number
    if re.match(r'^-?\d+\.?\d*$', answer):
        return 'number', answer

    # Check for short answer (< 20 words)
    word_count = len(answer.split())
    if word_count <= 5:
        return 'short_text', word_count
    elif word_count <= 20:
        return 'medium_text', word_count
    else:
        return 'long_text', word_count


def analyze_dataset(file_path, answer_col, dataset_name):
    """Analyze answer types in a dataset"""
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 60}")

    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    if answer_col not in df.columns:
        print(f"  Column '{answer_col}' not found")
        print(f"  Available columns: {df.columns.tolist()}")
        return None

    print(f"  Total rows: {len(df)}")
    print(f"  Answer column: {answer_col}")

    # Analyze answer types
    type_counts = Counter()
    type_details = {}
    sample_answers = {}

    for idx, row in df.iterrows():
        answer = row[answer_col]
        result = analyze_answer_type(answer)

        if isinstance(result, tuple):
            answer_type, detail = result
        else:
            answer_type = result
            detail = None

        type_counts[answer_type] += 1

        if answer_type not in type_details:
            type_details[answer_type] = []
        if detail is not None:
            type_details[answer_type].append(detail)

        # Store sample
        if answer_type not in sample_answers:
            sample_answers[answer_type] = str(answer)[:100]

    # Print results
    print("\n  Answer Type Distribution:")
    for answer_type, count in type_counts.most_common():
        pct = count / len(df) * 100
        print(f"    {answer_type}: {count} ({pct:.1f}%)")

        # Show details
        if answer_type in type_details and type_details[answer_type]:
            if answer_type == 'single_letter':
                letters = Counter(type_details[answer_type])
                print(f"      Letters: {dict(letters)}")
            elif answer_type in ['short_text', 'medium_text', 'long_text']:
                word_counts = type_details[answer_type]
                avg_words = sum(word_counts) / len(word_counts)
                print(f"      Avg words: {avg_words:.1f}")
            elif answer_type == 'json_list':
                list_lens = type_details[answer_type]
                avg_len = sum(list_lens) / len(list_lens)
                print(f"      Avg list length: {avg_len:.1f}")

        # Show sample
        if answer_type in sample_answers:
            print(f"      Sample: {sample_answers[answer_type]}")

    # Determine answer style
    total = len(df)
    definite_count = (type_counts.get('single_letter', 0) +
                      type_counts.get('boolean', 0) +
                      type_counts.get('number', 0))

    if definite_count / total > 0.8:
        answer_style = 'DEFINITE'
        confidence = 'HIGH'
    elif definite_count / total > 0.5:
        answer_style = 'MIXED'
        confidence = 'MEDIUM'
    else:
        answer_style = 'PREFERRED/OPEN'
        confidence = 'HIGH'

    print(f"\n  *** ANSWER STYLE: {answer_style} (Confidence: {confidence}) ***")
    print(f"  Definite answers: {definite_count}/{total} ({definite_count / total * 100:.1f}%)")

    # Check final answer column
    if 'full_pipeline_final_answer' in df.columns:
        print(f"\n  Analyzing Generated Answers:")
        gen_type_counts = Counter()
        for idx, row in df.head(100).iterrows():  # Sample first 100
            gen_answer = row['full_pipeline_final_answer']
            result = analyze_answer_type(gen_answer)
            if isinstance(result, tuple):
                gen_type, _ = result
            else:
                gen_type = result
            gen_type_counts[gen_type] += 1

        print(f"    Generated answer types (sample):")
        for gen_type, count in gen_type_counts.most_common():
            print(f"      {gen_type}: {count}")

    return {
        'dataset': dataset_name,
        'file': file_path,
        'total_rows': len(df),
        'answer_column': answer_col,
        'type_distribution': dict(type_counts),
        'answer_style': answer_style,
        'definite_ratio': definite_count / total,
        'samples': sample_answers
    }


def generate_recommendations(results):
    """Generate recommendations for fair similarity computation"""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FAIR SIMILARITY COMPUTATION")
    print("=" * 80)

    definite_datasets = []
    preferred_datasets = []

    for result in results:
        if result['answer_style'] == 'DEFINITE':
            definite_datasets.append(result['dataset'])
        else:
            preferred_datasets.append(result['dataset'])

    print("\nDATASET CATEGORIZATION:")
    print(f"\nDefinite Answer Datasets ({len(definite_datasets)}):")
    for ds in definite_datasets:
        print(f"  • {ds}")

    print(f"\nPreferred/Open Answer Datasets ({len(preferred_datasets)}):")
    for ds in preferred_datasets:
        print(f"  • {ds}")

    print("\n" + "-" * 80)
    print("FAIRNESS ISSUES:")
    print("-" * 80)

    if preferred_datasets and definite_datasets:
        print("""
1. INCOMPARABLE SIMILARITY SCORES:
   - Definite answer datasets: Similarity to specific "A", "True", "42"
   - Preferred answer datasets: Similarity to one of many possible good answers

2. BIAS IN RAW SCORES:
   - Definite: Binary match/mismatch → Lower raw similarity expected
   - Preferred: Semantic alignment → Higher raw similarity expected

3. EVALUATION MISMATCH:
   - Definite: Should use exact match (TRUE/FALSE evaluation)
   - Preferred: Should use semantic similarity or multiple reference answers
""")

    print("\n" + "-" * 80)
    print("RECOMMENDED SOLUTIONS:")
    print("-" * 80)
    print("""
OPTION 1: SEPARATE ANALYSIS
  - Analyze definite and preferred datasets separately
  - Don't compare raw similarity scores across groups
  - Use normalized scores only for cross-group comparison

OPTION 2: ADJUSTED SIMILARITY COMPUTATION
  For definite datasets:
    - Use exact string matching first
    - Only compute similarity if exact match fails
    - Weight similarity scores differently

  For preferred datasets:
    - Continue using semantic similarity
    - Consider multiple reference answers if available
    - Use current methodology

OPTION 3: ALTERNATIVE METRICS
  For definite datasets:
    - Primary: Accuracy (exact match)
    - Secondary: Semantic similarity as backup

  For preferred datasets:
    - Primary: Semantic similarity to best answer
    - Secondary: Similarity to multiple reference answers

OPTION 4: STRATIFIED NORMALIZATION
  - Normalize within definite group separately
  - Normalize within preferred group separately
  - Only compare normalized scores across groups
""")

    print("\n" + "-" * 80)
    print("IMPLEMENTATION SUGGESTION:")
    print("-" * 80)
    print("""
Add a 'dataset_type' field to the similarity analysis:
  1. Tag each dataset as 'definite' or 'preferred'
  2. Compute type-specific metrics
  3. Normalize within type groups
  4. Create separate visualizations for each type
  5. Use cross-type normalized scores for combined analysis
""")


def save_analysis(results, output_file='results/dataset_answer_type_analysis.json'):
    """Save analysis results"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze answer types across datasets')
    parser.add_argument('--output', type=str, default='results/dataset_answer_type_analysis.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("=" * 80)
    print("DATASET ANSWER TYPE ANALYSIS")
    print("=" * 80)

    results = []
    for dataset_key, config in DATASETS.items():
        result = analyze_dataset(
            config['file'],
            config['answer_col'],
            config['name']
        )
        if result:
            results.append(result)

    # Generate recommendations
    if results:
        generate_recommendations(results)
        save_analysis(results, args.output)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)