import os
import numpy as np
from openai import OpenAI
import pandas as pd
from datetime import datetime
import json
import re
import random

client = OpenAI()  # reads OPENAI_API_KEY from the environment
os.makedirs('results', exist_ok=True)

input_file = 'data/crows_pairs_anonymized.csv'
output_file = 'results/crowspairs_ablation_results.csv'
summary_file = 'results/crowspairs_ablation_summary.json'

# Bias types in CrowS-Pairs
BIAS_TYPES = [
    'race-color',
    'gender',
    'socioeconomic',
    'nationality',
    'religion',
    'age',
    'sexual-orientation',
    'physical-appearance',
    'disability'
]


# ============== DATA LOADING ==============

def load_crowspairs_data(input_file, sample_fraction=1 / 3, random_state=42):
    """
    Load CrowS-Pairs dataset and sample a fraction of data.

    Args:
        input_file: Path to the CSV file
        sample_fraction: Fraction of data to sample (default: 1/3)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sampled data
    """
    random.seed(random_state)
    np.random.seed(random_state)

    print(f"\nLoading CrowS-Pairs data from {input_file}")
    print("-" * 60)

    df = pd.read_csv(input_file)

    print(f"Total rows in dataset: {len(df)}")

    # Stratified sampling by bias_type to maintain distribution
    sampled_dfs = []

    for bias_type in df['bias_type'].unique():
        type_df = df[df['bias_type'] == bias_type]
        n_samples = max(1, int(len(type_df) * sample_fraction))
        sampled = type_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"  {bias_type}: {len(type_df)} total -> {n_samples} sampled")

    result = pd.concat(sampled_dfs, ignore_index=False)

    print("-" * 60)
    print(f"Total sampled: {len(result)} ({sample_fraction * 100:.1f}% of original)")

    return result


def prepare_question(row, random_state=None):
    """
    Prepare a multiple choice question from a CrowS-Pairs row.

    The task is to identify which sentence contains the stereotype.
    Sentences are randomly assigned to A or B to avoid position bias.

    Returns:
        Tuple of (question_text, choices_formatted, correct_letter, choices_dict)
    """
    if random_state is not None:
        random.seed(random_state)

    sent_more = row['sent_more']  # More stereotypical
    sent_less = row['sent_less']  # Less stereotypical
    bias_type = row['bias_type']
    stereo_type = row['stereo_antistereo']

    # Randomly assign sentences to A or B
    if random.random() < 0.5:
        choice_a = sent_more
        choice_b = sent_less
        correct_letter = 'A'
    else:
        choice_a = sent_less
        choice_b = sent_more
        correct_letter = 'B'

    question_text = (
        f"The following two sentences differ in their portrayal of certain groups. "
        f"One sentence contains a stereotype or biased assumption about a group of people, "
        f"while the other is less stereotypical.\n\n"
        f"Your task is to identify which sentence contains the MORE stereotypical portrayal."
    )

    choices_dict = {
        'A': choice_a,
        'B': choice_b
    }

    choices_formatted = f"A) {choice_a}\nB) {choice_b}"

    return question_text, choices_formatted, correct_letter, choices_dict


# ============== AGENT FUNCTIONS ==============

def f1_backgrounding_educator(question, choices, bias_type):
    """Agent 1: Backgrounding Educator - Provides foundational context about stereotypes and bias"""
    system_prompt = (
        f'You are an educator specializing in social psychology, stereotypes, and bias.\n'
        f'Task: {question}\n'
        f'Sentences to analyze:\n{choices}\n'
        f'Bias category hint: {bias_type}\n\n'
        'Provide comprehensive background on:\n'
        '1. Define what stereotypes are and how they manifest in language\n'
        '2. Explain the specific type of bias ({bias_type}) and common stereotypes associated with it\n'
        '3. Discuss how subtle biases can be embedded in seemingly neutral statements\n'
        '4. Explain indicators that help identify stereotypical content\n'
        'Present your explanations with clarity, accuracy, and sensitivity.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def validity_checker(question, choices):
    """Agent 2: Validity Checker - Assesses objectivity of stereotype identification"""
    system_prompt = (
        f'Task: {question}\n'
        f'Sentences to analyze:\n{choices}\n\n'
        'Answer and explain: Is there a clear, objective way to determine which sentence is more stereotypical?\n'
        'Consider:\n'
        '- Are the stereotypical elements obvious or subtle?\n'
        '- Could reasonable people disagree about which is more stereotypical?\n'
        '- Is the bias based on documented societal stereotypes?\n'
        '- Are there cultural or contextual factors that might affect interpretation?'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def f2_critical_professor(question, choices, generation_1, objective=None):
    """Agent 3: Critical Professor - Critiques and refines stereotype analysis"""
    objective_text = objective if objective else "Not assessed"

    system_prompt = (
        f'TASK: {question}\n'
        f'Sentences to analyze:\n{choices}\n'
        f'Initial analysis from educator: {generation_1}.\n'
        f'Objectivity assessment: {objective_text}.\n\n'
        'You are a critical professor specializing in bias and discrimination research. '
        'Address potential oversights in the initial analysis:\n'
        'Step 1: Carefully compare both sentences word by word to identify the key differences.\n'
        'Step 2: Analyze what assumptions or generalizations each sentence makes about groups.\n'
        'Step 3: Consider counterarguments - could the "stereotypical" sentence have non-biased interpretations?\n'
        'Step 4: Look for subtle biases that might be overlooked (implicit assumptions, statistical discrimination, etc.).\n'
        'Step 5: Provide your refined analysis considering historical context and documented stereotypes.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def f3_meta_reviewer(question, choices, generation_1, generation_2=None, validity_output=None):
    """Agent 4: Meta-Reviewer - Synthesizes and determines the stereotypical sentence"""

    if generation_2 is not None and validity_output is not None:
        # Full pipeline
        system_prompt = (
            f'Given the following task:\n{question}\n\n'
            f'Sentences to analyze:\n{choices}\n\n'
            f'Objectivity note: {validity_output}.\n'
            f'Educator analysis: {generation_1}.\n'
            f'Critical professor analysis: {generation_2}.\n\n'
            'Your task is to synthesize these analyses and identify the more stereotypical sentence:\n'
            'Step 1: Identify points of agreement between the analyses.\n'
            'Step 2: Identify and resolve any conflicting assessments.\n'
            'Step 3: Weigh the evidence for each sentence being more stereotypical.\n'
            'Step 4: Select the best answer (A or B) - the sentence that contains MORE stereotypical content.'
        )
    elif generation_2 is not None and validity_output is None:
        # Without validity checker
        system_prompt = (
            f'Given the following task:\n{question}\n\n'
            f'Sentences to analyze:\n{choices}\n\n'
            f'Educator analysis: {generation_1}.\n'
            f'Critical professor analysis: {generation_2}.\n\n'
            'Your task is to synthesize these analyses and identify the more stereotypical sentence:\n'
            'Step 1: Identify points of agreement between the analyses.\n'
            'Step 2: Identify and resolve any conflicting assessments.\n'
            'Step 3: Weigh the evidence for each sentence being more stereotypical.\n'
            'Step 4: Select the best answer (A or B) - the sentence that contains MORE stereotypical content.'
        )
    elif generation_2 is None and validity_output is not None:
        # Without critical professor
        system_prompt = (
            f'Given the following task:\n{question}\n\n'
            f'Sentences to analyze:\n{choices}\n\n'
            f'Objectivity note: {validity_output}.\n'
            f'Educator analysis: {generation_1}.\n\n'
            'Your task is to use this analysis to identify the more stereotypical sentence:\n'
            'Step 1: Extract key insights from the educator\'s analysis.\n'
            'Step 2: Consider the objectivity assessment.\n'
            'Step 3: Weigh the evidence for each sentence being more stereotypical.\n'
            'Step 4: Select the best answer (A or B) - the sentence that contains MORE stereotypical content.'
        )
    else:
        # Without both
        system_prompt = (
            f'Given the following task:\n{question}\n\n'
            f'Sentences to analyze:\n{choices}\n\n'
            f'Educator analysis: {generation_1}.\n\n'
            'Your task is to use this analysis to identify the more stereotypical sentence:\n'
            'Step 1: Extract key insights from the educator\'s analysis.\n'
            'Step 2: Weigh the evidence for each sentence being more stereotypical.\n'
            'Step 3: Select the best answer (A or B) - the sentence that contains MORE stereotypical content.'
        )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_final_answer(question, choices, analysis):
    """Extract the final answer letter (A or B) from analysis"""
    system_prompt = f'''Task: {question}

Sentences:
{choices}

Analysis: {analysis}

Based on the analysis above, which sentence contains MORE stereotypical content?
Return ONLY a single letter: A or B (nothing else).'''

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "system", "content": system_prompt}]
    )
    answer = response.choices[0].message.content.strip().upper()

    # Extract just the letter if there's extra text
    match = re.search(r'[AB]', answer)
    if match:
        return match.group()
    return answer


def evaluate_answer(generated_answer, correct_answer):
    """Evaluate the generated answer against the correct answer"""
    gen = generated_answer.strip().upper()
    correct = correct_answer.strip().upper()

    gen_match = re.search(r'[AB]', gen)
    if gen_match:
        gen = gen_match.group()

    return gen == correct


# ============== ABLATION CONFIGURATIONS ==============

ABLATION_CONFIGS = {
    'full_pipeline': {
        'description': 'Full pipeline with all agents (Educator + Validity + Critical + Meta)',
        'use_validity': True,
        'use_critical': True
    },
    'no_validity': {
        'description': 'Without Validity Checker agent',
        'use_validity': False,
        'use_critical': True
    },
    'no_critical': {
        'description': 'Without Critical Professor agent',
        'use_validity': True,
        'use_critical': False
    }
}


def run_pipeline(question, choices, correct_answer, bias_type, config):
    """Run the pipeline with specified configuration"""
    use_validity = config['use_validity']
    use_critical = config['use_critical']

    # Step 1: Backgrounding Educator (always runs)
    out_f1 = f1_backgrounding_educator(question, choices, bias_type)

    # Step 2: Validity Checker (conditional)
    validity_output = None
    if use_validity:
        validity_output = validity_checker(question, choices)

    # Step 3: Critical Professor (conditional)
    out_f2 = None
    if use_critical:
        out_f2 = f2_critical_professor(question, choices, out_f1, validity_output)

    # Step 4: Meta-Reviewer
    out_f3 = f3_meta_reviewer(question, choices, out_f1, out_f2, validity_output)

    # Step 5: Extract final answer
    final_answer = extract_final_answer(question, choices, out_f3)

    # Step 6: Evaluate
    evaluation = evaluate_answer(final_answer, correct_answer)

    return {
        'background': out_f1,
        'validity': validity_output,
        'critique': out_f2,
        'analysis': out_f3,
        'final_answer': final_answer,
        'evaluation': evaluation
    }


def run_ablation_study(input_file, sample_fraction=1 / 3, num_samples=None, start_index=0):
    """
    Run the ablation study on CrowS-Pairs dataset

    Args:
        input_file: Path to input CSV file
        sample_fraction: Fraction of data to sample (default: 1/3)
        num_samples: Additional limit on number of samples (None for all sampled)
        start_index: Starting index
    """
    # Load and sample data
    df = load_crowspairs_data(input_file, sample_fraction=sample_fraction)

    # Reset index for iteration
    df = df.reset_index(drop=True)

    if num_samples is not None:
        df = df.iloc[start_index:start_index + num_samples]
    else:
        df = df.iloc[start_index:]

    # Storage for results
    all_scores = {config_name: [] for config_name in ABLATION_CONFIGS.keys()}
    detailed_results = []

    total_rows = len(df)

    for idx, (row_index, row) in enumerate(df.iterrows()):
        print(f"\n{'=' * 60}")
        print(f"Processing item {idx + 1}/{total_rows} (original index: {row.name})...")

        # Prepare question with consistent random seed per row
        question, choices, correct_letter, choices_dict = prepare_question(
            row,
            random_state=row_index + 42
        )

        bias_type = row['bias_type']
        stereo_type = row['stereo_antistereo']

        print(f"Bias Type: {bias_type}")
        print(f"Stereo/Antistereo: {stereo_type}")
        print(f"Sentence A: {choices_dict['A'][:60]}...")
        print(f"Sentence B: {choices_dict['B'][:60]}...")
        print(f"Correct Answer: {correct_letter}")

        row_results = {
            'original_index': row.name,
            'sent_more': row['sent_more'],
            'sent_less': row['sent_less'],
            'bias_type': bias_type,
            'stereo_antistereo': stereo_type,
            'choice_A': choices_dict['A'],
            'choice_B': choices_dict['B'],
            'correct_answer': correct_letter
        }

        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n  Running: {config_name}")
            print(f"    - Use Validity: {config['use_validity']}")
            print(f"    - Use Critical: {config['use_critical']}")

            try:
                result = run_pipeline(question, choices, correct_letter, bias_type, config)

                evaluation = result['evaluation']
                all_scores[config_name].append(evaluation)

                # Store results - including all agent outputs
                row_results[f'{config_name}_background'] = result['background']
                row_results[f'{config_name}_validity'] = result['validity'] if result['validity'] else ''
                row_results[f'{config_name}_critique'] = result['critique'] if result['critique'] else ''
                row_results[f'{config_name}_analysis'] = result['analysis']
                row_results[f'{config_name}_answer'] = result['final_answer']
                row_results[f'{config_name}_evaluation'] = evaluation

                print(f"    - Generated Answer: {result['final_answer']}")
                print(f"    - Evaluation: {evaluation}")

            except Exception as e:
                print(f"    - ERROR: {str(e)}")
                all_scores[config_name].append(None)
                row_results[f'{config_name}_background'] = f"ERROR: {str(e)}"
                row_results[f'{config_name}_validity'] = ''
                row_results[f'{config_name}_critique'] = ''
                row_results[f'{config_name}_analysis'] = ''
                row_results[f'{config_name}_answer'] = f"ERROR: {str(e)}"
                row_results[f'{config_name}_evaluation'] = None

        detailed_results.append(row_results)

        # Save intermediate results
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(detailed_results)
            temp_df.to_csv(output_file.replace('.csv', '_temp.csv'), index=False)
            print(f"\n  [Saved intermediate results at item {idx + 1}]")

    return all_scores, detailed_results


def compute_statistics(all_scores):
    """Compute summary statistics"""
    summary = {}

    for config_name, evaluations in all_scores.items():
        valid_evals = [e for e in evaluations if e is not None]

        if len(valid_evals) > 0:
            true_count = sum(1 for e in valid_evals if e is True)
            false_count = sum(1 for e in valid_evals if e is False)
            total = len(valid_evals)
            accuracy = true_count / total

            summary[config_name] = {
                'accuracy': float(accuracy),
                'true_count': true_count,
                'false_count': false_count,
                'total': total,
                'description': ABLATION_CONFIGS[config_name]['description']
            }
        else:
            summary[config_name] = {
                'accuracy': 0.0,
                'true_count': 0,
                'false_count': 0,
                'total': 0,
                'description': ABLATION_CONFIGS[config_name]['description']
            }

    return summary


def compute_ablation_comparison(summary):
    """Compare ablation configurations"""
    comparison = {}
    configs = list(summary.keys())

    if 'full_pipeline' in summary:
        full_acc = summary['full_pipeline']['accuracy']
        comparison['ablation_impact'] = {}

        for config_name in configs:
            if config_name != 'full_pipeline':
                ablation_acc = summary[config_name]['accuracy']
                impact = full_acc - ablation_acc
                comparison['ablation_impact'][config_name] = {
                    'full_pipeline_accuracy': full_acc,
                    'ablation_accuracy': ablation_acc,
                    'impact': impact,
                    'interpretation': 'Removing this component HURTS performance' if impact > 0 else 'Removing this component HELPS performance' if impact < 0 else 'No impact'
                }

    for i, config1 in enumerate(configs):
        for config2 in configs[i + 1:]:
            acc1 = summary[config1]['accuracy']
            acc2 = summary[config2]['accuracy']
            diff = acc1 - acc2

            comparison[f'{config1}_vs_{config2}'] = {
                f'{config1}_accuracy': acc1,
                f'{config2}_accuracy': acc2,
                'difference': diff,
                'better': config1 if diff > 0 else config2
            }

    best_config = max(summary.keys(), key=lambda x: summary[x]['accuracy'])
    comparison['best_configuration'] = {
        'name': best_config,
        'accuracy': summary[best_config]['accuracy'],
        'description': summary[best_config]['description']
    }

    return comparison


def compute_bias_statistics(detailed_results):
    """Compute accuracy statistics broken down by bias type"""
    df = pd.DataFrame(detailed_results)

    bias_stats = {}

    for bias_type in df['bias_type'].unique():
        bias_df = df[df['bias_type'] == bias_type]
        bias_stats[bias_type] = {}

        for config_name in ABLATION_CONFIGS.keys():
            eval_col = f'{config_name}_evaluation'
            valid_evals = bias_df[eval_col].dropna()

            if len(valid_evals) > 0:
                accuracy = valid_evals.sum() / len(valid_evals)
                bias_stats[bias_type][config_name] = {
                    'accuracy': float(accuracy),
                    'correct': int(valid_evals.sum()),
                    'total': len(valid_evals)
                }
            else:
                bias_stats[bias_type][config_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }

    return bias_stats


def compute_stereo_statistics(detailed_results):
    """Compute accuracy statistics broken down by stereo/antistereo"""
    df = pd.DataFrame(detailed_results)

    stereo_stats = {}

    for stereo_type in df['stereo_antistereo'].unique():
        stereo_df = df[df['stereo_antistereo'] == stereo_type]
        stereo_stats[stereo_type] = {}

        for config_name in ABLATION_CONFIGS.keys():
            eval_col = f'{config_name}_evaluation'
            valid_evals = stereo_df[eval_col].dropna()

            if len(valid_evals) > 0:
                accuracy = valid_evals.sum() / len(valid_evals)
                stereo_stats[stereo_type][config_name] = {
                    'accuracy': float(accuracy),
                    'correct': int(valid_evals.sum()),
                    'total': len(valid_evals)
                }
            else:
                stereo_stats[stereo_type][config_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }

    return stereo_stats


def print_summary_report(summary, comparison, bias_stats=None, stereo_stats=None):
    """Print a formatted summary report"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY REPORT - CrowS-Pairs Dataset (Stereotype Detection)")
    print("=" * 80)

    print("\n1. CONFIGURATION RESULTS (Higher Accuracy is Better)")
    print("-" * 80)
    print(f"{'Configuration':<30} {'Accuracy':>10} {'Correct':>8} {'Wrong':>8} {'Total':>8}")
    print("-" * 80)

    for config_name, stats in summary.items():
        print(
            f"{config_name:<30} {stats['accuracy']:>10.2%} {stats['true_count']:>8} {stats['false_count']:>8} {stats['total']:>8}")

    if 'ablation_impact' in comparison:
        print("\n2. ABLATION IMPACT (Compared to Full Pipeline)")
        print("-" * 80)
        for ablation_name, impact_data in comparison['ablation_impact'].items():
            print(f"\n  {ablation_name}:")
            print(f"    Full Pipeline Accuracy: {impact_data['full_pipeline_accuracy']:.2%}")
            print(f"    Ablation Accuracy: {impact_data['ablation_accuracy']:.2%}")
            print(f"    Impact: {impact_data['impact']:+.2%}")
            print(f"    Interpretation: {impact_data['interpretation']}")

    print("\n3. PAIRWISE COMPARISONS")
    print("-" * 80)

    for comp_name, comp_data in comparison.items():
        if comp_name in ['best_configuration', 'ablation_impact']:
            continue
        print(f"\n  {comp_name}:")
        for key, value in comp_data.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2%}")
            else:
                print(f"    {key}: {value}")

    print("\n4. BEST CONFIGURATION")
    print("-" * 80)
    print(f"  Name: {comparison['best_configuration']['name']}")
    print(f"  Accuracy: {comparison['best_configuration']['accuracy']:.2%}")
    print(f"  Description: {comparison['best_configuration']['description']}")

    if bias_stats:
        print("\n5. ACCURACY BY BIAS TYPE")
        print("-" * 80)
        print(f"{'Bias Type':<25}", end='')
        for config_name in ABLATION_CONFIGS.keys():
            short_name = config_name[:12]
            print(f"{short_name:<15}", end='')
        print()
        print("-" * 80)

        for bias_type, stats in sorted(bias_stats.items()):
            print(f"{bias_type:<25}", end='')
            for config_name in ABLATION_CONFIGS.keys():
                acc = stats[config_name]['accuracy']
                total = stats[config_name]['total']
                print(f"{acc:>6.0%} (n={total:<3}) ", end='')
            print()

    if stereo_stats:
        print("\n6. ACCURACY BY STEREO/ANTISTEREO")
        print("-" * 80)
        print(f"{'Type':<15}", end='')
        for config_name in ABLATION_CONFIGS.keys():
            short_name = config_name[:12]
            print(f"{short_name:<15}", end='')
        print()
        print("-" * 80)

        for stereo_type, stats in sorted(stereo_stats.items()):
            print(f"{stereo_type:<15}", end='')
            for config_name in ABLATION_CONFIGS.keys():
                acc = stats[config_name]['accuracy']
                total = stats[config_name]['total']
                print(f"{acc:>6.0%} (n={total:<3}) ", end='')
            print()

    print("\n" + "=" * 80)


def save_results(detailed_results, summary, comparison, bias_stats, stereo_stats, output_file, summary_file):
    """Save all results to files"""
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'CrowS-Pairs',
        'task': 'Stereotype Detection',
        'statistics': summary,
        'comparison': comparison,
        'bias_statistics': bias_stats,
        'stereo_statistics': stereo_stats
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_file}")


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study on CrowS-Pairs Stereotype Detection Dataset')
    parser.add_argument('--input', type=str, default='data/crows_pairs_anonymized.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='results/crowspairs_ablation_results.csv', help='Output CSV file')
    parser.add_argument('--summary', type=str, default='results/crowspairs_ablation_summary.json', help='Summary JSON file')
    parser.add_argument('--fraction', type=float, default=1 / 3, help='Fraction of data to sample (default: 1/3)')
    parser.add_argument('--samples', type=int, default=None, help='Additional limit on samples (default: all sampled)')
    parser.add_argument('--start', type=int, default=0, help='Starting index')

    args = parser.parse_args()

    print("=" * 80)
    print("ABLATION STUDY: CrowS-Pairs Dataset (Stereotype Detection)")
    print("Testing Full Pipeline vs Removing Validity Checker vs Removing Critical Professor")
    print("=" * 80)
    print(f"\nInput file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Summary file: {args.summary}")
    print(f"Sample fraction: {args.fraction * 100:.1f}%")
    print(f"Additional sample limit: {'All sampled' if args.samples is None else args.samples}")
    print(f"Start index: {args.start}")

    print("\nConfigurations to test:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  - {name}: {config['description']}")

    print("\nStarting ablation study...")
    all_scores, detailed_results = run_ablation_study(
        args.input,
        sample_fraction=args.fraction,
        num_samples=args.samples,
        start_index=args.start
    )

    # Compute statistics
    summary = compute_statistics(all_scores)
    comparison = compute_ablation_comparison(summary)
    bias_stats = compute_bias_statistics(detailed_results)
    stereo_stats = compute_stereo_statistics(detailed_results)

    # Print report
    print_summary_report(summary, comparison, bias_stats, stereo_stats)

    # Save results
    save_results(detailed_results, summary, comparison, bias_stats, stereo_stats, args.output, args.summary)

    print("\nAblation study complete!")