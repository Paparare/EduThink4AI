import os
import numpy as np
from openai import OpenAI
import pandas as pd
from datetime import datetime
import json
import random
from datasets import load_dataset

client = OpenAI()  # reads OPENAI_API_KEY from the environment
os.makedirs('results', exist_ok=True)

output_file = 'results/creak_ablation_results.csv'
summary_file = 'results/creak_ablation_summary.json'


# ============== DATA LOADING ==============

def load_creak_data(sample_fraction=0.1, random_state=42):
    """
    Load CREAK dataset from HuggingFace and sample a fraction of the data.

    Args:
        sample_fraction: Fraction of data to sample (default: 0.1 = 10%)
        random_state: Random seed for reproducibility

    Returns:
        List of sampled data items
    """
    random.seed(random_state)
    np.random.seed(random_state)

    print(f"\nLoading CREAK dataset from HuggingFace...")
    print("-" * 60)

    ds = load_dataset("amydeng2000/CREAK")

    all_data = []

    for split in ds:
        split_data = ds[split]
        n_total = len(split_data)
        n_samples = max(1, int(n_total * sample_fraction))

        # Random sampling
        indices = random.sample(range(n_total), n_samples)

        for idx in indices:
            item = split_data[idx]
            item['split'] = split
            all_data.append(item)

        print(f"  {split}: {n_total} total -> {n_samples} sampled ({sample_fraction * 100:.0f}%)")

    print("-" * 60)
    print(f"Total samples: {len(all_data)}")

    return all_data


# ============== AGENT FUNCTIONS ==============

def f1_backgrounding_educator(sentence, entity=None):
    """Agent 1: Backgrounding Educator - Provides foundational context for CREAK claims"""
    entity_context = f"Entity: {entity}\n" if entity else ""

    system_prompt = (
        'You are an expert educator specializing in commonsense reasoning and entity knowledge.\n\n'
        '=== TASK: CREAK COMMONSENSE REASONING ===\n'
        'This claim is from the CREAK dataset, which tests commonsense reasoning about entity knowledge.\n'
        'The task is to determine whether the claim is TRUE or FALSE.\n\n'
        f'{entity_context}'
        f'CLAIM: "{sentence}"\n\n'
        'Provide comprehensive background to assess this claim:\n\n'
        '1. ENTITY KNOWLEDGE:\n'
        '   - What do we know about the entity/entities mentioned?\n'
        '   - What are the key facts about this entity?\n'
        '   - What properties, characteristics, or capabilities does it have?\n\n'
        '2. COMMONSENSE ANALYSIS:\n'
        '   - What commonsense knowledge is relevant here?\n'
        '   - What logical inferences can we make?\n'
        '   - Are there implicit assumptions in the claim?\n\n'
        '3. CLAIM EVALUATION:\n'
        '   - Is this claim factually accurate based on entity knowledge?\n'
        '   - Does the claim make logical sense?\n'
        '   - Are there any contradictions or impossibilities?\n\n'
        '4. PRELIMINARY ASSESSMENT:\n'
        '   - Based on the analysis, is this claim likely TRUE or FALSE?\n'
        '   - What evidence supports this assessment?\n\n'
        'Present your analysis clearly and thoroughly.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def validity_checker(sentence, entity=None):
    """Agent 2: Validity Checker - Assesses objectivity and verifiability of the claim"""
    entity_context = f"Entity: {entity}\n" if entity else ""

    system_prompt = (
        'You are an expert in evaluating claim validity and verifiability.\n\n'
        '=== TASK: ASSESS CLAIM OBJECTIVITY ===\n'
        'This claim is from the CREAK dataset for commonsense reasoning.\n\n'
        f'{entity_context}'
        f'CLAIM: "{sentence}"\n\n'
        'Assess the following aspects:\n\n'
        '1. VERIFIABILITY:\n'
        '   - Can this claim be objectively verified as true or false?\n'
        '   - Is the answer based on established facts or requires speculation?\n'
        '   - What type of knowledge is needed (factual, commonsense, or both)?\n\n'
        '2. REASONING TYPE:\n'
        '   - Does this require pure factual retrieval?\n'
        '   - Does this require commonsense reasoning?\n'
        '   - Is this a mix of both?\n\n'
        '3. POTENTIAL AMBIGUITIES:\n'
        '   - Are there multiple valid interpretations?\n'
        '   - Could the truth value depend on context or perspective?\n'
        '   - Are there edge cases to consider?\n\n'
        '4. CONFIDENCE LEVEL:\n'
        '   - Rate how objectively answerable this is: High / Medium / Low\n'
        '   - Explain your rating\n\n'
        'Your assessment will help calibrate confidence in the final answer.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def f2_critical_professor(sentence, generation_1, objective=None, entity=None):
    """Agent 3: Critical Professor - Critiques and refines the analysis"""
    objective_text = objective if objective else "Not assessed"
    entity_context = f"Entity: {entity}\n" if entity else ""

    system_prompt = (
        'You are a critical professor with expertise in commonsense reasoning and fact verification.\n\n'
        '=== TASK: CRITICAL REVIEW OF CREAK CLAIM ANALYSIS ===\n'
        f'{entity_context}'
        f'CLAIM: "{sentence}"\n\n'
        f'OBJECTIVITY ASSESSMENT:\n{objective_text}\n\n'
        f'INITIAL EDUCATOR ANALYSIS:\n{generation_1}\n\n'
        'Provide a rigorous critical review:\n\n'
        'STEP 1 - FACT VERIFICATION:\n'
        '   - Are the facts stated in the initial analysis accurate?\n'
        '   - Are there any factual errors or misconceptions?\n'
        '   - What additional facts should be considered?\n\n'
        'STEP 2 - LOGICAL REASONING CHECK:\n'
        '   - Is the reasoning in the initial analysis sound?\n'
        '   - Are there logical fallacies or gaps?\n'
        '   - What alternative reasoning paths exist?\n\n'
        'STEP 3 - COUNTERARGUMENTS:\n'
        '   - What arguments could be made for the opposite conclusion?\n'
        '   - Are there edge cases that could change the answer?\n'
        '   - What assumptions might be incorrect?\n\n'
        'STEP 4 - COMMONSENSE VALIDATION:\n'
        '   - Does the conclusion align with commonsense knowledge?\n'
        '   - Are there implicit commonsense assumptions being made?\n'
        '   - What commonsense inferences are most reliable?\n\n'
        'STEP 5 - REFINED CONCLUSION:\n'
        '   - Based on your critical analysis, is the claim TRUE or FALSE?\n'
        '   - Provide clear reasoning with specific evidence\n\n'
        'Be thorough, precise, and evidence-based in your critique.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def f3_meta_reviewer(sentence, generation_1, generation_2=None, validity_output=None, entity=None):
    """Agent 4: Meta-Reviewer - Synthesizes analyses and determines final answer"""
    entity_context = f"Entity: {entity}\n" if entity else ""

    base_context = (
        'You are a meta-analyst responsible for synthesizing expert analyses '
        'to determine whether a claim is TRUE or FALSE.\n\n'
        '=== TASK: FINAL TRUTH DETERMINATION FOR CREAK CLAIM ===\n'
        'This is a claim from the CREAK benchmark testing commonsense reasoning about entities.\n\n'
        f'{entity_context}'
        f'CLAIM: "{sentence}"\n\n'
    )

    if generation_2 is not None and validity_output is not None:
        # Full pipeline
        system_prompt = (
                base_context +
                f'OBJECTIVITY ASSESSMENT:\n{validity_output}\n\n'
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                f'CRITICAL PROFESSOR ANALYSIS:\n{generation_2}\n\n'
                'Synthesize these analyses to determine if the claim is TRUE or FALSE:\n\n'
                'STEP 1 - CONSENSUS IDENTIFICATION:\n'
                '   - Where do both analyses agree?\n'
                '   - What facts/reasoning are consistently supported?\n\n'
                'STEP 2 - CONFLICT RESOLUTION:\n'
                '   - Where do the analyses disagree?\n'
                '   - Which position has stronger evidence?\n\n'
                'STEP 3 - EVIDENCE SYNTHESIS:\n'
                '   - What is the strongest evidence for TRUE?\n'
                '   - What is the strongest evidence for FALSE?\n\n'
                'STEP 4 - FINAL DETERMINATION:\n'
                '   - Based on entity knowledge and commonsense reasoning\n'
                '   - Is this claim TRUE or FALSE?\n'
                '   - Provide clear justification\n\n'
                'Conclude with your final answer: TRUE or FALSE.'
        )
    elif generation_2 is not None and validity_output is None:
        # Without validity checker
        system_prompt = (
                base_context +
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                f'CRITICAL PROFESSOR ANALYSIS:\n{generation_2}\n\n'
                'Synthesize these analyses to determine if the claim is TRUE or FALSE:\n\n'
                'STEP 1 - CONSENSUS IDENTIFICATION:\n'
                '   - Where do both analyses agree?\n'
                '   - What facts/reasoning are consistently supported?\n\n'
                'STEP 2 - CONFLICT RESOLUTION:\n'
                '   - Where do the analyses disagree?\n'
                '   - Which position has stronger evidence?\n\n'
                'STEP 3 - EVIDENCE SYNTHESIS:\n'
                '   - What is the strongest evidence for TRUE?\n'
                '   - What is the strongest evidence for FALSE?\n\n'
                'STEP 4 - FINAL DETERMINATION:\n'
                '   - Based on entity knowledge and commonsense reasoning\n'
                '   - Is this claim TRUE or FALSE?\n'
                '   - Provide clear justification\n\n'
                'Conclude with your final answer: TRUE or FALSE.'
        )
    elif generation_2 is None and validity_output is not None:
        # Without critical professor
        system_prompt = (
                base_context +
                f'OBJECTIVITY ASSESSMENT:\n{validity_output}\n\n'
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                'Use this analysis to determine if the claim is TRUE or FALSE:\n\n'
                'STEP 1 - KEY FACTS:\n'
                '   - Extract key facts from the educator\'s analysis\n'
                '   - Consider the objectivity assessment\n\n'
                'STEP 2 - EVIDENCE EVALUATION:\n'
                '   - What evidence supports TRUE?\n'
                '   - What evidence supports FALSE?\n\n'
                'STEP 3 - FINAL DETERMINATION:\n'
                '   - Is this claim TRUE or FALSE?\n'
                '   - Provide clear justification\n\n'
                'Conclude with your final answer: TRUE or FALSE.'
        )
    else:
        # Without both
        system_prompt = (
                base_context +
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                'Use this analysis to determine if the claim is TRUE or FALSE:\n\n'
                'STEP 1 - KEY FACTS:\n'
                '   - Extract key facts from the educator\'s analysis\n\n'
                'STEP 2 - EVIDENCE EVALUATION:\n'
                '   - What evidence supports TRUE?\n'
                '   - What evidence supports FALSE?\n\n'
                'STEP 3 - FINAL DETERMINATION:\n'
                '   - Is this claim TRUE or FALSE?\n'
                '   - Provide clear justification\n\n'
                'Conclude with your final answer: TRUE or FALSE.'
        )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_final_answer(sentence, analysis, entity=None):
    """Extract the final answer (TRUE or FALSE) from analysis"""
    entity_context = f"Entity: {entity}\n" if entity else ""

    system_prompt = f'''You are evaluating a claim for the CREAK commonsense reasoning benchmark.

{entity_context}
CLAIM: "{sentence}"

ANALYSIS:
{analysis}

Based on the analysis above, is this claim TRUE or FALSE?

IMPORTANT:
- Return ONLY "TRUE" or "FALSE" (nothing else)
- TRUE means the claim is factually and logically correct
- FALSE means the claim is factually incorrect or logically impossible

YOUR ANSWER:'''

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "system", "content": system_prompt}]
    )
    answer = response.choices[0].message.content.strip().upper()

    # Extract just TRUE or FALSE
    if "TRUE" in answer and "FALSE" not in answer:
        return "TRUE"
    elif "FALSE" in answer:
        return "FALSE"
    return answer


def evaluate_answer(generated_answer, correct_label):
    """Evaluate the generated answer against the correct label"""
    gen = generated_answer.strip().upper()
    correct = correct_label.strip().upper()

    # Normalize
    if gen in ["TRUE", "T"]:
        gen = "TRUE"
    elif gen in ["FALSE", "F"]:
        gen = "FALSE"

    if correct in ["TRUE", "T"]:
        correct = "TRUE"
    elif correct in ["FALSE", "F"]:
        correct = "FALSE"

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


def run_pipeline(sentence, correct_label, config, entity=None):
    """Run the pipeline with specified configuration"""
    use_validity = config['use_validity']
    use_critical = config['use_critical']

    # Step 1: Backgrounding Educator (always runs)
    out_f1 = f1_backgrounding_educator(sentence, entity)

    # Step 2: Validity Checker (conditional)
    validity_output = None
    if use_validity:
        validity_output = validity_checker(sentence, entity)

    # Step 3: Critical Professor (conditional)
    out_f2 = None
    if use_critical:
        out_f2 = f2_critical_professor(sentence, out_f1, validity_output, entity)

    # Step 4: Meta-Reviewer
    out_f3 = f3_meta_reviewer(sentence, out_f1, out_f2, validity_output, entity)

    # Step 5: Extract final answer
    final_answer = extract_final_answer(sentence, out_f3, entity)

    # Step 6: Evaluate
    evaluation = evaluate_answer(final_answer, correct_label)

    return {
        'background': out_f1,
        'validity': validity_output,
        'critique': out_f2,
        'analysis': out_f3,
        'final_answer': final_answer,
        'evaluation': evaluation
    }


def run_ablation_study(sample_fraction=0.1, num_samples=None, start_index=0):
    """
    Run the ablation study on CREAK dataset

    Args:
        sample_fraction: Fraction of data to sample (default: 0.1 = 10%)
        num_samples: Additional limit on number of samples (None for all sampled)
        start_index: Starting index
    """
    # Load and sample data
    all_data = load_creak_data(sample_fraction=sample_fraction)

    if num_samples is not None:
        all_data = all_data[start_index:start_index + num_samples]
    else:
        all_data = all_data[start_index:]

    # Storage for results
    all_scores = {config_name: [] for config_name in ABLATION_CONFIGS.keys()}
    detailed_results = []

    total_rows = len(all_data)

    for idx, item in enumerate(all_data):
        print(f"\n{'=' * 60}")
        print(f"Processing item {idx + 1}/{total_rows}...")

        sentence = item['sentence']
        correct_label = item['label']
        entity = item.get('entity', '')
        explanation = item.get('explanation', '')
        ex_id = item.get('ex_id', '')
        split = item.get('split', '')

        print(f"Split: {split}")
        print(f"Claim: {sentence[:80]}...")
        print(f"Entity: {entity}")
        print(f"Correct Label: {correct_label}")

        row_results = {
            'ex_id': ex_id,
            'split': split,
            'sentence': sentence,
            'entity': entity,
            'explanation': explanation,
            'correct_label': correct_label
        }

        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n  Running: {config_name}")
            print(f"    - Use Validity: {config['use_validity']}")
            print(f"    - Use Critical: {config['use_critical']}")

            try:
                result = run_pipeline(sentence, correct_label, config, entity)

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
    """Compute summary statistics for ablation study (accuracy-based)"""
    summary = {}

    for config_name, evaluations in all_scores.items():
        # Filter out None values
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
    """Compare ablation configurations against each other"""
    comparison = {}

    configs = list(summary.keys())

    # Compute impact of removing each agent (compared to full pipeline)
    if 'full_pipeline' in summary:
        full_acc = summary['full_pipeline']['accuracy']
        comparison['ablation_impact'] = {}

        for config_name in ['no_validity', 'no_critical']:
            if config_name in summary:
                ablation_acc = summary[config_name]['accuracy']
                impact = full_acc - ablation_acc
                comparison['ablation_impact'][config_name] = {
                    'full_pipeline_accuracy': full_acc,
                    'ablation_accuracy': ablation_acc,
                    'impact': impact,
                    'interpretation': 'Removing this component HURTS performance' if impact > 0 else 'Removing this component HELPS performance' if impact < 0 else 'No impact'
                }

    # Pairwise comparisons
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

    # Best configuration
    best_config = max(summary.keys(), key=lambda x: summary[x]['accuracy'])
    comparison['best_configuration'] = {
        'name': best_config,
        'accuracy': summary[best_config]['accuracy'],
        'description': summary[best_config]['description']
    }

    return comparison


def compute_label_statistics(detailed_results):
    """Compute accuracy statistics broken down by true/false label"""
    df = pd.DataFrame(detailed_results)

    label_stats = {}

    for label in df['correct_label'].unique():
        label_df = df[df['correct_label'] == label]
        label_stats[label] = {}

        for config_name in ABLATION_CONFIGS.keys():
            eval_col = f'{config_name}_evaluation'
            valid_evals = label_df[eval_col].dropna()

            if len(valid_evals) > 0:
                accuracy = valid_evals.sum() / len(valid_evals)
                label_stats[label][config_name] = {
                    'accuracy': float(accuracy),
                    'correct': int(valid_evals.sum()),
                    'total': len(valid_evals)
                }
            else:
                label_stats[label][config_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }

    return label_stats


def compute_split_statistics(detailed_results):
    """Compute accuracy statistics broken down by data split"""
    df = pd.DataFrame(detailed_results)

    split_stats = {}

    for split in df['split'].unique():
        split_df = df[df['split'] == split]
        split_stats[split] = {}

        for config_name in ABLATION_CONFIGS.keys():
            eval_col = f'{config_name}_evaluation'
            valid_evals = split_df[eval_col].dropna()

            if len(valid_evals) > 0:
                accuracy = valid_evals.sum() / len(valid_evals)
                split_stats[split][config_name] = {
                    'accuracy': float(accuracy),
                    'correct': int(valid_evals.sum()),
                    'total': len(valid_evals)
                }
            else:
                split_stats[split][config_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }

    return split_stats


def print_summary_report(summary, comparison, label_stats=None, split_stats=None):
    """Print a formatted summary report"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY REPORT - CREAK Dataset (Commonsense Reasoning)")
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

    if label_stats:
        print("\n5. ACCURACY BY LABEL (TRUE vs FALSE)")
        print("-" * 80)
        print(f"{'Label':<15}", end='')
        for config_name in ABLATION_CONFIGS.keys():
            short_name = config_name[:12]
            print(f"{short_name:<15}", end='')
        print()
        print("-" * 80)

        for label, stats in sorted(label_stats.items()):
            print(f"{label:<15}", end='')
            for config_name in ABLATION_CONFIGS.keys():
                acc = stats[config_name]['accuracy']
                total = stats[config_name]['total']
                print(f"{acc:>6.0%} (n={total:<3}) ", end='')
            print()

    if split_stats:
        print("\n6. ACCURACY BY SPLIT")
        print("-" * 80)
        print(f"{'Split':<15}", end='')
        for config_name in ABLATION_CONFIGS.keys():
            short_name = config_name[:12]
            print(f"{short_name:<15}", end='')
        print()
        print("-" * 80)

        for split, stats in sorted(split_stats.items()):
            print(f"{split:<15}", end='')
            for config_name in ABLATION_CONFIGS.keys():
                acc = stats[config_name]['accuracy']
                total = stats[config_name]['total']
                print(f"{acc:>6.0%} (n={total:<3}) ", end='')
            print()

    print("\n" + "=" * 80)
    print("Note: Correct = Generated answer matches ground truth label (TRUE/FALSE)")
    print("=" * 80)


def save_results(detailed_results, summary, comparison, label_stats, split_stats, output_file, summary_file):
    """Save all results to files"""
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'CREAK',
        'task': 'Commonsense Reasoning over Entity Knowledge',
        'statistics': summary,
        'comparison': comparison,
        'label_statistics': label_stats,
        'split_statistics': split_stats
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_file}")


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study on CREAK Dataset (Commonsense Reasoning)')
    parser.add_argument('--output', type=str, default='results/creak_ablation_results.csv', help='Output CSV file')
    parser.add_argument('--summary', type=str, default='results/creak_ablation_summary.json', help='Summary JSON file')
    parser.add_argument('--fraction', type=float, default=0.02, help='Fraction of data to sample (default: 0.02, as in the paper)')
    parser.add_argument('--samples', type=int, default=None, help='Additional limit on samples (default: all sampled)')
    parser.add_argument('--start', type=int, default=0, help='Starting index')

    args = parser.parse_args()

    # Update global output file names
    output_file = args.output
    summary_file = args.summary

    print("=" * 80)
    print("ABLATION STUDY: CREAK Dataset (Commonsense Reasoning over Entity Knowledge)")
    print("Testing Full Pipeline vs Removing Validity Checker vs Removing Critical Professor")
    print("=" * 80)
    print(f"\nOutput file: {args.output}")
    print(f"Summary file: {args.summary}")
    print(f"Sample fraction: {args.fraction * 100:.1f}%")
    print(f"Additional sample limit: {'All sampled' if args.samples is None else args.samples}")
    print(f"Start index: {args.start}")

    print("\nConfigurations to test:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  - {name}: {config['description']}")

    print("\nStarting ablation study...")
    all_scores, detailed_results = run_ablation_study(
        sample_fraction=args.fraction,
        num_samples=args.samples,
        start_index=args.start
    )

    # Compute statistics
    summary = compute_statistics(all_scores)
    comparison = compute_ablation_comparison(summary)
    label_stats = compute_label_statistics(detailed_results)
    split_stats = compute_split_statistics(detailed_results)

    # Print report
    print_summary_report(summary, comparison, label_stats, split_stats)

    # Save results
    save_results(detailed_results, summary, comparison, label_stats, split_stats, args.output, args.summary)

    print("\nAblation study complete!")