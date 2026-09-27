import os
import numpy as np
from openai import OpenAI
import pandas as pd
from datetime import datetime
import json
import re

client = OpenAI()  # reads OPENAI_API_KEY from the environment
os.makedirs('results', exist_ok=True)

input_file = 'data/mmlu_test.csv'
output_file = 'results/mmlu_ablation_results.csv'
summary_file = 'results/mmlu_ablation_summary.json'


# ============== AGENT FUNCTIONS ==============

def f1_backgrounding_educator(question, choices, subject=None):
    """Agent 1: Backgrounding Educator - Provides foundational context for MMLU questions"""
    subject_context = f"Subject Area: {subject}\n" if subject else ""

    system_prompt = (
        'You are an expert educator preparing a student to answer a knowledge-based multiple choice question.\n\n'
        '=== TASK: MMLU KNOWLEDGE ASSESSMENT ===\n'
        'This question is from the MMLU (Massive Multitask Language Understanding) benchmark, '
        'which tests knowledge and reasoning across diverse academic subjects including STEM, '
        'humanities, social sciences, and professional domains.\n\n'
        f'{subject_context}'
        f'QUESTION:\n"{question}"\n\n'
        f'ANSWER CHOICES:\n{choices}\n\n'
        'Provide comprehensive background to support answering this question:\n\n'
        '1. DOMAIN KNOWLEDGE:\n'
        '   - Identify the specific topic/concept being tested\n'
        '   - Provide relevant definitions, formulas, or foundational concepts\n'
        '   - Explain the underlying principles or theories\n\n'
        '2. CHOICE ANALYSIS:\n'
        '   - Examine each answer choice (A, B, C, D)\n'
        '   - Explain what each choice represents or implies\n'
        '   - Identify any common misconceptions these choices might represent\n\n'
        '3. REASONING FRAMEWORK:\n'
        '   - What knowledge or reasoning is needed to solve this?\n'
        '   - Are there any key facts, dates, or relationships to consider?\n'
        '   - What approach should be used to determine the correct answer?\n\n'
        '4. PRELIMINARY ASSESSMENT:\n'
        '   - Based on the domain knowledge, which answer(s) appear most plausible?\n'
        '   - Which can be eliminated and why?\n\n'
        'Present your analysis clearly and accurately to support correct answer selection.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def validity_checker(question, choices, subject=None):
    """Agent 2: Validity Checker - Assesses objectivity and answer certainty for MMLU questions"""
    subject_context = f"Subject Area: {subject}\n" if subject else ""

    system_prompt = (
        'You are an expert in evaluating the objectivity and clarity of knowledge-based questions.\n\n'
        '=== TASK: ASSESS QUESTION OBJECTIVITY ===\n'
        'This question is from the MMLU benchmark, which is designed to have definitive correct answers '
        'based on established knowledge, facts, or accepted academic consensus.\n\n'
        f'{subject_context}'
        f'QUESTION:\n"{question}"\n\n'
        f'ANSWER CHOICES:\n{choices}\n\n'
        'Assess the following aspects:\n\n'
        '1. ANSWER OBJECTIVITY:\n'
        '   - Is there ONE definitively correct answer based on established facts/knowledge?\n'
        '   - Is the answer verifiable through scientific evidence, historical record, or academic consensus?\n'
        '   - Or could this be subjective/contested among experts?\n\n'
        '2. KNOWLEDGE TYPE:\n'
        '   - Is this testing factual recall (dates, names, definitions)?\n'
        '   - Is this testing conceptual understanding or application?\n'
        '   - Is this testing reasoning or problem-solving?\n\n'
        '3. POTENTIAL AMBIGUITIES:\n'
        '   - Are any answer choices potentially correct under different interpretations?\n'
        '   - Are there edge cases or exceptions to consider?\n'
        '   - Could the question wording lead to misinterpretation?\n\n'
        '4. CONFIDENCE LEVEL:\n'
        '   - Rate the objectivity: High (clear factual answer) / Medium (some nuance) / Low (debatable)\n'
        '   - Explain your rating\n\n'
        'Your assessment will help calibrate confidence in the final answer selection.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def f2_critical_professor(question, choices, generation_1, objective=None, subject=None):
    """Agent 3: Critical Professor - Critiques and refines analysis for MMLU questions"""
    objective_text = objective if objective else "Not assessed"
    subject_context = f"Subject Area: {subject}\n" if subject else ""

    system_prompt = (
        'You are a critical professor with expertise across multiple academic disciplines. '
        'Your role is to rigorously examine the initial analysis and identify any errors or oversights.\n\n'
        '=== TASK: CRITICAL REVIEW OF MMLU QUESTION ANALYSIS ===\n'
        'This is a multiple-choice knowledge question from the MMLU benchmark.\n\n'
        f'{subject_context}'
        f'QUESTION:\n"{question}"\n\n'
        f'ANSWER CHOICES:\n{choices}\n\n'
        f'OBJECTIVITY ASSESSMENT:\n{objective_text}\n\n'
        f'INITIAL EDUCATOR ANALYSIS:\n{generation_1}\n\n'
        'Provide a rigorous critical review:\n\n'
        'STEP 1 - QUESTION INTERPRETATION:\n'
        '   - Is the question being interpreted correctly?\n'
        '   - Are there alternative valid interpretations?\n'
        '   - What exactly is being asked?\n\n'
        'STEP 2 - KNOWLEDGE VERIFICATION:\n'
        '   - Are the facts/concepts in the initial analysis accurate?\n'
        '   - Are there any factual errors or misconceptions?\n'
        '   - Is any critical information missing?\n\n'
        'STEP 3 - CHOICE-BY-CHOICE EVALUATION:\n'
        '   For EACH answer choice (A, B, C, D):\n'
        '   - Is it correct, partially correct, or incorrect?\n'
        '   - What specific evidence supports or refutes it?\n'
        '   - Could it be a "trap" answer that seems right but isn\'t?\n\n'
        'STEP 4 - COUNTERARGUMENTS:\n'
        '   - What arguments could be made for choices NOT favored in the initial analysis?\n'
        '   - Are there any edge cases or exceptions?\n'
        '   - What common mistakes might lead to wrong answers?\n\n'
        'STEP 5 - REFINED CONCLUSION:\n'
        '   - Based on your critical analysis, which answer is DEFINITIVELY correct?\n'
        '   - Provide clear reasoning with specific evidence\n\n'
        'Be thorough, precise, and evidence-based in your critique.'
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def f3_meta_reviewer(question, choices, generation_1, generation_2=None, validity_output=None, subject=None):
    """Agent 4: Meta-Reviewer - Synthesizes analyses and determines the correct answer"""
    subject_context = f"Subject Area: {subject}\n" if subject else ""

    base_context = (
        'You are a meta-analyst responsible for synthesizing multiple expert analyses '
        'to determine the correct answer to a knowledge-based question.\n\n'
        '=== TASK: FINAL ANSWER DETERMINATION FOR MMLU QUESTION ===\n'
        'This is a multiple-choice question from the MMLU benchmark. '
        'There is ONE correct answer based on established knowledge.\n\n'
        f'{subject_context}'
        f'QUESTION:\n"{question}"\n\n'
        f'ANSWER CHOICES:\n{choices}\n\n'
    )

    if generation_2 is not None and validity_output is not None:
        # Full pipeline
        system_prompt = (
                base_context +
                f'OBJECTIVITY ASSESSMENT:\n{validity_output}\n\n'
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                f'CRITICAL PROFESSOR ANALYSIS:\n{generation_2}\n\n'
                'Synthesize these analyses to determine the correct answer:\n\n'
                'STEP 1 - CONSENSUS IDENTIFICATION:\n'
                '   - Where do both analyses agree?\n'
                '   - What facts/reasoning are consistently supported?\n\n'
                'STEP 2 - CONFLICT RESOLUTION:\n'
                '   - Where do the analyses disagree?\n'
                '   - Which position has stronger evidence/reasoning?\n'
                '   - Resolve any contradictions\n\n'
                'STEP 3 - EVIDENCE WEIGHING:\n'
                '   - What is the strongest evidence for each choice?\n'
                '   - Which answer has the most compelling support?\n'
                '   - Which answers can be definitively eliminated?\n\n'
                'STEP 4 - FINAL ANSWER:\n'
                '   - Select the SINGLE correct answer: A, B, C, or D\n'
                '   - Provide clear, concise justification\n\n'
                'Conclude with your definitive answer (A, B, C, or D).'
        )
    elif generation_2 is not None and validity_output is None:
        # Without validity checker
        system_prompt = (
                base_context +
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                f'CRITICAL PROFESSOR ANALYSIS:\n{generation_2}\n\n'
                'Synthesize these analyses to determine the correct answer:\n\n'
                'STEP 1 - CONSENSUS IDENTIFICATION:\n'
                '   - Where do both analyses agree?\n'
                '   - What facts/reasoning are consistently supported?\n\n'
                'STEP 2 - CONFLICT RESOLUTION:\n'
                '   - Where do the analyses disagree?\n'
                '   - Which position has stronger evidence/reasoning?\n'
                '   - Resolve any contradictions\n\n'
                'STEP 3 - EVIDENCE WEIGHING:\n'
                '   - What is the strongest evidence for each choice?\n'
                '   - Which answer has the most compelling support?\n'
                '   - Which answers can be definitively eliminated?\n\n'
                'STEP 4 - FINAL ANSWER:\n'
                '   - Select the SINGLE correct answer: A, B, C, or D\n'
                '   - Provide clear, concise justification\n\n'
                'Conclude with your definitive answer (A, B, C, or D).'
        )
    elif generation_2 is None and validity_output is not None:
        # Without critical professor
        system_prompt = (
                base_context +
                f'OBJECTIVITY ASSESSMENT:\n{validity_output}\n\n'
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                'Use this analysis to determine the correct answer:\n\n'
                'STEP 1 - KEY INSIGHTS:\n'
                '   - What are the main findings from the educator\'s analysis?\n'
                '   - What evidence points to specific answers?\n\n'
                'STEP 2 - OBJECTIVITY CONSIDERATION:\n'
                '   - How clear-cut is this question?\n'
                '   - Any ambiguities to factor in?\n\n'
                'STEP 3 - ANSWER SELECTION:\n'
                '   - Evaluate each choice based on the analysis\n'
                '   - Identify the best supported answer\n\n'
                'STEP 4 - FINAL ANSWER:\n'
                '   - Select the SINGLE correct answer: A, B, C, or D\n'
                '   - Provide clear, concise justification\n\n'
                'Conclude with your definitive answer (A, B, C, or D).'
        )
    else:
        # Without both
        system_prompt = (
                base_context +
                f'EDUCATOR ANALYSIS:\n{generation_1}\n\n'
                'Use this analysis to determine the correct answer:\n\n'
                'STEP 1 - KEY INSIGHTS:\n'
                '   - What are the main findings from the educator\'s analysis?\n'
                '   - What evidence points to specific answers?\n\n'
                'STEP 2 - ANSWER SELECTION:\n'
                '   - Evaluate each choice based on the analysis\n'
                '   - Identify the best supported answer\n\n'
                'STEP 3 - FINAL ANSWER:\n'
                '   - Select the SINGLE correct answer: A, B, C, or D\n'
                '   - Provide clear, concise justification\n\n'
                'Conclude with your definitive answer (A, B, C, or D).'
        )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def extract_final_answer(question, choices, analysis, subject=None):
    """Extract the final answer letter (A, B, C, or D) from analysis"""
    subject_context = f"Subject: {subject}\n" if subject else ""

    system_prompt = f'''You are answering a multiple-choice question from the MMLU knowledge benchmark.

=== MMLU QUESTION ===
{subject_context}
Question: {question}

Choices:
{choices}

Analysis provided:
{analysis}

Based on the analysis above, which choice is the CORRECT answer?

INSTRUCTIONS:
- MMLU questions have ONE definitively correct answer
- Select the answer best supported by the analysis and established knowledge
- Return ONLY a single letter: A, B, C, or D
- Do not include any explanation, just the letter

YOUR ANSWER:'''

    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content.strip().upper()

    # Extract just the letter if there's extra text
    match = re.search(r'[ABCD]', answer)
    if match:
        return match.group()
    return answer


def evaluate_answer(generated_answer, correct_answer):
    """Evaluate the generated answer against the correct answer - returns TRUE or FALSE"""
    # Normalize both answers
    gen = generated_answer.strip().upper()
    correct = correct_answer.strip().upper()

    # Extract just the letter if needed
    gen_match = re.search(r'[ABCD]', gen)
    if gen_match:
        gen = gen_match.group()

    return gen == correct


def format_choices(row):
    """Format the choices from the row into a readable string"""
    return f"A) {row['A']}\nB) {row['B']}\nC) {row['C']}\nD) {row['D']}"


# ============== ABLATION CONFIGURATIONS (INCLUDING FULL PIPELINE) ==============

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


def run_pipeline(question, choices, correct_answer, config, subject=None):
    """Run the pipeline with specified configuration"""
    use_validity = config['use_validity']
    use_critical = config['use_critical']

    # Step 1: Backgrounding Educator (always runs)
    out_f1 = f1_backgrounding_educator(question, choices, subject)

    # Step 2: Validity Checker (conditional)
    validity_output = None
    if use_validity:
        validity_output = validity_checker(question, choices, subject)

    # Step 3: Critical Professor (conditional) - now receives validity output
    out_f2 = None
    if use_critical:
        out_f2 = f2_critical_professor(question, choices, out_f1, validity_output, subject)

    # Step 4: Meta-Reviewer (adapts based on available inputs)
    out_f3 = f3_meta_reviewer(question, choices, out_f1, out_f2, validity_output, subject)

    # Step 5: Extract final answer (letter A, B, C, or D)
    final_answer = extract_final_answer(question, choices, out_f3, subject)

    # Step 6: Evaluate against correct answer (TRUE/FALSE)
    evaluation = evaluate_answer(final_answer, correct_answer)

    return {
        'background': out_f1,
        'validity': validity_output,
        'critique': out_f2,
        'analysis': out_f3,
        'final_answer': final_answer,
        'evaluation': evaluation  # Boolean: True or False
    }


def stratified_sample_by_subject(df, sample_fraction=0.1, random_state=42):
    """
    Sample a fraction of data from each subject to maintain balanced representation.

    Args:
        df: DataFrame with 'Subject' column
        sample_fraction: Fraction of data to sample from each subject (default: 0.1 = 10%)
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame with specified fraction from each subject
    """
    # Handle different column name cases
    subject_col = 'Subject' if 'Subject' in df.columns else 'subject'

    sampled_dfs = []

    print(f"\nStratified sampling: {sample_fraction * 100:.0f}% from each subject")
    print("-" * 60)

    for subject in df[subject_col].unique():
        subject_df = df[df[subject_col] == subject]
        n_samples = max(1, int(len(subject_df) * sample_fraction))  # At least 1 sample per subject
        sampled = subject_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"  {subject}: {len(subject_df)} total -> {n_samples} sampled")

    result = pd.concat(sampled_dfs, ignore_index=False)
    print("-" * 60)
    print(f"Total samples: {len(result)} (from {len(df)} original)")

    return result


def run_ablation_study(input_file, num_samples=None, start_index=0, sample_fraction=0.1):
    """
    Run the ablation study on MMLU dataset

    Args:
        input_file: Path to input CSV file
        num_samples: Number of samples to process (None for all, after stratified sampling)
        start_index: Starting row index
        sample_fraction: Fraction to sample from each subject (default: 0.1 = 10%)
    """
    result_df = pd.read_csv(input_file)

    # Apply stratified sampling by subject
    result_df = stratified_sample_by_subject(result_df, sample_fraction=sample_fraction)

    if num_samples is not None:
        result_df = result_df.iloc[start_index:start_index + num_samples]
    else:
        result_df = result_df.iloc[start_index:]

    # Storage for all results
    all_scores = {config_name: [] for config_name in ABLATION_CONFIGS.keys()}
    detailed_results = []

    total_rows = len(result_df)

    for idx, (row_index, row_data) in enumerate(result_df.iterrows()):
        print(f"\n{'=' * 60}")
        print(f"Processing row {idx + 1}/{total_rows} (index: {row_index})...")

        question = row_data['Question']
        correct_answer = row_data['Answer']
        subject = row_data.get('Subject', row_data.get('subject', ''))
        choices = format_choices(row_data)

        print(f"Subject: {subject}")
        print(f"Question: {question[:80]}...")
        print(f"Correct Answer: {correct_answer}")

        row_results = {
            'row_index': row_index,
            'question': question,
            'correct_answer': correct_answer,
            'subject': subject,
            'choice_A': row_data['A'],
            'choice_B': row_data['B'],
            'choice_C': row_data['C'],
            'choice_D': row_data['D']
        }

        for config_name, config in ABLATION_CONFIGS.items():
            print(f"\n  Running: {config_name}")
            print(f"    - Use Validity: {config['use_validity']}")
            print(f"    - Use Critical: {config['use_critical']}")

            try:
                result = run_pipeline(question, choices, correct_answer, config, subject)

                evaluation = result['evaluation']  # Boolean: True or False
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
            print(f"\n  [Saved intermediate results at row {idx + 1}]")

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


def compute_subject_statistics(detailed_results):
    """Compute accuracy statistics broken down by subject"""
    df = pd.DataFrame(detailed_results)

    subject_stats = {}

    for subject in df['subject'].unique():
        subject_df = df[df['subject'] == subject]
        subject_stats[subject] = {}

        for config_name in ABLATION_CONFIGS.keys():
            eval_col = f'{config_name}_evaluation'
            valid_evals = subject_df[eval_col].dropna()

            if len(valid_evals) > 0:
                accuracy = valid_evals.sum() / len(valid_evals)
                subject_stats[subject][config_name] = {
                    'accuracy': float(accuracy),
                    'correct': int(valid_evals.sum()),
                    'total': len(valid_evals)
                }
            else:
                subject_stats[subject][config_name] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0
                }

    return subject_stats


def compute_ablation_comparison(summary):
    """Compare ablation configurations against each other"""
    comparison = {}

    configs = list(summary.keys())

    # Compare ablations against full pipeline
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


def print_summary_report(summary, comparison, subject_stats=None):
    """Print a formatted summary report"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY REPORT - MMLU Dataset (Knowledge Assessment)")
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

    if subject_stats:
        print("\n5. ACCURACY BY SUBJECT")
        print("-" * 80)
        print(f"{'Subject':<35}", end='')
        for config_name in ABLATION_CONFIGS.keys():
            short_name = config_name[:12]
            print(f"{short_name:<15}", end='')
        print()
        print("-" * 80)

        for subject, stats in sorted(subject_stats.items()):
            print(f"{subject[:34]:<35}", end='')
            for config_name in ABLATION_CONFIGS.keys():
                acc = stats[config_name]['accuracy']
                total = stats[config_name]['total']
                print(f"{acc:>6.0%} (n={total:<3}) ", end='')
            print()

    print("\n" + "=" * 80)
    print("Note: Correct = Generated answer matches correct answer letter")
    print("=" * 80)


def save_results(detailed_results, summary, comparison, subject_stats, output_file, summary_file):
    """Save all results to files"""
    # Save detailed results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'MMLU',
        'task': 'Multiple-Choice Knowledge Assessment',
        'statistics': summary,
        'comparison': comparison,
        'subject_statistics': subject_stats
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved to: {summary_file}")


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study on MMLU Dataset (Knowledge Assessment)')
    parser.add_argument('--input', type=str, default='data/mmlu_test.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='results/mmlu_ablation_results.csv', help='Output CSV file')
    parser.add_argument('--summary', type=str, default='results/mmlu_ablation_summary.json', help='Summary JSON file')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to process (default: all)')
    parser.add_argument('--start', type=int, default=0, help='Starting row index')
    parser.add_argument('--fraction', type=float, default=0.05,
                        help='Fraction to sample from each subject (default: 0.05 = 5%%)')

    args = parser.parse_args()

    print("=" * 80)
    print("ABLATION STUDY: MMLU Dataset (Multiple-Choice Knowledge Assessment)")
    print("Testing Full Pipeline vs Removing Validity Checker vs Removing Critical Professor")
    print("=" * 80)
    print(f"\nInput file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Summary file: {args.summary}")
    print(f"Sample fraction per subject: {args.fraction * 100:.0f}%")
    print(f"Additional sample limit: {'All' if args.samples is None else args.samples}")
    print(f"Start index: {args.start}")

    print("\nConfigurations to test:")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  - {name}: {config['description']}")

    # Run ablation study
    print("\nStarting ablation study...")
    all_scores, detailed_results = run_ablation_study(args.input, args.samples, args.start, args.fraction)

    # Compute statistics
    summary = compute_statistics(all_scores)
    comparison = compute_ablation_comparison(summary)
    subject_stats = compute_subject_statistics(detailed_results)

    # Print report
    print_summary_report(summary, comparison, subject_stats)

    # Save results
    save_results(detailed_results, summary, comparison, subject_stats, args.output, args.summary)

    print("\nAblation study complete!")