import os
import numpy as np
from openai import OpenAI
import pandas as pd
from datetime import datetime
import json
import re
import glob
import random
import argparse

client = OpenAI()  # reads OPENAI_API_KEY from the environment
os.makedirs('results', exist_ok=True)

# ============== MULTI-EXPERT PROMPTING TEMPLATES ==============

MEP_PROMPTS = {
    "role_generation": {
        "system_message": "You are a helpful assistant tasked with identifying the best roles to answer a given question.",
        "user_message": "You are provided with a question:\n<QUESTION>{{QUESTION}}<QUESTION>\nGive me a dictionary of exactly {{NUM_EXPERTS}} experts that could answer the question above most thoroughly and truthfully. The dictionary should have each role as a key and a short description for each role as the value. Fill in the format below:\n{{ROLE_FORMAT}}"
    },
    "expert_answer_generation": {
        "system_message": "You are a helpful assistant tasked with generating the best answer to a question from an expert's perspective.",
        "user_message": "From now on, you are an excellent {{ROLE}} described as {{ROLE_DESCRIPTION}} You are helping me to answer the following question, which greatly requires your expertise as a {{ROLE}}: {{QUESTION}}. The answer must not contain more than 500 words."
    },
    "merging_prompt": {
        "system_message": "You are a helpful assistant tasked with combining answers from multiple experts to generate the most accurate and comprehensive answer to a question.",
        "user_message": "Given the following question: {{QUESTION}}, you have obtained {{NUM_EXPERTS}} answers from {{NUM_EXPERTS}} experts with different expertises:\n{{ANSWER_FORMAT}}\n\nYour task is to aggregate the experts' answers above, following the steps below.\n\nStep 1: Identify the facts that more than half of the answers agree upon.\nFacts that more than half of the answers agree upon (Agreed Facts):...\n\nStep 2: Identify the facts that conflict among the answers.\nConflicting facts among the answers (Conflicted Facts):...\n\nStep 3: Resolve the conflicting facts from Step 2. The facts that more people agree upon are more likely to be true.\nResolved facts from Step 2:...\n\nStep 4: Identify the facts that are not from Step 2 and 1, and are only mentioned in one of the answers.\nFacts that are excluded from Step 2 and 1 and are only mentioned in one of the answers:...\n\nStep 5: Combine the facts from Steps 1, 3, and 4 to obtain the final solution.\nFacts from Steps 1, 3, and 4:...\n\nStep 6: Generate a comprehensive final answer consisting of the facts from Step 5, in a newline.\nCombined answer:...\n\nStep 7: Given the answers from {{ANSWER_CHOICES}}, which do you think is the most truthful, informative, useful, factually-correct, and honest for answering this question: {{QUESTION}}?\n- Best answer choice: [Choose one among {{ANSWER_CHOICES}}]\n- Explanation: [Explanation for your choice of the best answer]\n- Final answer: [Output the exact content of the chosen answer, do not trim or modify the answer, in a newline]"
    }
}

NUM_EXPERTS = 3

# Dataset configurations
DATASET_CONFIGS = {
    'mafalda': {
        'name': 'MAFALDA',
        'description': 'Logical Fallacy Detection',
        'default_input': 'data/gold_standard_dataset.jsonl',
        'output_prefix': 'mafalda'
    },
    'mmlu': {
        'name': 'MMLU',
        'description': 'Multiple-Choice Knowledge Assessment',
        'default_input': 'data/mmlu_test.csv',
        'output_prefix': 'mmlu'
    },
    'bbq': {
        'name': 'BBQ',
        'description': 'Bias Benchmark for QA',
        'default_input': 'data/bbq',
        'output_prefix': 'bbq'
    },
    'crowspairs': {
        'name': 'CrowS-Pairs',
        'description': 'Stereotype Detection',
        'default_input': 'data/crows_pairs_anonymized.csv',
        'output_prefix': 'crowspairs'
    },
    'creak': {
        'name': 'CREAK',
        'description': 'Commonsense Reasoning over Entity Knowledge',
        'default_input': 'creak',
        'output_prefix': 'creak'
    }
}


# ============== DATASET LOADING FUNCTIONS ==============

def load_mafalda_data(input_file, random_state=42):
    """Load MAFALDA dataset from JSONL file."""
    random.seed(random_state)
    np.random.seed(random_state)

    all_data = []

    # All possible fallacy types in MAFALDA
    fallacy_types = [
        'ad hominem', 'ad populum', 'appeal to (false) authority', 'appeal to anger',
        'appeal to fear', 'appeal to nature', 'appeal to pity', 'appeal to positive emotion',
        'Appeal to Ridicule', 'appeal to tradition', 'appeal to worse problems',
        'causal oversimplification', 'circular reasoning', 'equivocation', 'fallacy of division',
        'false analogy', 'false causality', 'false dilemma', 'guilt by association',
        'hasty generalization', 'slippery slope', 'straw man', 'tu quoque'
    ]

    print(f"\nLoading MAFALDA data from {input_file}")
    print("-" * 60)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                item = json.loads(line)

                primary_fallacy = None
                fallacy_span = None

                for label in item['labels']:
                    start, end, fallacy_type = label
                    if fallacy_type not in ['nothing', 'to clean']:
                        primary_fallacy = fallacy_type
                        fallacy_span = item['text'][start:end]
                        break

                if primary_fallacy is None:
                    continue

                # Generate multiple choice options
                available_distractors = [f for f in fallacy_types if f.lower() != primary_fallacy.lower()]
                random.seed(line_num + 42)
                distractors = random.sample(available_distractors, 3)
                choices_list = [primary_fallacy] + distractors
                random.shuffle(choices_list)
                correct_index = choices_list.index(primary_fallacy)
                correct_letter = chr(65 + correct_index)

                processed_item = {
                    'id': line_num,
                    'question': f"What logical fallacy is present in the following text?\n\nText: \"{item['text'].strip()}\"",
                    'choices': choices_list,
                    'correct_answer': correct_letter,
                    'correct_label': primary_fallacy,
                    'dataset': 'mafalda',
                    'category': 'logical_fallacy'
                }
                all_data.append(processed_item)

    print(f"Total valid items: {len(all_data)}")
    print("-" * 60)

    return all_data


def load_mmlu_data(input_file, sample_fraction=0.05, random_state=42):
    """Load MMLU dataset from CSV file with stratified sampling."""
    random.seed(random_state)
    np.random.seed(random_state)

    df = pd.read_csv(input_file)
    subject_col = 'Subject' if 'Subject' in df.columns else 'subject'

    print(f"\nLoading MMLU data from {input_file}")
    print("-" * 60)

    # Stratified sampling
    sampled_dfs = []
    for subject in df[subject_col].unique():
        subject_df = df[df[subject_col] == subject]
        n_samples = max(1, int(len(subject_df) * sample_fraction))
        sampled = subject_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"  {subject}: {len(subject_df)} total -> {n_samples} sampled")

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    all_data = []
    for idx, row in result_df.iterrows():
        processed_item = {
            'id': idx,
            'question': row['Question'],
            'choices': [row['A'], row['B'], row['C'], row['D']],
            'correct_answer': row['Answer'],
            'correct_label': row['Answer'],
            'dataset': 'mmlu',
            'category': row.get('Subject', row.get('subject', ''))
        }
        all_data.append(processed_item)

    print("-" * 60)
    print(f"Total samples: {len(all_data)}")

    return all_data


def load_bbq_data(bbq_folder, samples_per_category=50, random_state=42):
    """Load BBQ dataset from JSONL files."""
    random.seed(random_state)
    np.random.seed(random_state)

    all_data = []
    jsonl_files = glob.glob(os.path.join(bbq_folder, '*.jsonl'))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {bbq_folder}")

    print(f"\nLoading BBQ data from {bbq_folder}")
    print("-" * 60)

    label_to_letter = {0: 'A', 1: 'B', 2: 'C'}

    for filepath in sorted(jsonl_files):
        category_name = os.path.basename(filepath).replace('.jsonl', '')

        items = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item['category'] = category_name
                    items.append(item)

        n_samples = min(samples_per_category, len(items))
        sampled = random.sample(items, n_samples)

        for item in sampled:
            processed_item = {
                'id': item.get('example_id', ''),
                'question': f"{item['context']}\n\nQuestion: {item['question']}",
                'choices': [item['ans0'], item['ans1'], item['ans2']],
                'correct_answer': label_to_letter[item['label']],
                'correct_label': item[f"ans{item['label']}"],
                'dataset': 'bbq',
                'category': category_name
            }
            all_data.append(processed_item)

        print(f"  {category_name}: {len(items)} total -> {n_samples} sampled")

    print("-" * 60)
    print(f"Total samples: {len(all_data)}")

    return all_data


def load_crowspairs_data(input_file, sample_fraction=1 / 3, random_state=42):
    """Load CrowS-Pairs dataset from CSV file with stratified sampling."""
    random.seed(random_state)
    np.random.seed(random_state)

    print(f"\nLoading CrowS-Pairs data from {input_file}")
    print("-" * 60)

    df = pd.read_csv(input_file)
    print(f"Total rows in dataset: {len(df)}")

    # Stratified sampling by bias_type
    sampled_dfs = []
    for bias_type in df['bias_type'].unique():
        type_df = df[df['bias_type'] == bias_type]
        n_samples = max(1, int(len(type_df) * sample_fraction))
        sampled = type_df.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled)
        print(f"  {bias_type}: {len(type_df)} total -> {n_samples} sampled")

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    all_data = []
    for idx, row in result_df.iterrows():
        # Randomly assign sentences to A or B to avoid position bias
        random.seed(idx + 42)
        sent_more = row['sent_more']  # More stereotypical
        sent_less = row['sent_less']  # Less stereotypical

        if random.random() < 0.5:
            choice_a = sent_more
            choice_b = sent_less
            correct_letter = 'A'
        else:
            choice_a = sent_less
            choice_b = sent_more
            correct_letter = 'B'

        question_text = (
            "The following two sentences differ in their portrayal of certain groups. "
            "One sentence contains a stereotype or biased assumption about a group of people, "
            "while the other is less stereotypical.\n\n"
            "Your task is to identify which sentence contains the MORE stereotypical portrayal."
        )

        processed_item = {
            'id': idx,
            'question': question_text,
            'choices': [choice_a, choice_b],
            'correct_answer': correct_letter,
            'correct_label': sent_more,
            'dataset': 'crowspairs',
            'category': row['bias_type'],
            'stereo_antistereo': row.get('stereo_antistereo', '')
        }
        all_data.append(processed_item)

    print("-" * 60)
    print(f"Total sampled: {len(all_data)} ({sample_fraction * 100:.1f}% of original)")

    return all_data


def load_creak_data(input_source, sample_fraction=0.1, random_state=42):
    """
    Load CREAK dataset from HuggingFace or local files.

    Args:
        input_source: Either 'huggingface' to load from HF, or path to local JSON/JSONL files
        sample_fraction: Fraction of data to sample (default: 0.1 = 10%)
        random_state: Random seed for reproducibility

    Returns:
        List of processed data items formatted for MEP
    """
    random.seed(random_state)
    np.random.seed(random_state)

    all_data = []

    print(f"\nLoading CREAK data from {input_source}")
    print("-" * 60)

    # Try loading from HuggingFace
    if input_source == 'huggingface' or input_source == 'creak':
        try:
            from datasets import load_dataset
            ds = load_dataset("amydeng2000/CREAK")

            for split in ds:
                split_data = ds[split]
                n_total = len(split_data)
                n_samples = max(1, int(n_total * sample_fraction))

                # Random sampling
                indices = random.sample(range(n_total), n_samples)

                for idx in indices:
                    item = split_data[idx]

                    # Format as binary choice question
                    sentence = item['sentence']
                    label = item['label'].upper()  # 'true' or 'false' -> 'TRUE' or 'FALSE'
                    entity = item.get('entity', '')

                    # Create question
                    question_text = (
                        f"Determine whether the following claim is TRUE or FALSE based on commonsense reasoning and entity knowledge.\n\n"
                        f"Claim: \"{sentence}\""
                    )
                    if entity:
                        question_text += f"\n\n(This claim is about: {entity})"

                    # Choices are always TRUE (A) and FALSE (B)
                    choices = ['TRUE', 'FALSE']
                    correct_letter = 'A' if label == 'TRUE' else 'B'

                    processed_item = {
                        'id': item.get('ex_id', f'{split}_{idx}'),
                        'question': question_text,
                        'choices': choices,
                        'correct_answer': correct_letter,
                        'correct_label': label,
                        'dataset': 'creak',
                        'category': split,
                        'entity': entity
                    }
                    all_data.append(processed_item)

                print(f"  {split}: {n_total} total -> {n_samples} sampled ({sample_fraction * 100:.0f}%)")

        except ImportError:
            print("  datasets library not available, trying local files...")
            input_source = 'creak_data'  # Fall back to local
        except Exception as e:
            print(f"  Error loading from HuggingFace: {e}")
            print("  Trying local files...")
            input_source = 'creak_data'

    # Try loading from local files if HuggingFace failed or local path specified
    if not all_data and input_source != 'huggingface':
        # Check for local JSON/JSONL files
        local_files = []
        if os.path.isdir(input_source):
            local_files = glob.glob(os.path.join(input_source, '*.json')) + \
                          glob.glob(os.path.join(input_source, '*.jsonl'))
        elif os.path.isfile(input_source):
            local_files = [input_source]

        if local_files:
            for filepath in local_files:
                split_name = os.path.basename(filepath).replace('.json', '').replace('.jsonl', '')

                items = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    if filepath.endswith('.jsonl'):
                        for line in f:
                            if line.strip():
                                items.append(json.loads(line))
                    else:
                        content = json.load(f)
                        if isinstance(content, list):
                            items = content
                        else:
                            items = [content]

                n_total = len(items)
                n_samples = max(1, int(n_total * sample_fraction))
                sampled = random.sample(items, min(n_samples, n_total))

                for idx, item in enumerate(sampled):
                    sentence = item.get('sentence', item.get('claim', ''))
                    label = item.get('label', 'unknown').upper()
                    entity = item.get('entity', '')

                    question_text = (
                        f"Determine whether the following claim is TRUE or FALSE based on commonsense reasoning and entity knowledge.\n\n"
                        f"Claim: \"{sentence}\""
                    )
                    if entity:
                        question_text += f"\n\n(This claim is about: {entity})"

                    choices = ['TRUE', 'FALSE']
                    correct_letter = 'A' if label == 'TRUE' else 'B'

                    processed_item = {
                        'id': item.get('ex_id', f'{split_name}_{idx}'),
                        'question': question_text,
                        'choices': choices,
                        'correct_answer': correct_letter,
                        'correct_label': label,
                        'dataset': 'creak',
                        'category': split_name,
                        'entity': entity
                    }
                    all_data.append(processed_item)

                print(f"  {split_name}: {n_total} total -> {len(sampled)} sampled")
        else:
            print(f"  No local files found at {input_source}")

    print("-" * 60)
    print(f"Total samples: {len(all_data)}")

    return all_data


# ============== MULTI-EXPERT PROMPTING FUNCTIONS ==============

def format_choices(choices):
    """Format choices into a readable string."""
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    formatted = []
    for i, choice in enumerate(choices):
        formatted.append(f"{letters[i]}) {choice}")
    return '\n'.join(formatted)


def get_answer_choices_string(choices):
    """Get answer choices as A, B, C, D string."""
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ', '.join([letters[i] for i in range(len(choices))])


def generate_role_format(num_experts):
    """Generate the role format template for expert generation."""
    roles = []
    for i in range(1, num_experts + 1):
        roles.append(f'"Role {i}": "Description of Role {i}"')
    return '{\n' + ',\n'.join(roles) + '\n}'


def generate_answer_format(experts_answers):
    """Generate the answer format for merging prompt."""
    formatted = []
    for i, (role, desc, answer) in enumerate(experts_answers, 1):
        formatted.append(f"Expert {i} ({role}): {answer}")
    return '\n\n'.join(formatted)


def call_openai(system_message, user_message, model="gpt-3.5-turbo", temperature=0):
    """Make an OpenAI API call."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


def generate_experts(question, num_experts=3):
    """Generate expert roles for a given question."""
    role_format = generate_role_format(num_experts)

    system_msg = MEP_PROMPTS["role_generation"]["system_message"]
    user_msg = MEP_PROMPTS["role_generation"]["user_message"]
    user_msg = user_msg.replace("{{QUESTION}}", question)
    user_msg = user_msg.replace("{{NUM_EXPERTS}}", str(num_experts))
    user_msg = user_msg.replace("{{ROLE_FORMAT}}", role_format)

    response = call_openai(system_msg, user_msg)

    # Parse the response to extract roles
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            roles_dict = json.loads(json_match.group())
        else:
            # Fallback: try to parse the entire response
            roles_dict = json.loads(response)
    except json.JSONDecodeError:
        # Fallback to default roles if parsing fails
        roles_dict = {
            "Domain Expert": "An expert with deep knowledge in the relevant subject matter",
            "Critical Analyst": "An expert in critical thinking and logical analysis",
            "Practical Specialist": "A specialist with hands-on experience in applying knowledge"
        }

    # Convert to list of tuples
    experts = [(role, desc) for role, desc in roles_dict.items()][:num_experts]

    # Pad with default roles if needed
    default_roles = [
        ("Subject Matter Expert", "An expert with comprehensive knowledge in the relevant field"),
        ("Analytical Reviewer", "An expert in systematic analysis and evaluation"),
        ("Applied Practitioner", "A professional with practical experience in the domain")
    ]

    while len(experts) < num_experts:
        experts.append(default_roles[len(experts) % len(default_roles)])

    return experts


def generate_expert_answer(question, role, role_description):
    """Generate an answer from a specific expert's perspective."""
    system_msg = MEP_PROMPTS["expert_answer_generation"]["system_message"]
    user_msg = MEP_PROMPTS["expert_answer_generation"]["user_message"]
    user_msg = user_msg.replace("{{ROLE}}", role)
    user_msg = user_msg.replace("{{ROLE_DESCRIPTION}}", role_description)
    user_msg = user_msg.replace("{{QUESTION}}", question)

    response = call_openai(system_msg, user_msg)
    return response


def merge_expert_answers(question, experts_answers, choices):
    """Merge answers from multiple experts and determine the final answer."""
    answer_format = generate_answer_format(experts_answers)
    answer_choices = get_answer_choices_string(choices)

    system_msg = MEP_PROMPTS["merging_prompt"]["system_message"]
    user_msg = MEP_PROMPTS["merging_prompt"]["user_message"]
    user_msg = user_msg.replace("{{QUESTION}}", question)
    user_msg = user_msg.replace("{{NUM_EXPERTS}}", str(len(experts_answers)))
    user_msg = user_msg.replace("{{ANSWER_FORMAT}}", answer_format)
    user_msg = user_msg.replace("{{ANSWER_CHOICES}}", answer_choices)

    response = call_openai(system_msg, user_msg)
    return response


def extract_final_answer(merged_response, num_choices):
    """Extract the final answer letter from the merged response."""
    valid_letters = [chr(65 + i) for i in range(num_choices)]
    valid_pattern = ''.join(valid_letters)

    # Look for "Best answer choice:" or "Final answer:" patterns
    patterns = [
        r'Best answer choice[:\s]*\[?\s*([' + valid_pattern + r'])\s*\]?',
        r'Final answer[:\s]*\[?\s*([' + valid_pattern + r'])\s*\]?',
        r'answer is[:\s]*\[?\s*([' + valid_pattern + r'])\s*\]?',
        r'\b([' + valid_pattern + r'])\)'
    ]

    for pattern in patterns:
        match = re.search(pattern, merged_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find any valid letter
    match = re.search(r'[' + valid_pattern + r']', merged_response)
    if match:
        return match.group().upper()

    return 'A'  # Default fallback


def run_mep_pipeline(item):
    """Run the full Multi-Expert Prompting pipeline for a single item."""
    question_with_choices = f"{item['question']}\n\nChoices:\n{format_choices(item['choices'])}"

    # Step 1: Generate experts
    experts = generate_experts(question_with_choices, NUM_EXPERTS)

    # Step 2: Generate answers from each expert
    experts_answers = []
    for role, description in experts:
        answer = generate_expert_answer(question_with_choices, role, description)
        experts_answers.append((role, description, answer))

    # Step 3: Merge expert answers
    merged_response = merge_expert_answers(question_with_choices, experts_answers, item['choices'])

    # Step 4: Extract final answer
    final_answer = extract_final_answer(merged_response, len(item['choices']))

    return {
        'experts': experts,
        'experts_answers': experts_answers,
        'merged_response': merged_response,
        'final_answer': final_answer
    }


def evaluate_answer(generated_answer, correct_answer):
    """Evaluate the generated answer against the correct answer."""
    gen = generated_answer.strip().upper()
    correct = correct_answer.strip().upper()

    gen_match = re.search(r'[A-Z]', gen)
    if gen_match:
        gen = gen_match.group()

    return gen == correct


# ============== EXPERIMENT RUNNER ==============

def run_experiment(dataset_name, data, output_file, summary_file, num_samples=None, start_index=0):
    """Run the MEP experiment on a dataset."""

    if num_samples is not None:
        data = data[start_index:start_index + num_samples]
    else:
        data = data[start_index:]

    all_evaluations = []
    detailed_results = []

    total_items = len(data)

    for idx, item in enumerate(data):
        print(f"\n{'=' * 60}")
        print(f"Processing item {idx + 1}/{total_items} (id: {item['id']})...")
        print(f"Dataset: {dataset_name}")
        print(f"Category: {item.get('category', 'N/A')}")
        print(f"Question: {item['question'][:100]}...")
        print(f"Correct Answer: {item['correct_answer']}")

        try:
            result = run_mep_pipeline(item)

            evaluation = evaluate_answer(result['final_answer'], item['correct_answer'])
            all_evaluations.append(evaluation)

            print(f"Generated Answer: {result['final_answer']}")
            print(f"Evaluation: {'CORRECT' if evaluation else 'INCORRECT'}")

            # Store detailed results
            row_result = {
                'id': item['id'],
                'dataset': dataset_name,
                'category': item.get('category', ''),
                'question': item['question'],
                'choices': json.dumps(item['choices']),
                'correct_answer': item['correct_answer'],
                'correct_label': item['correct_label'],
                'generated_answer': result['final_answer'],
                'evaluation': evaluation,
                'experts': json.dumps([(r, d) for r, d, _ in result['experts_answers']]),
                'merged_response': result['merged_response']
            }
            detailed_results.append(row_result)

        except Exception as e:
            print(f"ERROR: {str(e)}")
            all_evaluations.append(None)
            row_result = {
                'id': item['id'],
                'dataset': dataset_name,
                'category': item.get('category', ''),
                'question': item['question'],
                'choices': json.dumps(item['choices']),
                'correct_answer': item['correct_answer'],
                'correct_label': item['correct_label'],
                'generated_answer': f"ERROR: {str(e)}",
                'evaluation': None,
                'experts': '',
                'merged_response': ''
            }
            detailed_results.append(row_result)

        # Save intermediate results
        if (idx + 1) % 10 == 0:
            temp_df = pd.DataFrame(detailed_results)
            temp_df.to_csv(output_file.replace('.csv', '_temp.csv'), index=False)
            print(f"\n  [Saved intermediate results at item {idx + 1}]")

    # Compute statistics
    valid_evals = [e for e in all_evaluations if e is not None]
    if len(valid_evals) > 0:
        accuracy = sum(valid_evals) / len(valid_evals)
        correct_count = sum(valid_evals)
        incorrect_count = len(valid_evals) - correct_count
    else:
        accuracy = 0.0
        correct_count = 0
        incorrect_count = 0

    summary = {
        'dataset': dataset_name,
        'method': 'Multi-Expert Prompting (MEP)',
        'num_experts': NUM_EXPERTS,
        'accuracy': float(accuracy),
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'total': len(valid_evals),
        'timestamp': datetime.now().isoformat()
    }

    # Compute per-category statistics
    df = pd.DataFrame(detailed_results)
    category_stats = {}
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        valid_cat_evals = cat_df['evaluation'].dropna()
        if len(valid_cat_evals) > 0:
            category_stats[category] = {
                'accuracy': float(valid_cat_evals.sum() / len(valid_cat_evals)),
                'correct': int(valid_cat_evals.sum()),
                'total': len(valid_cat_evals)
            }

    summary['category_statistics'] = category_stats

    # Save results
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return summary, detailed_results


def print_dataset_summary(summary):
    """Print summary for a single dataset."""
    print("\n" + "-" * 60)
    print(f"Dataset: {summary['dataset']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Correct: {summary['correct_count']}")
    print(f"Incorrect: {summary['incorrect_count']}")
    print(f"Total: {summary['total']}")

    if summary.get('category_statistics'):
        print(f"\nAccuracy by Category:")
        for cat, stats in sorted(summary['category_statistics'].items()):
            print(f"  {cat[:40]:<40} {stats['accuracy']:.0%} (n={stats['total']})")


def print_final_report(all_summaries):
    """Print final report comparing all datasets."""
    print("\n" + "=" * 80)
    print("MULTI-EXPERT PROMPTING - FINAL SUMMARY REPORT")
    print("=" * 80)

    print(f"\nMethod: Multi-Expert Prompting with {NUM_EXPERTS} experts")
    print(f"Model: gpt-3.5-turbo")
    print(f"Timestamp: {datetime.now().isoformat()}")

    print("\n" + "-" * 80)
    print(f"{'Dataset':<20} {'Accuracy':>12} {'Correct':>10} {'Incorrect':>10} {'Total':>10}")
    print("-" * 80)

    total_correct = 0
    total_incorrect = 0
    total_samples = 0

    for summary in all_summaries:
        print(
            f"{summary['dataset']:<20} {summary['accuracy']:>12.2%} {summary['correct_count']:>10} {summary['incorrect_count']:>10} {summary['total']:>10}")
        total_correct += summary['correct_count']
        total_incorrect += summary['incorrect_count']
        total_samples += summary['total']

    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        print("-" * 80)
        print(
            f"{'OVERALL':<20} {overall_accuracy:>12.2%} {total_correct:>10} {total_incorrect:>10} {total_samples:>10}")

    print("=" * 80)


def run_all_datasets(args):
    """Run MEP on the selected datasets sequentially."""

    all_summaries = []
    all_results = []

    datasets_to_run = args.datasets

    for dataset_name in datasets_to_run:
        config = DATASET_CONFIGS[dataset_name]

        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {config['name']} ({config['description']})")
        print("=" * 80)

        # Determine input path
        input_path = getattr(args, f'{dataset_name}_input', None) or config['default_input']

        # Check if input exists
        if dataset_name == 'bbq':
            if not os.path.isdir(input_path):
                print(f"WARNING: BBQ folder '{input_path}' not found. Skipping...")
                continue
        elif dataset_name == 'creak':
            # CREAK can load from HuggingFace, so don't skip if path doesn't exist
            pass
        else:
            if not os.path.isfile(input_path):
                print(f"WARNING: Input file '{input_path}' not found. Skipping...")
                continue

        # Load data
        try:
            if dataset_name == 'mafalda':
                data = load_mafalda_data(input_path)
            elif dataset_name == 'mmlu':
                data = load_mmlu_data(input_path, sample_fraction=args.mmlu_fraction)
            elif dataset_name == 'bbq':
                data = load_bbq_data(input_path, samples_per_category=args.bbq_samples_per_category)
            elif dataset_name == 'crowspairs':
                data = load_crowspairs_data(input_path, sample_fraction=args.crowspairs_fraction)
            elif dataset_name == 'creak':
                data = load_creak_data(input_path, sample_fraction=args.creak_fraction)
        except Exception as e:
            print(f"ERROR loading {dataset_name}: {str(e)}. Skipping...")
            continue

        if not data:
            print(f"WARNING: No data loaded for {dataset_name}. Skipping...")
            continue

        # Determine output paths
        output_file = f'results/{config["output_prefix"]}_mep_results.csv'
        summary_file = f'results/{config["output_prefix"]}_mep_summary.json'

        # Run experiment
        summary, results = run_experiment(
            dataset_name=config['name'],
            data=data,
            output_file=output_file,
            summary_file=summary_file,
            num_samples=args.samples_per_dataset,
            start_index=0
        )

        all_summaries.append(summary)
        all_results.extend(results)

        # Print dataset summary
        print_dataset_summary(summary)

    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv('results/all_datasets_mep_results.csv', index=False)
        print(f"\nCombined results saved to: all_datasets_mep_results.csv")

    # Save combined summary
    if all_summaries:
        combined_summary = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Multi-Expert Prompting (MEP)',
            'num_experts': NUM_EXPERTS,
            'datasets': all_summaries
        }
        with open('results/all_datasets_mep_summary.json', 'w') as f:
            json.dump(combined_summary, f, indent=2)
        print(f"Combined summary saved to: all_datasets_mep_summary.json")

    # Print final report
    print_final_report(all_summaries)

    return all_summaries


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Expert Prompting for MAFALDA, MMLU, BBQ, CrowS-Pairs, and CREAK datasets')

    # General arguments
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS),
                        default=['creak', 'mmlu', 'bbq', 'crowspairs'],
                        help='Datasets to run (default: the four extended benchmarks in the paper)')
    parser.add_argument('--samples_per_dataset', type=int, default=None,
                        help='Number of samples to process per dataset (default: all)')

    # Dataset-specific input paths
    parser.add_argument('--mafalda_input', type=str, default='data/gold_standard_dataset.jsonl',
                        help='Input file for MAFALDA')
    parser.add_argument('--mmlu_input', type=str, default='data/mmlu_test.csv',
                        help='Input file for MMLU')
    parser.add_argument('--bbq_input', type=str, default='data/bbq',
                        help='Input folder for BBQ')
    parser.add_argument('--crowspairs_input', type=str, default='data/crows_pairs_anonymized.csv',
                        help='Input file for CrowS-Pairs')
    parser.add_argument('--creak_input', type=str, default='huggingface',
                        help='Input source for CREAK (folder path or "huggingface")')

    # Sampling parameters
    parser.add_argument('--mmlu_fraction', type=float, default=0.05,
                        help='Sample fraction for MMLU (default: 0.05)')
    parser.add_argument('--bbq_samples_per_category', type=int, default=50,
                        help='Samples per category for BBQ (default: 50)')
    parser.add_argument('--crowspairs_fraction', type=float, default=1 / 3,
                        help='Sample fraction for CrowS-Pairs (default: 1/3)')
    parser.add_argument('--creak_fraction', type=float, default=0.02,
                        help='Sample fraction for CREAK (default: 0.02)')

    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-EXPERT PROMPTING - RUNNING ALL DATASETS")
    print("=" * 80)
    print(f"\nNumber of experts: {NUM_EXPERTS}")
    print(f"Samples per dataset: {'All' if args.samples_per_dataset is None else args.samples_per_dataset}")
    print(f"\nDataset configurations:")
    print(f"  - MAFALDA: {args.mafalda_input}")
    print(f"  - MMLU: {args.mmlu_input} (fraction: {args.mmlu_fraction})")
    print(f"  - BBQ: {args.bbq_input} (samples/category: {args.bbq_samples_per_category})")
    print(f"  - CrowS-Pairs: {args.crowspairs_input} (fraction: {args.crowspairs_fraction:.1%})")
    print(f"  - CREAK: {args.creak_input} (fraction: {args.creak_fraction:.1%})")

    # Run all datasets
    all_summaries = run_all_datasets(args)

    print("\n" + "=" * 80)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("=" * 80)

    return all_summaries


if __name__ == "__main__":
    main()