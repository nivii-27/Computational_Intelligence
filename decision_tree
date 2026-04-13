import csv
import math

def display_raw_dataset(filename):
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = [row for row in reader if row and len(row) == len(headers)]

            print("\n" + "="*75)
            print("--- DATASET---")
            print("="*75)

            fmt = "{:<12} | {:<10} | {:<8} | {:<15} | {:<10}"
            print(fmt.format(*headers))
            print("-" * 75)

            for row in data:
                print(fmt.format(*row))
            print("="*75 + "\n")
    except FileNotFoundError:
        print(f"Error: {filename} not found.")

def calculate_entropy_with_steps(data, target_col):
    if not data: return 0, "0"
    total = len(data)
    counts = {}
    for row in data:
        val = row[target_col]
        counts[val] = counts.get(val, 0) + 1

    ent = 0
    steps = []
    for val, count in counts.items():
        p = count / total
        term = p * math.log2(p)
        ent -= term
        steps.append(f"({count}/{total}*log2({count}/{total}))")

    formula = "-[" + " + ".join(steps) + "]" if steps else "0"
    return ent, formula

def get_class_counts(subset, target_col):
    classes = list(set(row[target_col] for row in subset))
    counts = {cls: sum(1 for row in subset if row[target_col] == cls) for cls in classes}
    yes = counts.get('Yes', 0)
    no = counts.get('No', 0)
    return yes, no

def process_and_display(data, feature, target, parent_entropy):
    subsets = {}
    for row in data:
        val = row[feature]
        if val not in subsets: subsets[val] = []
        subsets[val].append(row)

    print(f"\n[ STEP-BY-STEP CALCULATION FOR: {feature.upper()} ]")
    header = f"{'Value':<12} | {'Total':<6} | {'Yes':<4} | {'No':<4} | {'Entropy Calculation':<40} | {'Result':<8}"
    print(header)
    print("-" * len(header))

    weighted_entropy_sum = 0
    total_rows = len(data)
    calculation_steps = []

    for val, subset in subsets.items():
        count = len(subset)
        yes, no = get_class_counts(subset, target)
        ent, formula = calculate_entropy_with_steps(subset, target)

        weight = count / total_rows
        weighted_entropy_sum += (weight * ent)
        calculation_steps.append(f"({count}/{total_rows} * {ent:.4f})")

        print(f"{val:<12} | {count:<6} | {yes:<4} | {no:<4} | {formula:<40} | {ent:<8.4f}")

    gain = parent_entropy - weighted_entropy_sum

    print("-" * len(header))
    print(f"Total Weighted Entropy = {' + '.join(calculation_steps)}")
    print(f"                       = {weighted_entropy_sum:.4f}")
    print(f"Information Gain       = {parent_entropy:.4f} - {weighted_entropy_sum:.4f}")
    print(f"                       = {gain:.4f}")

    return gain, weighted_entropy_sum

# --- MAIN EXECUTION ---
filename = 'dataset1.csv'
display_raw_dataset(filename)

try:
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if any(row.values())]
        headers = reader.fieldnames

    if not headers:
        raise ValueError("CSV file is empty or headers are missing.")

    target = headers[-1]
    features = headers[:-1]

    print("--- FREQUENCY DISTRIBUTION ---")
    for col in headers:
        dist = {}
        for r in rows:
            val = r[col]
            dist[val] = dist.get(val, 0) + 1
        print(f"{col:<15}: {dist}")

    parent_ent, parent_formula = calculate_entropy_with_steps(rows, target)
    print(f"\nINITIAL SYSTEM ENTROPY: {parent_formula} = {parent_ent:.4f}")

    # Track results for final table
    summary_data = []
    gains = {}

    for feat in features:
        gain, weighted_ent = process_and_display(rows, feat, target, parent_ent)
        gains[feat] = gain
        summary_data.append([feat, parent_ent, weighted_ent, gain])

    # --- FINAL TABULAR SUMMARY ---
    print("\n" + "="*85)
    print("--- INFORMATION GAIN TABLE ---")
    print("="*85)
    print(f"{'Feature':<15} | {'Parent Entropy':<15} | {'Weighted Entropy':<18} | {'Info Gain':<10}")
    print("-" * 85)
    for row in summary_data:
        print(f"{row[0]:<15} | {row[1]:<15.4f} | {row[2]:<18.4f} | {row[3]:<10.4f}")
    print("="*85)

    root_node = max(gains, key=gains.get)
    print("\n" + "*"*55)
    print(f"WINNER: THE ROOT NODE IS '{root_node.upper()}'")
    print(f"HIGHEST INFORMATION GAIN: {gains[root_node]:.4f}")
    print("*"*55)

except Exception as e:
    print(f"An error occurred: {e}")
