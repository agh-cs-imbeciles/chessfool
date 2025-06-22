import csv
import random

def process_csv(input_file, output_file, max_lines=None):
    if max_lines is not None:
        random.seed(42)
    rows = []
    second_col_values = []

    # Read CSV and collect second column values (excluding those starting with '#')
    with open(input_file, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        for row in reader:
            rows.append(row)
            if len(row) > 1 and not row[1].startswith('#'):
                try:
                    val = float(row[1])
                    second_col_values.append(val)
                except ValueError:
                    pass  # Ignore non-numeric values

    if not second_col_values:
        print("No valid numeric values found in second column.")
        return

    proper_rows = []
    # Modify rows as per requirement
    for row in rows:
        if len(row) > 1 and row[1].startswith('#'):
            try:
                num = int(row[1][1:])
                if num < 0:
                    row[1] = str(num)
                else:
                    row[1] = str(num)
            except ValueError:
                pass  # Ignore if not a valid number after '#'
        if len(row) > 1 and len(row[0].split()) > 1 and row[0].split()[1][0] == 'b':
            proper_rows.append(row)

    # Write modified rows to output file
    if max_lines is not None:
        proper_rows = proper_rows[:max_lines]
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(proper_rows)

# Example usage:
process_csv('data/evals/chessData.csv', 'data/evals_trans/chessData.csv',4300000)