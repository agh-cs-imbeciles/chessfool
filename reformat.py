import csv

def process_csv(input_file, output_file):
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

    max_val = max(second_col_values)
    min_val = min(second_col_values)

    # Modify rows as per requirement
    for row in rows:
        if len(row) > 1 and row[1].startswith('#'):
            try:
                num = int(row[1][1:])
                if num < 0:
                    row[1] = str(min_val)
                else:
                    row[1] = str(max_val)
            except ValueError:
                pass  # Ignore if not a valid number after '#'
        if len(row) > 1 and len(row[0].split()) > 1 and row[0].split()[1][0] == 'b':
            try:
                val = float(row[1])
                row[1] = str(-val)
                row[0] = row[0].replace(' b', ' w', 1)
            except ValueError:
                pass  # Ignore non-numeric values
    # Write modified rows to output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

# Example usage:
process_csv('data/evals/chessData.csv', 'data/evals_trans/chessData.csv')