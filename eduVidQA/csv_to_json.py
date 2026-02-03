import csv
import json
import os

def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Converts a CSV file to a JSON file.
    Each row in the CSV becomes an object in a JSON array.
    """
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_input_path = os.path.join(script_dir, "synthetic_train.csv")
    json_output_path = os.path.join(script_dir, "synthetic_train.json")
    convert_csv_to_json(csv_input_path, json_output_path)
    print(f"Successfully converted '{csv_input_path}' to '{json_output_path}'")