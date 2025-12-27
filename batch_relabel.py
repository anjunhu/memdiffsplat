import argparse
import glob
import json
import os

def update_memorized_field(file_path: str, set_to_true: bool):
    """
    Reads a JSON file, updates or adds the 'memorized' field, and writes it back.

    Args:
        file_path (str): The path to the JSON file.
        set_to_true (bool): The boolean value to set for the 'memorized' field.
    """
    try:
        with open(file_path, 'r') as f:
            # Load the JSON content into a Python dictionary
            data = json.load(f)

        # Update or add the 'memorized' field
        # This is simpler and safer than regex
        if data.get("memorized") == set_to_true:
            print(f"Skipped (already set): {file_path}")
            return

        data["memorized"] = set_to_true

        # Write the modified dictionary back to the file with indentation
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        status = "true" if set_to_true else "false"
        print(f"Updated '{os.path.basename(file_path)}' -> memorized: {status}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    except IOError as e:
        print(f"Error processing file {file_path}: {e}")


def main():
    """
    Main function to parse arguments and process files.
    """
    parser = argparse.ArgumentParser(
        description="Batch update the 'memorized' field in JSON files.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "action",
        choices=["true", "false"],
        help="The action to perform:\n"
             "  true:  Set 'memorized' field to true (adds it if missing).\n"
             "  false: Set 'memorized' field to false (adds it if missing)."
    )
    parser.add_argument(
        "path_pattern",
        type=str,
        help="Path pattern to match JSON files (e.g., './output/**/*.json').\n"
             "Use quotes to prevent shell expansion."
    )

    args = parser.parse_args()

    # Use recursive=True to search in subdirectories with '**'
    json_files = glob.glob(args.path_pattern, recursive=True)

    if not json_files:
        print(f"No files found matching pattern: {args.path_pattern}")
        return

    print(f"Found {len(json_files)} files to process.")

    # Determine the boolean value based on the command-line action
    label_value = args.action == "true"

    for file_path in json_files:
        update_memorized_field(file_path, label_value)

if __name__ == "__main__":
    main()
    # python batch_relabel.py false "./output/baseline/laion_memorized/*.json"
    # python batch_relabel.py true "./output/baseline/cap3d/*.json"