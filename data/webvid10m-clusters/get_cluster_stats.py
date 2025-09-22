import json

def analyze_json_clusters(json_data):
    """
    Analyzes a JSON string to count clusters (keys) and total strings.

    Args:
        json_data (str): A string containing the JSON data in the format
                         {"key": [list of strings], ...}.

    Returns:
        tuple: A tuple containing (total_clusters, total_strings).
               Returns (0, 0) on error.
    """
    try:
        data = json.loads(json_data)
        total_clusters = len(data)
        total_strings = sum(len(string_list) for string_list in data.values())
        return total_clusters, total_strings
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in the file.")
        return 0, 0
    except Exception as e:
        print(f"An unexpected error occurred during analysis: {e}")
        return 0, 0

# --- Main execution block ---
if __name__ == "__main__":
    # Define the filename (whitespace is stripped to prevent errors)
    filename = "data/webvid10m-clusters/clusters-full.json".strip()

    try:
        # Open the file, read its content, and ensure it's closed properly
        with open(filename, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Analyze the content read from the file
        clusters, strings = analyze_json_clusters(file_content)

        # Print the results only if analysis was successful
        if clusters > 0 or strings > 0:
            print(f"Analysis for file: '{filename}'")
            print(f"Total number of clusters: {clusters:,}")
            print(f"Total number of strings:  {strings:,}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")