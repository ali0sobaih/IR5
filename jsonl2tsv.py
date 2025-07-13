import json
import os

def convert_jsonl_to_tsv(input_jsonl_path, output_tsv_path):
    """
    Converts a JSONL file containing query IDs and text to a TSV file.

    Args:
        input_jsonl_path (str): Path to the input .jsonl file.
        output_tsv_path (str): Path for the output .tsv file.
    """
    if not os.path.exists(input_jsonl_path):
        print(f"Error: Input file not found at '{input_jsonl_path}'")
        return

    print(f"Converting '{input_jsonl_path}' to '{output_tsv_path}'...")
    
    lines_processed = 0
    lines_converted = 0

    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
             open(output_tsv_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                lines_processed += 1
                line = line.strip()
                if not line:
                    continue # Skip empty lines

                try:
                    data = json.loads(line)
                    query_id = str(data.get("_id", "")) # Get ID, ensure string
                    query_text = data.get("text", "") # Get text

                    if query_id and query_text:
                        # Write ID and text separated by a tab
                        outfile.write(f"{query_id}\t{query_text}\n")
                        lines_converted += 1
                    else:
                        print(f"Warning: Skipping malformed line (missing '_id' or 'text') in '{input_jsonl_path}' at line {lines_processed}: {line[:100]}...")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in '{input_jsonl_path}' at line {lines_processed}: {line[:100]}...")
                except Exception as e:
                    print(f"An unexpected error occurred processing line {lines_processed}: {e}. Line content: {line[:100]}...")

        print(f"Conversion complete! Processed {lines_processed} lines, converted {lines_converted} queries.")
        print(f"Output saved to: {output_tsv_path}")

    except IOError as e:
        print(f"Error reading or writing files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your input and output file paths
    # Assuming queries.jsonl is in data/quora/
    input_file = "data/quora/queries.jsonl"
    output_file = "data/quora/queries.tsv" # New TSV file name

    convert_jsonl_to_tsv(input_file, output_file)

    print("\nAfter running this script, remember to update EVAL_DATASETS_INFO:")
    print("Change 'queries_path' to 'data/quora/queries.tsv' and 'query_format' to 'tsv'.")
    print("Example:")
    print("""
EVAL_DATASETS_INFO = {
    "antique": { 
        "queries_path": "data/antique/queries.txt",
        "qrels_path": "data/antique/qrels.tsv",
        "query_format": "tsv"
    },
    "quora": { 
        "queries_path": "data/quora/queries.tsv", # Updated path
        "qrels_path": "data/quora/qrels/test.tsv", 
        "query_format": "tsv" # Updated format
    }
}
""")