import argparse
import sys
from rag_rater import process_csv_file

def main():
    """
    Запуск оценщика нашей системы.
    """
    parser = argparse.ArgumentParser(description='Evaluate RAG system performance')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--api-key', help='Google API key (optional if set in .env)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of rows to process before saving results (default: 10)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (process only one row)')
    
    args = parser.parse_args()
    
    try:
        process_csv_file(
            args.input,
            args.output,
            args.api_key,
            test_mode=args.test,
            batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 