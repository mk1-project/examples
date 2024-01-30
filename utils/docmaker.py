import os
import argparse
from pathlib import Path


def parse_and_write_markdown(file_path, output_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    if len(lines) == 0:
        return

    with open(output_path, "w") as output_file:
        was_markdown = True

        for line in lines:
            is_markdown = line.startswith("##")
            is_empty = line == "##\n"

            code_starts = was_markdown and not is_markdown
            code_ends = not was_markdown and is_markdown

            if code_starts:
                output_file.write("\n```python")
                if not line.startswith("\n"):
                    output_file.write("\n")

            if code_ends:
                output_file.write("```\n")

            if is_markdown:
                output_file.write(line[3:])
            else:
                output_file.write(line)

            if is_empty and is_markdown:
                output_file.write("\n")

            was_markdown = is_markdown

        if not was_markdown:
            output_file.write("```\n")


def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_dir)
                output_file_path = Path(output_dir) / relative_path
                output_file_path = output_file_path.with_suffix(".md")

                output_file_path.parent.mkdir(parents=True, exist_ok=True)

                parse_and_write_markdown(input_file_path, output_file_path)


def main():
    parser = argparse.ArgumentParser(description="Convert Python files to Markdown")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input directory containing .py files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output directory for the generated Markdown files",
    )

    args = parser.parse_args()
    process_directory(args.input, args.output)


if __name__ == "__main__":
    main()
