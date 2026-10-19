"""Run the walkthrough's plain Python cells and capture standard notebook outputs."""

import argparse
import base64
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


def execute(path, write=False):
    book = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    original = plt.show
    try:
        for index, cell in enumerate(book.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            outputs = []
            stream = io.StringIO()

            def show(*_args, outputs=outputs, **_kwargs):
                for number in plt.get_fignums():
                    figure = plt.figure(number)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
                    outputs.append(
                        nbformat.v4.new_output(
                            "display_data",
                            data={
                                "image/png": base64.b64encode(buffer.getvalue()).decode(),
                                "text/plain": "<Matplotlib figure>",
                            },
                            metadata={},
                        )
                    )
                    plt.close(figure)

            plt.show = show
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, f"{path}:cell-{index}", "exec"), namespace)
            if stream.getvalue():
                outputs.append(
                    nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue())
                )
            cell.outputs, cell.execution_count = outputs, count
        nbformat.validate(book)
        if write:
            nbformat.write(book, path)
        print(f"Executed {count} Python cells; figures captured; notebook schema valid.")
        return count
    finally:
        plt.show = original
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, default=Path("notebooks/01_synthetic_unmixing.ipynb")
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    execute(args.path, args.write)
