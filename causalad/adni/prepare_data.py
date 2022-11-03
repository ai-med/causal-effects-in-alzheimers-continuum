# This file is part of Estimation of Causal Effects in the Alzheimer's Continuum (Causal-AD).
#
# Causal-AD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Causal-AD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Causal-AD. If not, see <https://www.gnu.org/licenses/>.
import argparse
import logging
from pathlib import Path

import papermill as pm


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adni_csv", required=True, help="Path to CSV with FreeSurfer features.")
    parser.add_argument("--outputs_dir", required=True, help="Path to directory where to write outputs to.")

    args = parser.parse_args(args=args)

    output_dir = Path(args.outputs_dir)
    parameters = {
        "adni_csv_file": args.adni_csv,
        "output_dir": str(output_dir.resolve()),
    }

    pm.execute_notebook(
        Path(__file__).parent / "notebooks" / "prepare_data.ipynb",
        output_dir / "adni_prepare_data_outputs.ipynb",
        parameters=parameters,
        progress_bar=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
