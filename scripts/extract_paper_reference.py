"""Extract displayed vector curves from the supplied arXiv v5 PDF.

This is figure digitization, not recovery of the authors' simulation dataset.
Calibrations below are tied to axis tick locations in this specific document.
Requires pdfplumber; the reproduction itself uses the saved CSV references.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import pdfplumber


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/paper_reference"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(args.pdf) as document:
        page = document.pages[3]
        curves = [
            c
            for c in page.curves
            if c["width"] > 140 and c["x0"] > 350 and len(c["pts"]) == 16 and c["top"] < 180
        ]
        assert len(curves) == 4, "Unexpected Figure 2 vector layout"
        rows = []
        for color, quantity in [
            ((0.0, 0.0, 1.0), "range_rcrb_m"),
            ((1.0, 0.0, 0.0), "angle_rcrb_deg"),
        ]:
            series = sorted(
                [c for c in curves if c["stroking_color"] == color],
                key=lambda c: c["pts"][0][1],
                reverse=True,
            )
            for architecture, curve in zip(
                ["fully-digital-sdr", "hybrid-two-stage-sdr"], series, strict=True
            ):
                for x, y in curve["pts"]:
                    rate = round((x - 365.1904) / (503.4373528 - 365.1904) * 10, 3)
                    exponent = (
                        -2 + (141.723892 - y) / (141.723892 - 66.6372)
                        if quantity == "range_rcrb_m"
                        else -4 + (128.4374992 - y) / (128.4374992 - 75.8901984)
                    )
                    rows.append(
                        dict(
                            minimum_rate=rate,
                            architecture=architecture,
                            quantity=quantity,
                            value=10**exponent,
                        )
                    )
        write_csv(args.output / "figure2.csv", rows)
        page = document.pages[4]
        curves = [
            c
            for c in page.curves
            if c["width"] > 140 and c["x0"] < 110 and c["top"] < 200 and len(c["pts"]) in [8, 12]
        ]
        assert len(curves) == 4, "Unexpected Figure 4 vector layout"
        rows = []
        for c in curves:
            color = c["stroking_color"]
            quantity = "range_rcrb_m" if color == (0.0, 0.0, 1.0) else "angle_rcrb_deg"
            architecture = (
                ("fully-digital-sdr" if c["pts"][0][1] > 105 else "hybrid-two-stage-sdr")
                if quantity == "range_rcrb_m"
                else ("fully-digital-sdr" if color == (1.0, 0.0, 0.0) else "hybrid-two-stage-sdr")
            )
            for x, y in c["pts"]:
                distance = round(5 + (x - 102.1764) / (253.3876 - 102.1764) * 35, 3)
                if quantity == "range_rcrb_m":
                    value = 10 ** (-4 + (115.2408 - y) / (115.2408 - 89.2150984) * 2)
                elif architecture == "fully-digital-sdr":
                    value = 3.45e-5 + (177.30614144 - y) / (177.30614144 - 134.7292) * 4e-7
                else:
                    value = 1.8e-4 + (175.25915504 - y) / (175.25915504 - 145.7831144) * 4e-5
                assert math.isfinite(value) and value > 0
                rows.append(
                    dict(
                        distance_m=distance,
                        architecture=architecture,
                        quantity=quantity,
                        value=value,
                    )
                )
        write_csv(args.output / "figure4.csv", rows)
        fd_far_y, hb_far_y = 176.77070352, 161.15575216
        metadata = {
            "source": "Near-Field ISAC.pdf, arXiv:2302.01153v5, 23 September 2025",
            "source_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            "method": "Vector path extraction with linear/logarithmic tick calibration",
            "limitations": "Displayed plot coordinates, not raw author data; allow rounding error.",
            "figure3_estimated_range_m": 19.952,
            "figure3_estimated_angle_deg": 45.0,
            "figure4_fd_far_angle_deg": 3.45e-5
            + (177.30614144 - fd_far_y) / (177.30614144 - 134.7292) * 4e-7,
            "figure4_hb_far_angle_deg": 1.8e-4
            + (175.25915504 - hb_far_y) / (175.25915504 - 145.7831144) * 4e-5,
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Extracted Figures 2 and 4 to {args.output}")


if __name__ == "__main__":
    main()
