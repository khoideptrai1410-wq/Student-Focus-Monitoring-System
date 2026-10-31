import csv
import os

class ReportGenerator:

    def __init__(self):
        os.makedirs("reports", exist_ok=True)

    def generate(self, report):

        filename = report["filename"]

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(["Metric", "Value"])

            for key, value in report.items():

                if key == "filename":
                    continue

                writer.writerow([key, value])

        return filename