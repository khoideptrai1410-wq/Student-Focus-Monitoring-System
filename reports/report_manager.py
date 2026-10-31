import json
import os
from datetime import datetime


class ReportManager:

    def __init__(self):

        os.makedirs(
            "reports/output",
            exist_ok=True
        )

    def save_report(
        self,
        student_info,
        logs,
        result
    ):

        report = {

            "student": {

                "name": student_info["name"],
                "class": student_info["class"],
                "sbd": student_info["sbd"]

            },

            "result": result,

            "created_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "logs": logs

        }

        filename = os.path.join(

            "reports/output",

            f"{student_info['sbd']}.json"

        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(
            f"[REPORT SAVED] {filename}"
        )

        return filename