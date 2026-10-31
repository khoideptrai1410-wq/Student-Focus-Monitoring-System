"""
Feature Extraction Module
-------------------------
Extract feature vector for pattern classification.
"""


class FeatureExtractor:

    def extract(
        self,
        face_present,
        phone_detected,
        gaze_direction,
        head_direction,
        eyes_closed,
        yaw
    ):

        if not face_present:
            gaze_direction = "NO_FACE"
            head_direction = "NO_FACE"
            yaw = 0.0

        return {
            "face_present": face_present,
            "phone_detected": phone_detected,
            "gaze_direction": gaze_direction,
            "head_direction": head_direction,
            "eyes_closed": eyes_closed,
            "yaw": round(float(yaw), 2)
        }