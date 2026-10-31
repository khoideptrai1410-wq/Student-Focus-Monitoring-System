"""
Focus Analyzer
--------------
Pattern Classification Module
"""

class FocusAnalyzer:

    def __init__(self):

        self.last_status = "Focused"

    def classify(self, features):

        """
        features = {
            face_present,
            phone_detected,
            gaze_direction,
            head_direction,
            eyes_closed
        }
        """

        # ===========================
        # No face
        # ===========================

        if not features["face_present"]:

            self.last_status = "No Face"

            return self.last_status

        # ===========================
        # Phone
        # ===========================

        if features["phone_detected"]:

            self.last_status = "Phone Detected"

            return self.last_status

        # ===========================
        # Eyes Closed
        # ===========================

        if features["eyes_closed"]:

            self.last_status = "Eyes Closed"

            return self.last_status

        # ===========================
        # Gaze
        # ===========================

        if features["gaze_direction"] == "DOWN":

            self.last_status = "Looking Down"

            return self.last_status

        if features["gaze_direction"] == "LEFT":

            self.last_status = "Looking Left"

            return self.last_status

        if features["gaze_direction"] == "RIGHT":

            self.last_status = "Looking Right"

            return self.last_status

        # ===========================
        # Head
        # ===========================

        if features["head_direction"] == "Left":

            self.last_status = "Head Left"

            return self.last_status

        if features["head_direction"] == "Right":

            self.last_status = "Head Right"

            return self.last_status

        # ===========================
        # Focused
        # ===========================

        self.last_status = "Focused"

        return self.last_status