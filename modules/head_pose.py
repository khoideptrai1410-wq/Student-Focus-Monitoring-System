class HeadPose:

    def __init__(self):

        self.prev_yaw = 0.0
        self.alpha = 0.3

    def smooth(self, current, previous):

        return previous * (1 - self.alpha) + current * self.alpha

    def get_direction(self, frame, face, landmarks):

        if landmarks is None:

            return (
                0.0,
                0.0,
                0.0,
                "Forward"
            )

        try:

            nose = landmarks[1]

            left_eye = landmarks[33]

            right_eye = landmarks[263]

            eye_center_x = (
                left_eye.x +
                right_eye.x
            ) / 2

            yaw = (
                nose.x -
                eye_center_x
            ) * 100

            yaw = self.smooth(
                yaw,
                self.prev_yaw
            )

            self.prev_yaw = yaw

            direction = "Forward"

            if yaw < -4:

                direction = "Left"

            elif yaw > 4:

                direction = "Right"

            return (

                0.0,

                round(yaw, 2),

                0.0,

                direction

            )

        except Exception:

            return (

                0.0,

                0.0,

                0.0,

                "Forward"

            )