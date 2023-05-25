class MissingToMethodError(RuntimeError):
    """An error to handle case where to methods hasn't been called."""

    def str(self):
        return (
            "One of the method `to_center`, `to_top_left_corner`,"
            " `to_bottom_left_corner` or `to_two_corners` must be called."
        )
