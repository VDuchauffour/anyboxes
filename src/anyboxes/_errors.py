class MissingToMethodError(RuntimeError):
    """An error to handle case where to methods hasn't been called."""

    def __str__(self):
        """Error message to print."""
        return (
            "One of the method `to_top_left_corner`, `to_bottom_left_corner`,"
            " `to_two_corners` or `to_center` must be called."
        )


class OptionalDependencyImportError(ModuleNotFoundError):
    """An error to handle case where an optional dependency is missing."""

    def __init__(self, dependency: str):
        """Instantiate a MissingImportError.

        Args:
            dependency (str): Missing dependency.
        """
        self.dependency = dependency

    def __str__(self):
        """Error message to print."""
        return (
            f'The optional dependency "{self.dependency}" is missing, please consider'
            " to reinstall the project with the command `pip install"
            f' "anyboxes[{self.dependency}]"`. You may need also to combine it with'
            " others optional dependencies, you can use the command `pip install"
            ' "anyboxes[all]". for installing all optional dependencies.'
        )
