class VolleySenseError(Exception):
    """Base error for VolleySense."""


class PluginError(VolleySenseError):
    pass


class ValidationError(VolleySenseError):
    pass


class NotFoundError(VolleySenseError):
    pass
