

class ProgramExited(Exception):
    """Raised when the executed program calls for an exit."""
    pass

class FutureError(Exception):
    """Raised when an operation cannot be completed because it depends on a future event."""
    pass