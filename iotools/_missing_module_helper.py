"""
The module helps printing a comprehensible message for users who
did not install required libraries for their iotools usage.
"""


def _error_msg_one(*args):
    return "The module '{}' requires the library '{}' which is not installed.".format(*args)


def _error_msg_sev(*args):
    msg = "The module '{}' requires to install at least one of the following backends:" \
          "\n - pip install {}  # (recommended)" \
          "\n - pip install {}"
    args = list(args)
    for i, arg in enumerate(args):
        if not isinstance(arg, str):
            args[i] = " ".join(arg)
    return msg.format(args[0], args[1], "\n - pip install ".join(args[2:]))


def _warning_msg_default_one(*args):
    msg = "Warning: Function '{}' does not have the default behavior, " \
          "because library '{}' is not installed." \
          "\nPlease install it or use function '{}' instead."
    return msg.format(*args)


def _warning_msg_default_sev(func1, libs, func2):
    msg = "Warning: Function '{}' does not have the default behavior, " \
          "because libraries '{}' and '{}' are not installed." \
          "\nPlease install them or use function '{}' instead."
    return msg.format(func1, "', '".join(libs[:-1]), libs[-1], func2)


class _EmptyModule:

    def __init__(self, name, *other):
        self._name = name
        self._other = other

    def __call__(self, *args, **kwargs):
        if len(self._other) == 0:
            msg = "Module '{}' is required but not installed.\n(pip install {})"
            raise ModuleNotFoundError(msg.format(self._name, self._name))
        msg = "Modules '{}' and '{}' are required but not installed.\n(pip install {})"
        modules = [self._name] + list(self._other)
        raise ModuleNotFoundError(msg.format(", ".join(modules[:-1]), modules[-1], " ".join(modules)))

    def __getattr__(self, x):
        if len(self._other) == 0:
            msg = "Module '{}' is required but not installed.\n(pip install {})"
            raise ModuleNotFoundError(msg.format(self._name, self._name))
        msg = "Modules '{}' and '{}' are required but not installed.\n(pip install {})"
        modules = [self._name] + list(self._other)
        raise ModuleNotFoundError(msg.format(", ".join(modules[:-1]), modules[-1], " ".join(modules)))
