
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_PATH = os.path.join(ROOT_PATH, "examples")
SIMPLIFIED_IOTOOLS_PATH = os.path.join(ROOT_PATH, "simplified_iotools")


def _help_default(extension):
    examples_path = os.path.join(EXAMPLES_PATH, "examples_{}.py".format(extension))
    if "_" in extension:
        ext, using = extension.split("_", 1)
        simplified_iotools_path = os.path.join(SIMPLIFIED_IOTOOLS_PATH, ext + "io_" + using + ".py")
    else:
        simplified_iotools_path = os.path.join(SIMPLIFIED_IOTOOLS_PATH, extension + "io.py")
    with open(examples_path, "r") as f:
        example_code = f.read()
    with open(simplified_iotools_path, "r") as f:
        simplified_code = f.read()
    print("\n*** iotools simplified code ***\n\n{}\n".format(simplified_code))
    print("\n*** Examples ***\n\n{}\n".format(example_code))


def _help_multi_packages(extension, other_packages):
    _help_default(extension)
    print(
        "*** Misc ***\n\nOther packages for image io may be available,",
        "check help_{}_using_[{}]() for more info.".format(extension, other_packages)
    )
