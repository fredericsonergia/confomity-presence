import argparse
import sys

from generate_images import Datagenerator
from style_transfer.style_transfer import transfer_random_style_folder


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1 :]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


class generator:
    def __init__(datagenerator, number_of_ok, number_of_ko):
        self.d = datagenerator
        self.number_of_ok = number_of_ok
        self.number_of_ko = number_of_ko

    def generate_train_set():
        self.d.create_folder()
        self.d.generate_folder(number_of_ok, number_of_ko)
        self.d.create_train_set(0.3)

    def generate_test_set():
        self.d.create_folder()
        self.d.generate_folder(number_of_ok, number_of_ko)
        self.d.create_test_set()


if __name__ == "__main__":
    parser = ArgumentParserForBlender()

    parser.add_argument("-a", "--action", help="Select the function you want to run")
    parser.add_argument(
        "-r",
        "--rootfolder",
        default=".",
        help="root folder where the VOC2021 folder will be / is, default is .",
    )
    parser.add_argument(
        "-y",
        "--ok",
        type=int,
        default=200,
        help="number of ok to generate, default is 200",
    )
    parser.add_argument(
        "-n",
        "--ko",
        type=int,
        default=200,
        help="number of ko to generate, default is 200",
    )
    parser.add_argument(
        "-s",
        "--style_folder",
        help="path to existing folder where images are to be used to be style references",
    )
    parser.add_argument(
        "-c",
        "--content_folder",
        help="path to existing folder where images are to be used to be content references",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="path to existing folder where styled images will be put",
    )
    args = parser.parse_args()
    ACTION = args.action
    ROOT = args.rootfolder
    OK = args.ok
    KO = args.ko
    STYLE_FOLDER = args.style_folder
    CONTENT_FOLDER = args.content_folder
    OUTPUT_FOLDER = args.output_folder

    if ACTION[0] == "g":
        datagenerator = Datagenerator(ROOT)
        generator = generator(datagenerator, number_of_ok=OK, number_of_ko=KO)
        if ACTION == "generate_train":
            generator.generate_train_set()
        elif ACTION == "generate_test":
            cli.generate_test_set()
    elif ACTION == "style_image":
        output_image_size = 512
        transfer_random_style_folder(
            CONTENT_FOLDER, STYLE_FOLDER, OUTPUT_FOLDER, output_image_size
        )

