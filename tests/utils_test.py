"""
Test the core utility functions of the package.
"""
from lyscripts.utils import load_yaml_params


def test_load_yaml_params(capsys):
    """Check the basic loading of YAML parameters from a file."""
    non_existing_filepath = "./whatever"
    test_yaml_filepath = "./tests/test.yaml"
    failure_output = "✗ Parameter YAML file not found at whatever.\n"
    test_output = "✓ Read in YAML params from tests/test.yaml.\n"

    non_existing_params = load_yaml_params(non_existing_filepath)
    non_existing_capture = capsys.readouterr()
    test_params = load_yaml_params(test_yaml_filepath)
    test_capture = capsys.readouterr()

    assert non_existing_params == {}, (
        "When the file is not found, an empty dict should be returned."
    )
    assert non_existing_capture.out == failure_output, (
        "On `FileNotFound`, no failure is reported."
    )
    assert test_params == {"test": "This is just for testing"}, (
        "The function did not load a test YAML file correctly."
    )
    assert test_capture.out == test_output, (
        "On success, not the right message is printed."
    )
