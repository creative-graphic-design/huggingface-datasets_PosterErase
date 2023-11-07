import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "PosterErase.py"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames=("expected_num_train", "expected_num_valid", "expected_num_test"),
    argvalues=((58114, 148, 146),),
)
def test_load_dataset(
    dataset_path: str,
    expected_num_train: int,
    expected_num_valid: int,
    expected_num_test: int,
):
    dataset = ds.load_dataset(path=dataset_path, token=True)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["validation"].num_rows == expected_num_valid
    assert dataset["test"].num_rows == expected_num_test
