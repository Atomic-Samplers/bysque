from __future__ import annotations

from template.examples.example import example_function


def test_example_function() -> None:
    """Test the example_function from the template.examples module."""
    result = example_function()

    assert result.startswith("Hello from")
