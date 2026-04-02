"""WRDS credential validation utilities."""

import os
import sys
from typing import Literal


def check_wrds_credentials() -> Literal[
    "valid", "missing_username", "missing_password", "both_missing"
]:
    """Check if WRDS credentials are available.

    Returns:
        Status of credentials
    """
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")

    if not username and not password:
        return "both_missing"
    elif not username:
        return "missing_username"
    elif not password:
        return "missing_password"
    else:
        return "valid"


def validate_and_exit() -> None:
    """Validate credentials and exit if missing.

    Prints helpful error message with instructions.
    """
    status = check_wrds_credentials()

    if status == "valid":
        return

    error_messages = {
        "both_missing": "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD environment variables.",
        "missing_username": "WRDS_USERNAME environment variable not set.",
        "missing_password": "WRDS_PASSWORD environment variable not set.",
    }

    print(f"\n❌ ERROR: {error_messages[status]}", file=sys.stderr)
    print("\nTo set credentials:", file=sys.stderr)
    print("  export WRDS_USERNAME=your_username", file=sys.stderr)
    print("  export WRDS_PASSWORD=your_password", file=sys.stderr)
    print("\nOr run with mock data explicitly (for testing only):", file=sys.stderr)
    print("  python -m scripts.preprocess_features --use-mock\n", file=sys.stderr)

    sys.exit(1)
