"""Tests for WRDS credential validation."""

import os
import pytest
from unittest.mock import patch

from src.data.credentials import check_wrds_credentials, validate_and_exit


class TestCredentialValidation:
    """Test WRDS credential validation."""
    
    def test_both_credentials_present(self):
        """Test when both credentials are set."""
        with patch.dict(os.environ, {"WRDS_USERNAME": "test_user", "WRDS_PASSWORD": "test_pass"}):
            assert check_wrds_credentials() == "valid"
    
    def test_missing_username(self):
        """Test when username is missing."""
        with patch.dict(os.environ, {"WRDS_PASSWORD": "test_pass"}, clear=True):
            assert check_wrds_credentials() == "missing_username"
    
    def test_missing_password(self):
        """Test when password is missing."""
        with patch.dict(os.environ, {"WRDS_USERNAME": "test_user"}, clear=True):
            assert check_wrds_credentials() == "missing_password"
    
    def test_both_missing(self):
        """Test when both are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert check_wrds_credentials() == "both_missing"
    
    def test_exit_on_missing_credentials(self):
        """Test that validate_and_exit exits when credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                validate_and_exit()
            assert exc_info.value.code == 1
