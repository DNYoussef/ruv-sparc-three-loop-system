#!/usr/bin/env python3
"""
Test suite for SSL setup script
Tests ssl-setup.py functionality
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import script
sys.path.insert(0, str(Path(__file__).parent.parent / "resources" / "scripts"))

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class TestSSLSetup(unittest.TestCase):
    """Test SSL setup functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not CRYPTO_AVAILABLE:
            raise unittest.SkipTest("cryptography library not available")

        # Import SSLSetup after checking availability
        try:
            import ssl_setup
            cls.ssl_setup_module = ssl_setup
        except ImportError:
            # If import fails, we'll skip tests that need it
            cls.ssl_setup_module = None

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "network-security"

        if self.ssl_setup_module:
            self.ssl_setup = self.ssl_setup_module.SSLSetup(
                self.config_dir,
                dry_run=False
            )

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_creation(self):
        """Test that SSL setup creates necessary directories"""
        self.assertTrue(self.ssl_setup.certs_dir.exists())
        self.assertTrue(self.ssl_setup.ca_dir.exists())

    def test_generate_private_key(self):
        """Test private key generation"""
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = self.ssl_setup.generate_private_key(2048)

        self.assertIsInstance(key, rsa.RSAPrivateKey)
        self.assertEqual(key.key_size, 2048)

    def test_generate_ca_certificate(self):
        """Test CA certificate generation"""
        ca_key, ca_cert = self.ssl_setup.generate_ca_certificate(
            common_name="Test CA",
            validity_days=365
        )

        # Verify certificate properties
        self.assertEqual(
            ca_cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value,
            "Test CA"
        )

        # Check that it's a CA certificate
        basic_constraints = ca_cert.extensions.get_extension_for_oid(
            x509.ExtensionOID.BASIC_CONSTRAINTS
        ).value
        self.assertTrue(basic_constraints.ca)

        # Check validity period
        now = datetime.utcnow()
        self.assertLess(ca_cert.not_valid_before, now)
        self.assertGreater(ca_cert.not_valid_after, now)

    def test_generate_server_certificate(self):
        """Test server certificate generation"""
        # Generate CA first
        ca_key, ca_cert = self.ssl_setup.generate_ca_certificate(
            common_name="Test CA"
        )

        # Generate server certificate
        san_domains = ["example.com", "www.example.com"]
        server_key, server_cert = self.ssl_setup.generate_server_certificate(
            common_name="example.com",
            ca_key=ca_key,
            ca_cert=ca_cert,
            san_domains=san_domains,
            validity_days=365
        )

        # Verify certificate properties
        self.assertEqual(
            server_cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value,
            "example.com"
        )

        # Check issuer is the CA
        self.assertEqual(server_cert.issuer, ca_cert.subject)

        # Check SAN extension
        san_ext = server_cert.extensions.get_extension_for_oid(
            x509.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        ).value

        san_dns_names = [name.value for name in san_ext if isinstance(name, x509.DNSName)]
        self.assertIn("example.com", san_dns_names)
        self.assertIn("www.example.com", san_dns_names)

        # Check that it's NOT a CA certificate
        basic_constraints = server_cert.extensions.get_extension_for_oid(
            x509.ExtensionOID.BASIC_CONSTRAINTS
        ).value
        self.assertFalse(basic_constraints.ca)

    def test_save_certificate(self):
        """Test saving certificate and key to disk"""
        ca_key, ca_cert = self.ssl_setup.generate_ca_certificate(
            common_name="Test CA"
        )

        paths = self.ssl_setup.save_certificate(
            "test-ca",
            ca_key,
            ca_cert,
            "ca"
        )

        # Verify files were created
        self.assertTrue(paths["key"].exists())
        self.assertTrue(paths["cert"].exists())

        # Verify file permissions
        key_stat = paths["key"].stat()
        key_mode = oct(key_stat.st_mode)[-3:]
        self.assertEqual(key_mode, "600")

        # Verify certificate can be loaded
        with open(paths["cert"], "rb") as f:
            loaded_cert = x509.load_pem_x509_certificate(
                f.read(),
                default_backend()
            )
        self.assertEqual(loaded_cert.subject, ca_cert.subject)

    def test_validate_certificate(self):
        """Test certificate validation"""
        # Generate and save certificate
        ca_key, ca_cert = self.ssl_setup.generate_ca_certificate(
            common_name="Test CA",
            validity_days=365
        )

        paths = self.ssl_setup.save_certificate(
            "test-ca",
            ca_key,
            ca_cert,
            "ca"
        )

        # Validate certificate
        details = self.ssl_setup.validate_certificate(paths["cert"])

        # Check validation results
        self.assertTrue(details["is_valid"])
        self.assertTrue(details["is_ca"])
        self.assertGreater(details["days_remaining"], 360)
        self.assertIn("Test CA", details["subject"])

    def test_expired_certificate_validation(self):
        """Test validation of expired certificate"""
        # Generate certificate with past validity
        ca_key, ca_cert = self.ssl_setup.generate_ca_certificate(
            common_name="Expired CA",
            validity_days=-1  # Already expired
        )

        paths = self.ssl_setup.save_certificate(
            "expired-ca",
            ca_key,
            ca_cert,
            "ca"
        )

        # Validate certificate
        details = self.ssl_setup.validate_certificate(paths["cert"])

        # Check validation results
        self.assertFalse(details["is_valid"])
        self.assertLess(details["days_remaining"], 0)

    def test_dry_run_mode(self):
        """Test dry run mode doesn't create files"""
        dry_run_setup = self.ssl_setup_module.SSLSetup(
            self.config_dir / "dry-run",
            dry_run=True
        )

        ca_key, ca_cert = dry_run_setup.generate_ca_certificate(
            common_name="Dry Run CA"
        )

        paths = dry_run_setup.save_certificate(
            "dry-run-ca",
            ca_key,
            ca_cert,
            "ca"
        )

        # Files should NOT exist in dry run mode
        self.assertFalse(paths["key"].exists())
        self.assertFalse(paths["cert"].exists())


class TestSSLSetupCLI(unittest.TestCase):
    """Test SSL setup CLI functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        if not CRYPTO_AVAILABLE:
            raise unittest.SkipTest("cryptography library not available")

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "network-security"
        self.config_dir.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_help(self):
        """Test CLI help output"""
        import subprocess

        script_path = Path(__file__).parent.parent / "resources" / "scripts" / "ssl-setup.py"

        if not script_path.exists():
            self.skipTest("ssl-setup.py not found")

        result = subprocess.run(
            ["python3", str(script_path), "--help"],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("SSL/TLS Certificate Setup", result.stdout)
        self.assertIn("--action", result.stdout)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSSLSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestSSLSetupCLI))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
