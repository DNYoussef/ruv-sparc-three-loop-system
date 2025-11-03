#!/usr/bin/env python3
"""
SSL/TLS Certificate Setup Script for Network Security
Handles certificate generation, validation, and trust store configuration
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID, ExtensionOID
except ImportError:
    print("Error: cryptography library not installed")
    print("Install with: pip install cryptography")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SSLSetup:
    """Manages SSL/TLS certificate setup and configuration"""

    def __init__(self, config_dir: Path, dry_run: bool = False):
        self.config_dir = config_dir
        self.dry_run = dry_run
        self.certs_dir = config_dir / "certs"
        self.ca_dir = config_dir / "ca"

        # Create directories
        if not dry_run:
            self.certs_dir.mkdir(parents=True, exist_ok=True)
            self.ca_dir.mkdir(parents=True, exist_ok=True)

    def generate_private_key(self, key_size: int = 2048) -> rsa.RSAPrivateKey:
        """Generate RSA private key"""
        logger.info(f"Generating {key_size}-bit RSA private key...")
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )

    def generate_ca_certificate(
        self,
        common_name: str = "Network Security CA",
        validity_days: int = 3650
    ) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
        """Generate self-signed CA certificate"""
        logger.info(f"Generating CA certificate: {common_name}")

        # Generate private key
        private_key = self.generate_private_key(4096)

        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Network Security"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).sign(private_key, hashes.SHA256(), default_backend())

        return private_key, cert

    def generate_server_certificate(
        self,
        common_name: str,
        ca_key: rsa.RSAPrivateKey,
        ca_cert: x509.Certificate,
        san_domains: Optional[List[str]] = None,
        validity_days: int = 365
    ) -> Tuple[rsa.RSAPrivateKey, x509.Certificate]:
        """Generate server certificate signed by CA"""
        logger.info(f"Generating server certificate: {common_name}")

        # Generate private key
        private_key = self.generate_private_key(2048)

        # Build SAN extension
        san_list = [x509.DNSName(common_name)]
        if san_domains:
            san_list.extend([x509.DNSName(domain) for domain in san_domains])

        # Generate certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Network Security"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            ca_cert.subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=False,
        ).sign(ca_key, hashes.SHA256(), default_backend())

        return private_key, cert

    def save_certificate(
        self,
        name: str,
        private_key: rsa.RSAPrivateKey,
        certificate: x509.Certificate,
        cert_type: str = "server"
    ) -> Dict[str, Path]:
        """Save certificate and private key to disk"""
        logger.info(f"Saving {cert_type} certificate: {name}")

        output_dir = self.ca_dir if cert_type == "ca" else self.certs_dir

        # Define file paths
        key_path = output_dir / f"{name}.key"
        cert_path = output_dir / f"{name}.crt"

        if self.dry_run:
            logger.info(f"[DRY RUN] Would save private key to {key_path}")
            logger.info(f"[DRY RUN] Would save certificate to {cert_path}")
            return {"key": key_path, "cert": cert_path}

        # Save private key
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        os.chmod(key_path, 0o600)
        logger.info(f"Saved private key to {key_path}")

        # Save certificate
        with open(cert_path, "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))
        logger.info(f"Saved certificate to {cert_path}")

        return {"key": key_path, "cert": cert_path}

    def validate_certificate(self, cert_path: Path) -> Dict[str, any]:
        """Validate certificate and return details"""
        logger.info(f"Validating certificate: {cert_path}")

        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        # Extract details
        details = {
            "subject": cert.subject.rfc4514_string(),
            "issuer": cert.issuer.rfc4514_string(),
            "serial_number": cert.serial_number,
            "not_before": cert.not_valid_before.isoformat(),
            "not_after": cert.not_valid_after.isoformat(),
            "is_ca": False,
            "san_domains": [],
        }

        # Check if CA certificate
        try:
            basic_constraints = cert.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value
            details["is_ca"] = basic_constraints.ca
        except x509.ExtensionNotFound:
            pass

        # Extract SAN domains
        try:
            san = cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
            details["san_domains"] = [
                name.value for name in san
                if isinstance(name, x509.DNSName)
            ]
        except x509.ExtensionNotFound:
            pass

        # Check validity
        now = datetime.utcnow()
        details["is_valid"] = (
            cert.not_valid_before <= now <= cert.not_valid_after
        )
        details["days_remaining"] = (cert.not_valid_after - now).days

        logger.info(f"Certificate is {'valid' if details['is_valid'] else 'INVALID'}")
        logger.info(f"Days remaining: {details['days_remaining']}")

        return details

    def install_ca_certificate(self, ca_cert_path: Path) -> bool:
        """Install CA certificate to system trust store"""
        logger.info("Installing CA certificate to system trust store...")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would install {ca_cert_path} to trust store")
            return True

        # Detect OS and install accordingly
        if sys.platform == "linux":
            return self._install_ca_linux(ca_cert_path)
        elif sys.platform == "darwin":
            return self._install_ca_macos(ca_cert_path)
        elif sys.platform == "win32":
            return self._install_ca_windows(ca_cert_path)
        else:
            logger.error(f"Unsupported platform: {sys.platform}")
            return False

    def _install_ca_linux(self, ca_cert_path: Path) -> bool:
        """Install CA certificate on Linux"""
        try:
            # Copy to system CA directory
            ca_dir = Path("/usr/local/share/ca-certificates")
            ca_dir.mkdir(parents=True, exist_ok=True)

            dest = ca_dir / ca_cert_path.name
            subprocess.run(["cp", str(ca_cert_path), str(dest)], check=True)

            # Update CA certificates
            subprocess.run(["update-ca-certificates"], check=True)

            logger.info("CA certificate installed successfully on Linux")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CA certificate: {e}")
            return False

    def _install_ca_macos(self, ca_cert_path: Path) -> bool:
        """Install CA certificate on macOS"""
        try:
            subprocess.run([
                "security", "add-trusted-cert",
                "-d", "-r", "trustRoot",
                "-k", "/Library/Keychains/System.keychain",
                str(ca_cert_path)
            ], check=True)

            logger.info("CA certificate installed successfully on macOS")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CA certificate: {e}")
            return False

    def _install_ca_windows(self, ca_cert_path: Path) -> bool:
        """Install CA certificate on Windows"""
        try:
            subprocess.run([
                "certutil", "-addstore", "-f", "Root",
                str(ca_cert_path)
            ], check=True)

            logger.info("CA certificate installed successfully on Windows")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CA certificate: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="SSL/TLS Certificate Setup for Network Security"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("/etc/network-security"),
        help="Configuration directory (default: /etc/network-security)"
    )
    parser.add_argument(
        "--action",
        choices=["generate-ca", "generate-server", "validate", "install-ca"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--common-name",
        help="Common name for certificate"
    )
    parser.add_argument(
        "--san-domains",
        nargs="+",
        help="Subject Alternative Name domains"
    )
    parser.add_argument(
        "--cert-path",
        type=Path,
        help="Path to certificate file (for validate action)"
    )
    parser.add_argument(
        "--validity-days",
        type=int,
        default=365,
        help="Certificate validity in days (default: 365)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without making changes"
    )

    args = parser.parse_args()

    # Initialize SSL setup
    ssl_setup = SSLSetup(args.config_dir, args.dry_run)

    # Perform action
    if args.action == "generate-ca":
        ca_key, ca_cert = ssl_setup.generate_ca_certificate(
            common_name=args.common_name or "Network Security CA",
            validity_days=args.validity_days
        )
        ssl_setup.save_certificate("ca", ca_key, ca_cert, "ca")

    elif args.action == "generate-server":
        if not args.common_name:
            logger.error("--common-name required for server certificate")
            sys.exit(1)

        # Load CA certificate and key
        ca_key_path = ssl_setup.ca_dir / "ca.key"
        ca_cert_path = ssl_setup.ca_dir / "ca.crt"

        if not ca_key_path.exists() or not ca_cert_path.exists():
            logger.error("CA certificate not found. Run 'generate-ca' first.")
            sys.exit(1)

        with open(ca_key_path, "rb") as f:
            ca_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

        with open(ca_cert_path, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(
                f.read(), default_backend()
            )

        # Generate server certificate
        server_key, server_cert = ssl_setup.generate_server_certificate(
            common_name=args.common_name,
            ca_key=ca_key,
            ca_cert=ca_cert,
            san_domains=args.san_domains,
            validity_days=args.validity_days
        )
        ssl_setup.save_certificate(
            args.common_name.replace(".", "-"),
            server_key,
            server_cert,
            "server"
        )

    elif args.action == "validate":
        if not args.cert_path:
            logger.error("--cert-path required for validate action")
            sys.exit(1)

        details = ssl_setup.validate_certificate(args.cert_path)
        print(json.dumps(details, indent=2))

    elif args.action == "install-ca":
        ca_cert_path = ssl_setup.ca_dir / "ca.crt"
        if not ca_cert_path.exists():
            logger.error("CA certificate not found")
            sys.exit(1)

        ssl_setup.install_ca_certificate(ca_cert_path)


if __name__ == "__main__":
    main()
