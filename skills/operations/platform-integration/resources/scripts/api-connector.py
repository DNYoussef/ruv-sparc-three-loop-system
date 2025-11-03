#!/usr/bin/env python3
"""
API Connector Generator & Discovery Tool
Generates platform-specific API clients with authentication, rate limiting, and error handling.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime
import requests
from jinja2 import Template


class APIConnector:
    """Base API connector with common functionality"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url')
        self.auth = config.get('auth', {})
        self.session = requests.Session()
        self._setup_auth()

    def _setup_auth(self):
        """Configure authentication"""
        auth_type = self.auth.get('type')

        if auth_type == 'api_key':
            header = self.auth.get('header', 'X-API-Key')
            self.session.headers[header] = self.auth.get('key')

        elif auth_type == 'bearer':
            token = self.auth.get('token')
            self.session.headers['Authorization'] = f'Bearer {token}'

        elif auth_type == 'oauth2':
            # OAuth2 token will be managed separately
            pass

        elif auth_type == 'basic':
            from requests.auth import HTTPBasicAuth
            self.session.auth = HTTPBasicAuth(
                self.auth.get('username'),
                self.auth.get('password')
            )

    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with error handling"""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {str(e)}")
            raise

    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request('GET', endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request('POST', endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request('PUT', endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request('DELETE', endpoint, **kwargs)


class PlatformDiscovery:
    """Discover platform capabilities via API"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connector = APIConnector(config)

    def discover_endpoints(self) -> List[Dict[str, Any]]:
        """Discover available endpoints"""
        endpoints = []

        # Check for OpenAPI/Swagger
        for spec_path in ['/openapi.json', '/swagger.json', '/api/v1/spec']:
            try:
                spec = self.connector.get(spec_path)
                if 'paths' in spec:
                    for path, methods in spec['paths'].items():
                        for method, details in methods.items():
                            endpoints.append({
                                'path': path,
                                'method': method.upper(),
                                'description': details.get('summary', ''),
                                'parameters': details.get('parameters', [])
                            })
                    break
            except:
                continue

        return endpoints

    def test_authentication(self) -> bool:
        """Test if authentication is working"""
        try:
            # Try a simple health check or user endpoint
            for endpoint in ['/health', '/user', '/me', '/api/v1/status']:
                try:
                    self.connector.get(endpoint)
                    return True
                except:
                    continue
            return False
        except:
            return False

    def analyze_rate_limits(self) -> Dict[str, Any]:
        """Analyze rate limiting headers"""
        try:
            response = self.connector.session.get(self.config['base_url'])
            return {
                'limit': response.headers.get('X-RateLimit-Limit'),
                'remaining': response.headers.get('X-RateLimit-Remaining'),
                'reset': response.headers.get('X-RateLimit-Reset')
            }
        except:
            return {}


class ConnectorGenerator:
    """Generate platform-specific connector code"""

    CONNECTOR_TEMPLATE = '''#!/usr/bin/env python3
"""
{{ platform_name }} API Connector
Auto-generated on {{ timestamp }}
"""

import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


class {{ class_name }}Connector:
    """{{ platform_name }} API Client"""

    def __init__(self, {% for param in auth_params %}{{ param }}: str{% if not loop.last %}, {% endif %}{% endfor %}):
        self.base_url = "{{ base_url }}"
        {% for param in auth_params %}
        self.{{ param }} = {{ param }}
        {% endfor %}
        self.session = requests.Session()
        self._setup_auth()

    def _setup_auth(self):
        """Configure authentication"""
        {% if auth_type == 'api_key' %}
        self.session.headers['{{ auth_header }}'] = self.api_key
        {% elif auth_type == 'bearer' %}
        self.session.headers['Authorization'] = f'Bearer {self.token}'
        {% elif auth_type == 'oauth2' %}
        # OAuth2 token management
        self._refresh_token()
        {% endif %}

    {% if auth_type == 'oauth2' %}
    def _refresh_token(self):
        """Refresh OAuth2 token"""
        response = requests.post(
            "{{ token_url }}",
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
        )
        token_data = response.json()
        self.session.headers['Authorization'] = f"Bearer {token_data['access_token']}"
    {% endif %}

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with error handling"""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            print(f"API Error: {str(e)}")
            raise

    # Generated methods for discovered endpoints
    {% for endpoint in endpoints %}
    def {{ endpoint.method_name }}(self, {% if endpoint.params %}{{ endpoint.params }}{% endif %}**kwargs) -> Dict[str, Any]:
        """{{ endpoint.description }}"""
        return self._request('{{ endpoint.method }}', '{{ endpoint.path }}', **kwargs)

    {% endfor %}


# Example usage
if __name__ == '__main__':
    connector = {{ class_name }}Connector(
        {% for param in auth_params %}
        {{ param }}=os.getenv('{{ param.upper() }}'){% if not loop.last %},{% endif %}
        {% endfor %}
    )

    # Example API calls
    {% for example in examples %}
    # {{ example.description }}
    # result = connector.{{ example.method }}({{ example.params }})
    {% endfor %}
'''

    def __init__(self, platform_config: Dict[str, Any]):
        self.config = platform_config

    def generate(self, output_path: str, endpoints: List[Dict[str, Any]] = None):
        """Generate connector code"""
        platform_name = self.config['name']
        class_name = ''.join(word.capitalize() for word in platform_name.split('_'))

        # Extract auth parameters
        auth_params = []
        auth_type = self.config.get('auth', {}).get('type', 'api_key')

        if auth_type == 'api_key':
            auth_params = ['api_key']
        elif auth_type == 'bearer':
            auth_params = ['token']
        elif auth_type == 'oauth2':
            auth_params = ['client_id', 'client_secret']
        elif auth_type == 'basic':
            auth_params = ['username', 'password']

        # Process endpoints
        processed_endpoints = []
        if endpoints:
            for ep in endpoints[:10]:  # Limit to first 10 endpoints
                method_name = self._generate_method_name(ep['path'], ep['method'])
                processed_endpoints.append({
                    'method_name': method_name,
                    'method': ep['method'],
                    'path': ep['path'],
                    'description': ep.get('description', ''),
                    'params': self._extract_params(ep.get('parameters', []))
                })

        # Render template
        template = Template(self.CONNECTOR_TEMPLATE)
        code = template.render(
            platform_name=platform_name.replace('_', ' ').title(),
            class_name=class_name,
            timestamp=datetime.now().isoformat(),
            base_url=self.config['base_url'],
            auth_type=auth_type,
            auth_header=self.config.get('auth', {}).get('header', 'X-API-Key'),
            auth_params=auth_params,
            token_url=self.config.get('auth', {}).get('token_url', ''),
            endpoints=processed_endpoints,
            examples=self._generate_examples(processed_endpoints[:3])
        )

        # Write to file
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)

        os.chmod(output_path, 0o755)
        print(f"✅ Generated connector: {output_path}")

    def _generate_method_name(self, path: str, method: str) -> str:
        """Generate Python method name from endpoint path"""
        # Remove API version prefix
        path = path.replace('/api/v1/', '').replace('/api/', '')

        # Convert to method name
        parts = [p for p in path.split('/') if p and not p.startswith('{')]
        name = '_'.join(parts)

        # Add method prefix
        prefix = {
            'GET': 'get',
            'POST': 'create',
            'PUT': 'update',
            'DELETE': 'delete',
            'PATCH': 'patch'
        }.get(method, 'call')

        return f"{prefix}_{name}" if name else prefix

    def _extract_params(self, parameters: List[Dict[str, Any]]) -> str:
        """Extract required parameters from endpoint spec"""
        required = [p['name'] for p in parameters if p.get('required')]
        return ', '.join(f"{p}: str" for p in required[:3])  # Limit to 3 params

    def _generate_examples(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate usage examples"""
        examples = []
        for ep in endpoints:
            examples.append({
                'description': f"{ep['method']} {ep['path']}",
                'method': ep['method_name'],
                'params': ''
            })
        return examples


def main():
    parser = argparse.ArgumentParser(description='API Connector Tool')
    parser.add_argument('--mode', choices=['discovery', 'generate'], required=True)
    parser.add_argument('--config', help='Platform configuration JSON file')
    parser.add_argument('--platform', help='Platform name')
    parser.add_argument('--output', help='Output file/directory')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        print("Error: --config required")
        sys.exit(1)

    if args.mode == 'discovery':
        # Discover platform capabilities
        platforms = config.get('platforms', [])
        results = {'platforms': []}

        for platform in platforms:
            print(f"\n🔍 Discovering: {platform['name']}")
            discovery = PlatformDiscovery(platform)

            auth_ok = discovery.test_authentication()
            endpoints = discovery.discover_endpoints()
            rate_limits = discovery.analyze_rate_limits()

            results['platforms'].append({
                'name': platform['name'],
                'authentication': 'success' if auth_ok else 'failed',
                'endpoints_found': len(endpoints),
                'endpoints': endpoints,
                'rate_limits': rate_limits
            })

            print(f"  ✅ Auth: {auth_ok}")
            print(f"  📡 Endpoints: {len(endpoints)}")

        # Save results
        output_file = args.output or 'discovery-report.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Discovery complete: {output_file}")

    elif args.mode == 'generate':
        # Generate connector code
        if args.platform:
            # Generate for specific platform
            platforms = [p for p in config.get('platforms', []) if p['name'] == args.platform]
        else:
            # Generate for all platforms
            platforms = config.get('platforms', [])

        for platform in platforms:
            platform_name = platform['name']
            output_file = args.output or f"{platform_name}_connector.py"

            print(f"\n🔧 Generating connector: {platform_name}")

            generator = ConnectorGenerator(platform)

            # Try to discover endpoints first
            try:
                discovery = PlatformDiscovery(platform)
                endpoints = discovery.discover_endpoints()
            except:
                endpoints = []

            generator.generate(output_file, endpoints)


if __name__ == '__main__':
    main()
