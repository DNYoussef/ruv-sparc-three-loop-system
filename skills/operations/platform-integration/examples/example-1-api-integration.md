# Example 1: Salesforce to HubSpot API Integration

## Overview

This example demonstrates a complete API integration between Salesforce and HubSpot, including:
- OAuth2 authentication for both platforms
- Bidirectional contact synchronization
- Real-time API calls with rate limiting
- Error handling and retry logic
- Data transformation and mapping

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  Salesforce │◀───────▶│ Integration  │◀───────▶│   HubSpot   │
│     CRM     │  OAuth2 │   Service    │  OAuth2 │     CRM     │
└─────────────┘         └──────────────┘         └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │    Redis     │
                        │ (Rate Limit) │
                        └──────────────┘
```

## Prerequisites

```bash
# Install dependencies
pip install requests requests-oauthlib redis python-dotenv

# Environment variables
export SALESFORCE_CLIENT_ID="your_client_id"
export SALESFORCE_CLIENT_SECRET="your_client_secret"
export SALESFORCE_USERNAME="your_username"
export SALESFORCE_PASSWORD="your_password"
export SALESFORCE_SECURITY_TOKEN="your_token"

export HUBSPOT_ACCESS_TOKEN="your_access_token"
export REDIS_URL="redis://localhost:6379"
```

## Implementation

### 1. Salesforce Connector

```python
#!/usr/bin/env python3
"""
Salesforce API Connector with OAuth2
"""

import os
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class SalesforceConnector:
    """Salesforce REST API Client"""

    def __init__(self):
        self.client_id = os.getenv('SALESFORCE_CLIENT_ID')
        self.client_secret = os.getenv('SALESFORCE_CLIENT_SECRET')
        self.username = os.getenv('SALESFORCE_USERNAME')
        self.password = os.getenv('SALESFORCE_PASSWORD')
        self.security_token = os.getenv('SALESFORCE_SECURITY_TOKEN')

        self.base_url = None
        self.access_token = None
        self.token_expires_at = None
        self.session = requests.Session()

        self._authenticate()

    def _authenticate(self):
        """Authenticate using OAuth2 password flow"""
        token_url = 'https://login.salesforce.com/services/oauth2/token'

        response = requests.post(token_url, data={
            'grant_type': 'password',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'username': self.username,
            'password': f"{self.password}{self.security_token}"
        })

        response.raise_for_status()
        token_data = response.json()

        self.access_token = token_data['access_token']
        self.base_url = f"{token_data['instance_url']}/services/data/v57.0"
        self.token_expires_at = datetime.now() + timedelta(hours=1)

        self.session.headers.update({
            'Authorization': f"Bearer {self.access_token}",
            'Content-Type': 'application/json'
        })

        print("✅ Salesforce authenticated")

    def _check_token_expiry(self):
        """Refresh token if expired"""
        if datetime.now() >= self.token_expires_at:
            print("🔄 Refreshing Salesforce token...")
            self._authenticate()

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        self._check_token_expiry()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, **kwargs)

        try:
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            print(f"Salesforce API Error: {e.response.text}")
            raise

    def get_contacts(self, limit: int = 100, modified_since: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch contacts from Salesforce"""
        query = "SELECT Id, FirstName, LastName, Email, Phone, MobilePhone, Company, Title, CreatedDate, LastModifiedDate FROM Contact"

        if modified_since:
            query += f" WHERE LastModifiedDate > {modified_since}"

        query += f" LIMIT {limit}"

        response = self._request('GET', 'query', params={'q': query})
        return response.get('records', [])

    def get_contact_by_id(self, contact_id: str) -> Dict[str, Any]:
        """Get specific contact by ID"""
        return self._request('GET', f'sobjects/Contact/{contact_id}')

    def create_contact(self, contact_data: Dict[str, Any]) -> str:
        """Create new contact"""
        response = self._request('POST', 'sobjects/Contact', json=contact_data)
        return response['id']

    def update_contact(self, contact_id: str, contact_data: Dict[str, Any]):
        """Update existing contact"""
        self._request('PATCH', f'sobjects/Contact/{contact_id}', json=contact_data)

    def delete_contact(self, contact_id: str):
        """Delete contact"""
        self._request('DELETE', f'sobjects/Contact/{contact_id}')

    def query(self, soql: str) -> List[Dict[str, Any]]:
        """Execute SOQL query"""
        response = self._request('GET', 'query', params={'q': soql})
        return response.get('records', [])


### 2. HubSpot Connector

```python
#!/usr/bin/env python3
"""
HubSpot API Connector
"""

import os
import requests
from typing import Dict, Any, List, Optional


class HubSpotConnector:
    """HubSpot REST API Client"""

    def __init__(self):
        self.access_token = os.getenv('HUBSPOT_ACCESS_TOKEN')
        self.base_url = 'https://api.hubapi.com'
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f"Bearer {self.access_token}",
            'Content-Type': 'application/json'
        })

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(method, url, **kwargs)

        try:
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.HTTPError as e:
            print(f"HubSpot API Error: {e.response.text}")
            raise

    def get_contacts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch contacts from HubSpot"""
        response = self._request('GET', 'crm/v3/objects/contacts', params={
            'limit': limit,
            'properties': ['email', 'firstname', 'lastname', 'phone', 'mobilephone', 'company', 'jobtitle']
        })
        return response.get('results', [])

    def get_contact_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get contact by email"""
        try:
            response = self._request('GET', f'crm/v3/objects/contacts/{email}', params={
                'idProperty': 'email'
            })
            return response
        except:
            return None

    def create_contact(self, properties: Dict[str, Any]) -> str:
        """Create new contact"""
        response = self._request('POST', 'crm/v3/objects/contacts', json={
            'properties': properties
        })
        return response['id']

    def update_contact(self, contact_id: str, properties: Dict[str, Any]):
        """Update existing contact"""
        self._request('PATCH', f'crm/v3/objects/contacts/{contact_id}', json={
            'properties': properties
        })

    def delete_contact(self, contact_id: str):
        """Delete contact"""
        self._request('DELETE', f'crm/v3/objects/contacts/{contact_id}')


### 3. Data Transformation

```python
#!/usr/bin/env python3
"""
Data transformation between Salesforce and HubSpot
"""

from typing import Dict, Any


class ContactTransformer:
    """Transform contact data between platforms"""

    @staticmethod
    def salesforce_to_hubspot(sf_contact: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Salesforce contact to HubSpot format"""
        return {
            'email': sf_contact.get('Email', '').lower(),
            'firstname': sf_contact.get('FirstName'),
            'lastname': sf_contact.get('LastName'),
            'phone': sf_contact.get('Phone'),
            'mobilephone': sf_contact.get('MobilePhone'),
            'company': sf_contact.get('Company'),
            'jobtitle': sf_contact.get('Title'),
            'salesforce_id': sf_contact.get('Id')
        }

    @staticmethod
    def hubspot_to_salesforce(hs_contact: Dict[str, Any]) -> Dict[str, Any]:
        """Transform HubSpot contact to Salesforce format"""
        props = hs_contact.get('properties', {})

        return {
            'Email': props.get('email', '').lower(),
            'FirstName': props.get('firstname'),
            'LastName': props.get('lastname'),
            'Phone': props.get('phone'),
            'MobilePhone': props.get('mobilephone'),
            'Company': props.get('company'),
            'Title': props.get('jobtitle'),
            'External_ID__c': hs_contact.get('id')
        }


### 4. Integration Service with Rate Limiting

```python
#!/usr/bin/env python3
"""
Integration service with rate limiting
"""

import redis
import time
from typing import Dict, Any, List


class RateLimiter:
    """Token bucket rate limiter using Redis"""

    def __init__(self, redis_url: str, requests_per_second: int = 10):
        self.redis_client = redis.from_url(redis_url)
        self.requests_per_second = requests_per_second

    def acquire(self, key: str) -> bool:
        """Acquire rate limit token"""
        current = int(time.time())
        redis_key = f"ratelimit:{key}:{current}"

        count = self.redis_client.incr(redis_key)
        if count == 1:
            self.redis_client.expire(redis_key, 1)

        return count <= self.requests_per_second

    def wait_if_needed(self, key: str):
        """Wait if rate limit exceeded"""
        while not self.acquire(key):
            time.sleep(0.1)


class IntegrationService:
    """Main integration orchestration"""

    def __init__(self):
        self.sf_connector = SalesforceConnector()
        self.hs_connector = HubSpotConnector()
        self.transformer = ContactTransformer()
        self.rate_limiter = RateLimiter(os.getenv('REDIS_URL', 'redis://localhost:6379'))

    def sync_salesforce_to_hubspot(self, limit: int = 100):
        """Sync contacts from Salesforce to HubSpot"""
        print("🔄 Syncing Salesforce → HubSpot...")

        # Fetch Salesforce contacts
        sf_contacts = self.sf_connector.get_contacts(limit=limit)
        print(f"  Found {len(sf_contacts)} contacts in Salesforce")

        synced = 0
        errors = 0

        for sf_contact in sf_contacts:
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed('hubspot')

                # Transform data
                hs_data = self.transformer.salesforce_to_hubspot(sf_contact)

                # Check if contact exists in HubSpot
                existing = self.hs_connector.get_contact_by_email(hs_data['email'])

                if existing:
                    # Update existing contact
                    self.hs_connector.update_contact(existing['id'], hs_data)
                    print(f"  ✓ Updated: {hs_data['email']}")
                else:
                    # Create new contact
                    self.hs_connector.create_contact(hs_data)
                    print(f"  ✓ Created: {hs_data['email']}")

                synced += 1

            except Exception as e:
                print(f"  ✗ Error syncing {sf_contact.get('Email')}: {str(e)}")
                errors += 1

        print(f"✅ Sync complete: {synced} synced, {errors} errors")

    def sync_hubspot_to_salesforce(self, limit: int = 100):
        """Sync contacts from HubSpot to Salesforce"""
        print("🔄 Syncing HubSpot → Salesforce...")

        # Fetch HubSpot contacts
        hs_contacts = self.hs_connector.get_contacts(limit=limit)
        print(f"  Found {len(hs_contacts)} contacts in HubSpot")

        synced = 0
        errors = 0

        for hs_contact in hs_contacts:
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed('salesforce')

                # Transform data
                sf_data = self.transformer.hubspot_to_salesforce(hs_contact)

                # Check if contact exists in Salesforce
                email = sf_data.get('Email')
                existing = self.sf_connector.query(f"SELECT Id FROM Contact WHERE Email = '{email}' LIMIT 1")

                if existing:
                    # Update existing contact
                    self.sf_connector.update_contact(existing[0]['Id'], sf_data)
                    print(f"  ✓ Updated: {email}")
                else:
                    # Create new contact
                    self.sf_connector.create_contact(sf_data)
                    print(f"  ✓ Created: {email}")

                synced += 1

            except Exception as e:
                props = hs_contact.get('properties', {})
                print(f"  ✗ Error syncing {props.get('email')}: {str(e)}")
                errors += 1

        print(f"✅ Sync complete: {synced} synced, {errors} errors")


### 5. Main Execution

```python
#!/usr/bin/env python3
"""
Main integration execution
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    print("================================================================")
    print("Salesforce ↔ HubSpot API Integration")
    print("================================================================")

    # Initialize integration service
    service = IntegrationService()

    # Bidirectional sync
    service.sync_salesforce_to_hubspot(limit=50)
    print()
    service.sync_hubspot_to_salesforce(limit=50)

    print()
    print("================================================================")
    print("Integration Complete!")
    print("================================================================")


if __name__ == '__main__':
    main()
```

## Usage

```bash
# Run integration
python api-integration-example.py

# Expected output:
# ================================================================
# Salesforce ↔ HubSpot API Integration
# ================================================================
# ✅ Salesforce authenticated
# 🔄 Syncing Salesforce → HubSpot...
#   Found 50 contacts in Salesforce
#   ✓ Updated: john.doe@example.com
#   ✓ Created: jane.smith@example.com
#   ...
# ✅ Sync complete: 48 synced, 2 errors
#
# 🔄 Syncing HubSpot → Salesforce...
#   Found 50 contacts in HubSpot
#   ✓ Updated: alice@example.com
#   ✓ Created: bob@example.com
#   ...
# ✅ Sync complete: 50 synced, 0 errors
# ================================================================
```

## Key Features

1. **OAuth2 Authentication**: Automatic token refresh for Salesforce
2. **Rate Limiting**: Redis-based token bucket algorithm
3. **Data Transformation**: Automatic field mapping between platforms
4. **Error Handling**: Graceful error handling with detailed logging
5. **Idempotency**: Updates existing records, creates only new ones
6. **Bidirectional Sync**: Sync in both directions

## Production Enhancements

```python
# Add retry logic with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def api_call_with_retry(self, func, *args, **kwargs):
    return func(*args, **kwargs)

# Add monitoring
from prometheus_client import Counter, Histogram

api_calls = Counter('api_calls_total', 'Total API calls', ['platform', 'status'])
api_latency = Histogram('api_call_duration_seconds', 'API call latency', ['platform'])

# Add webhook notifications
def send_notification(message: str):
    requests.post(
        os.getenv('SLACK_WEBHOOK_URL'),
        json={'text': message}
    )
```

## Troubleshooting

**Authentication Failed**:
- Verify credentials in `.env`
- Check Salesforce security token
- Ensure OAuth app is configured

**Rate Limit Exceeded**:
- Reduce `requests_per_second` setting
- Implement exponential backoff
- Use batch API operations

**Data Sync Conflicts**:
- Implement conflict resolution strategy
- Use `LastModifiedDate` for timestamp comparison
- Add manual review queue for conflicts

---

**Generated with Platform Integration Skill v2.0.0**
