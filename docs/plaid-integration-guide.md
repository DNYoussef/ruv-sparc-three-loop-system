# Plaid API Integration Guide - FastAPI + React
**For trader-ai Dashboard**
*Last Updated: November 7, 2025*

---

## Table of Contents
1. [Overview](#overview)
2. [Authentication Flow](#authentication-flow)
3. [Python Backend Setup (FastAPI)](#python-backend-setup-fastapi)
4. [React Frontend Setup](#react-frontend-setup)
5. [Database Schema](#database-schema)
6. [Security Best Practices](#security-best-practices)
7. [Wells Fargo & Venmo Support](#wells-fargo--venmo-support)
8. [Rate Limits & Development Environment](#rate-limits--development-environment)
9. [Common Gotchas & Solutions](#common-gotchas--solutions)
10. [Resources](#resources)

---

## Overview

Plaid provides a secure way to connect users' bank accounts to your application. This guide covers integration with:
- **Backend**: Python FastAPI (using plaid-python SDK v37.0.0+)
- **Frontend**: React (using react-plaid-link)
- **Products**: Auth (account balances) + Transactions (transaction history)
- **Institutions**: Wells Fargo and Venmo (via Plaid's 12,000+ supported institutions)

### Key Concepts

- **Link Token**: Short-lived (4 hours), one-time use token to initialize Plaid Link UI
- **Public Token**: Temporary token (30 min) obtained after user completes Link flow
- **Access Token**: Permanent token for API requests, stored securely server-side
- **Item**: Represents a user's connection to a financial institution

---

## Authentication Flow

```
┌─────────────┐                          ┌──────────────┐
│   Client    │                          │    Server    │
│  (React)    │                          │  (FastAPI)   │
└──────┬──────┘                          └──────┬───────┘
       │                                        │
       │  1. Request Link Token                 │
       ├───────────────────────────────────────>│
       │                                        │
       │                                        │  2. Create Link Token
       │                                        │  POST /link/token/create
       │                                        ├─────────────────────┐
       │                                        │                     │
       │  3. Return link_token                  │<────────────────────┘
       │<───────────────────────────────────────┤
       │                                        │
       │                                        │
       │  4. Initialize Plaid Link              │
       │     with link_token                    │
       ├────────────────────┐                   │
       │                    │                   │
       │  5. User completes │                   │
       │     OAuth flow     │                   │
       │<───────────────────┘                   │
       │                                        │
       │                                        │
       │  6. Receive public_token               │
       │     from onSuccess callback            │
       ├────────────────────┐                   │
       │<───────────────────┘                   │
       │                                        │
       │                                        │
       │  7. Send public_token to server        │
       ├───────────────────────────────────────>│
       │                                        │
       │                                        │  8. Exchange Token
       │                                        │  POST /item/public_token/exchange
       │                                        ├─────────────────────┐
       │                                        │                     │
       │  9. Return success                     │<────────────────────┘
       │<───────────────────────────────────────┤
       │                                        │
       │                                        │  10. Store access_token
       │                                        │      + item_id in DB
       │                                        ├─────────────────────┐
       │                                        │<────────────────────┘
       │                                        │
```

### Flow Steps Explained

1. **Client requests link token** from your FastAPI backend
2. **Server creates link token** via Plaid API `/link/token/create`
3. **Server returns link_token** to client (expires in 4 hours)
4. **Client initializes Plaid Link** UI component with link_token
5. **User authenticates** with their bank (OAuth redirect for institutions like Wells Fargo)
6. **Plaid Link returns public_token** in `onSuccess` callback (expires in 30 minutes)
7. **Client sends public_token** to server
8. **Server exchanges public_token for access_token** via `/item/public_token/exchange`
9. **Server returns success** to client
10. **Server stores access_token and item_id** securely in database

---

## Python Backend Setup (FastAPI)

### Installation

```bash
pip install plaid-python fastapi[all] python-dotenv sqlalchemy psycopg2-binary
```

### Environment Variables

Create a `.env` file:

```bash
# Plaid Configuration
PLAID_CLIENT_ID=your_client_id_here
PLAID_SECRET=your_secret_here
PLAID_ENV=sandbox  # Options: sandbox, development, production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/trader_ai

# API Version
PLAID_API_VERSION=2020-09-14
```

### Plaid Client Configuration

**File**: `app/plaid_client.py`

```python
import os
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
import plaid
from dotenv import load_dotenv

load_dotenv()

# Configure Plaid client
configuration = plaid.Configuration(
    host=plaid.Environment.Sandbox,  # Change to Development or Production as needed
    api_key={
        'clientId': os.getenv('PLAID_CLIENT_ID'),
        'secret': os.getenv('PLAID_SECRET'),
        'plaidVersion': os.getenv('PLAID_API_VERSION', '2020-09-14')
    }
)

api_client = plaid.ApiClient(configuration)
plaid_client = plaid_api.PlaidApi(api_client)
```

### FastAPI Endpoints

**File**: `app/routes/plaid_routes.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.plaid_client import plaid_client
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.transactions_sync_request import TransactionsSyncRequest
import plaid
from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/api/plaid", tags=["plaid"])


# Request/Response models
class LinkTokenResponse(BaseModel):
    link_token: str


class PublicTokenRequest(BaseModel):
    public_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    item_id: str


# Endpoint 1: Create Link Token
@router.post("/create_link_token", response_model=LinkTokenResponse)
async def create_link_token(user_id: str, db: Session = Depends(get_db)):
    """
    Create a link token for initializing Plaid Link.

    Args:
        user_id: Your application's user identifier
        db: Database session

    Returns:
        LinkTokenResponse with link_token
    """
    try:
        request = LinkTokenCreateRequest(
            products=[Products("auth"), Products("transactions")],
            client_name="Trader AI Dashboard",
            country_codes=[CountryCode("US")],
            language="en",
            user=LinkTokenCreateRequestUser(
                client_user_id=user_id
            ),
            # Optional: Configure OAuth redirect URI for mobile web
            redirect_uri="https://yourdomain.com/oauth-redirect",  # Change for production
        )

        response = plaid_client.link_token_create(request)
        return LinkTokenResponse(link_token=response.link_token)

    except plaid.ApiException as e:
        raise HTTPException(status_code=400, detail=f"Plaid API error: {e}")


# Endpoint 2: Exchange Public Token for Access Token
@router.post("/exchange_public_token", response_model=AccessTokenResponse)
async def exchange_public_token(
    request: PublicTokenRequest,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Exchange a public token for an access token after user completes Link.

    Args:
        request: PublicTokenRequest with public_token
        user_id: Your application's user identifier
        db: Database session

    Returns:
        AccessTokenResponse with access_token and item_id
    """
    try:
        exchange_request = ItemPublicTokenExchangeRequest(
            public_token=request.public_token
        )

        exchange_response = plaid_client.item_public_token_exchange(exchange_request)

        access_token = exchange_response.access_token
        item_id = exchange_response.item_id

        # TODO: Store access_token and item_id in database
        # Example:
        # plaid_item = PlaidItem(
        #     user_id=user_id,
        #     access_token=access_token,  # Encrypt this!
        #     item_id=item_id
        # )
        # db.add(plaid_item)
        # db.commit()

        return AccessTokenResponse(
            access_token=access_token,
            item_id=item_id
        )

    except plaid.ApiException as e:
        raise HTTPException(status_code=400, detail=f"Plaid API error: {e}")


# Endpoint 3: Get Account Balances
@router.get("/accounts/{item_id}")
async def get_accounts(item_id: str, db: Session = Depends(get_db)):
    """
    Retrieve account balances for a connected Item.

    Args:
        item_id: Plaid Item ID
        db: Database session

    Returns:
        Account balances and details
    """
    try:
        # TODO: Retrieve access_token from database using item_id
        # access_token = get_access_token_from_db(db, item_id)

        # For demonstration purposes only (replace with DB lookup):
        access_token = "access-sandbox-xxx"  # NEVER hardcode in production!

        request = AccountsGetRequest(access_token=access_token)
        response = plaid_client.accounts_get(request)

        accounts = []
        for account in response.accounts:
            accounts.append({
                "account_id": account.account_id,
                "name": account.name,
                "type": account.type,
                "subtype": account.subtype,
                "balance": {
                    "current": account.balances.current,
                    "available": account.balances.available,
                    "currency": account.balances.iso_currency_code
                }
            })

        return {"accounts": accounts}

    except plaid.ApiException as e:
        raise HTTPException(status_code=400, detail=f"Plaid API error: {e}")


# Endpoint 4: Sync Transactions
@router.post("/transactions/sync/{item_id}")
async def sync_transactions(item_id: str, db: Session = Depends(get_db)):
    """
    Sync transactions for a connected Item using the new Transactions Sync API.

    Args:
        item_id: Plaid Item ID
        db: Database session

    Returns:
        Synced transactions
    """
    try:
        # TODO: Retrieve access_token from database
        access_token = "access-sandbox-xxx"  # Replace with DB lookup

        # Initialize sync with cursor (retrieve from DB or set to empty for first sync)
        cursor = ""  # Start from beginning, or use stored cursor for incremental sync

        added = []
        modified = []
        removed = []
        has_more = True

        # Paginate through all available transaction updates
        while has_more:
            request = TransactionsSyncRequest(
                access_token=access_token,
                cursor=cursor
            )
            response = plaid_client.transactions_sync(request)

            # Add transactions to the response lists
            added.extend(response.added)
            modified.extend(response.modified)
            removed.extend(response.removed)

            has_more = response.has_more
            cursor = response.next_cursor

        # TODO: Store cursor in database for next sync
        # update_cursor_in_db(db, item_id, cursor)

        return {
            "added": len(added),
            "modified": len(modified),
            "removed": len(removed),
            "cursor": cursor,
            "transactions": [
                {
                    "transaction_id": tx.transaction_id,
                    "date": str(tx.date),
                    "name": tx.name,
                    "amount": tx.amount,
                    "category": tx.category
                }
                for tx in added[:10]  # Return first 10 for demo
            ]
        }

    except plaid.ApiException as e:
        raise HTTPException(status_code=400, detail=f"Plaid API error: {e}")
```

### Error Handling

```python
from plaid import ApiException
from fastapi import HTTPException

def handle_plaid_error(e: ApiException):
    """
    Centralized Plaid error handler.
    """
    error_data = e.body
    if error_data:
        error_code = error_data.get('error_code')
        error_message = error_data.get('error_message')

        # Map Plaid error codes to HTTP status codes
        if error_code == 'INVALID_ACCESS_TOKEN':
            raise HTTPException(status_code=401, detail="Invalid access token")
        elif error_code == 'ITEM_LOGIN_REQUIRED':
            raise HTTPException(status_code=401, detail="User needs to re-authenticate")
        elif error_code == 'RATE_LIMIT_EXCEEDED':
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        else:
            raise HTTPException(status_code=400, detail=error_message)
    else:
        raise HTTPException(status_code=500, detail="Unknown Plaid error")
```

---

## React Frontend Setup

### Installation

```bash
npm install react-plaid-link axios
```

### Plaid Link Component (Hooks - Preferred Method)

**File**: `src/components/PlaidLinkButton.jsx`

```jsx
import React, { useState, useCallback, useEffect } from 'react';
import { usePlaidLink } from 'react-plaid-link';
import axios from 'axios';

const PlaidLinkButton = ({ userId, onSuccess }) => {
  const [linkToken, setLinkToken] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch link token from backend
  useEffect(() => {
    const fetchLinkToken = async () => {
      try {
        setLoading(true);
        const response = await axios.post(
          'http://localhost:8000/api/plaid/create_link_token',
          null,
          { params: { user_id: userId } }
        );
        setLinkToken(response.data.link_token);
      } catch (error) {
        console.error('Error fetching link token:', error);
        alert('Failed to initialize Plaid Link');
      } finally {
        setLoading(false);
      }
    };

    if (userId) {
      fetchLinkToken();
    }
  }, [userId]);

  // Handle successful Link completion
  const onPlaidSuccess = useCallback(async (publicToken, metadata) => {
    try {
      // Exchange public token for access token
      const response = await axios.post(
        'http://localhost:8000/api/plaid/exchange_public_token',
        { public_token: publicToken },
        { params: { user_id: userId } }
      );

      console.log('Access token received:', response.data.access_token);
      console.log('Item ID:', response.data.item_id);
      console.log('Institution:', metadata.institution);
      console.log('Accounts:', metadata.accounts);

      // Notify parent component
      if (onSuccess) {
        onSuccess(response.data);
      }

      alert(`Successfully linked ${metadata.institution.name}!`);
    } catch (error) {
      console.error('Error exchanging public token:', error);
      alert('Failed to complete bank linking');
    }
  }, [userId, onSuccess]);

  // Handle Link exit
  const onPlaidExit = useCallback((error, metadata) => {
    if (error) {
      console.error('Plaid Link error:', error);
    }
    console.log('Link exited:', metadata);
  }, []);

  // Initialize Plaid Link
  const config = {
    token: linkToken,
    onSuccess: onPlaidSuccess,
    onExit: onPlaidExit,
    onEvent: (eventName, metadata) => {
      console.log('Plaid Link event:', eventName, metadata);
    },
  };

  const { open, ready } = usePlaidLink(config);

  return (
    <button
      onClick={() => open()}
      disabled={!ready || loading}
      style={{
        padding: '12px 24px',
        backgroundColor: ready && !loading ? '#4A90E2' : '#CCCCCC',
        color: 'white',
        border: 'none',
        borderRadius: '4px',
        fontSize: '16px',
        cursor: ready && !loading ? 'pointer' : 'not-allowed',
        fontWeight: 'bold',
      }}
    >
      {loading ? 'Loading...' : 'Connect Bank Account'}
    </button>
  );
};

export default PlaidLinkButton;
```

### Handling OAuth Redirects (Required for Wells Fargo)

**File**: `src/components/PlaidOAuthRedirect.jsx`

```jsx
import React, { useEffect, useState } from 'react';
import { usePlaidLink } from 'react-plaid-link';
import { useLocation, useNavigate } from 'react-router-dom';

const PlaidOAuthRedirect = ({ userId, onSuccess }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [linkToken, setLinkToken] = useState(null);

  // Check if this is an OAuth redirect
  const isOAuthRedirect = location.search.includes('oauth_state_id');

  useEffect(() => {
    if (isOAuthRedirect) {
      // Retrieve the stored link token from localStorage
      const storedLinkToken = localStorage.getItem('plaid_link_token');
      if (storedLinkToken) {
        setLinkToken(storedLinkToken);
      } else {
        console.error('No stored link token found for OAuth redirect');
        navigate('/');
      }
    }
  }, [isOAuthRedirect, navigate]);

  const config = {
    token: linkToken,
    receivedRedirectUri: window.location.href, // Full URL with query params
    onSuccess: async (publicToken, metadata) => {
      try {
        // Exchange public token for access token
        const response = await fetch(
          `http://localhost:8000/api/plaid/exchange_public_token?user_id=${userId}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ public_token: publicToken }),
          }
        );

        const data = await response.json();
        console.log('OAuth success:', data);

        // Clean up localStorage
        localStorage.removeItem('plaid_link_token');

        // Notify parent and navigate
        if (onSuccess) {
          onSuccess(data);
        }
        navigate('/dashboard');
      } catch (error) {
        console.error('Error handling OAuth redirect:', error);
        navigate('/');
      }
    },
    onExit: (error, metadata) => {
      console.log('OAuth Link exited:', error, metadata);
      localStorage.removeItem('plaid_link_token');
      navigate('/');
    },
  };

  const { open, ready } = usePlaidLink(config);

  useEffect(() => {
    // Automatically open Link when ready (no user interaction needed)
    if (ready && isOAuthRedirect) {
      open();
    }
  }, [ready, open, isOAuthRedirect]);

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <h2>Completing bank connection...</h2>
      <p>Please wait while we finalize your OAuth authentication.</p>
    </div>
  );
};

export default PlaidOAuthRedirect;
```

### Store Link Token Before OAuth (Required)

Before opening Plaid Link, store the link token for OAuth redirects:

```jsx
// In PlaidLinkButton.jsx, before calling open():
useEffect(() => {
  if (linkToken) {
    localStorage.setItem('plaid_link_token', linkToken);
  }
}, [linkToken]);
```

### Axios API Service (Optional - Clean Architecture)

**File**: `src/services/plaidApi.js`

```javascript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const plaidApi = {
  // Create link token
  createLinkToken: async (userId) => {
    const response = await axios.post(
      `${API_BASE_URL}/api/plaid/create_link_token`,
      null,
      { params: { user_id: userId } }
    );
    return response.data;
  },

  // Exchange public token
  exchangePublicToken: async (publicToken, userId) => {
    const response = await axios.post(
      `${API_BASE_URL}/api/plaid/exchange_public_token`,
      { public_token: publicToken },
      { params: { user_id: userId } }
    );
    return response.data;
  },

  // Get account balances
  getAccounts: async (itemId) => {
    const response = await axios.get(
      `${API_BASE_URL}/api/plaid/accounts/${itemId}`
    );
    return response.data;
  },

  // Sync transactions
  syncTransactions: async (itemId) => {
    const response = await axios.post(
      `${API_BASE_URL}/api/plaid/transactions/sync/${itemId}`
    );
    return response.data;
  },
};
```

---

## Database Schema

### Recommended PostgreSQL Schema

```sql
-- Users table (your existing user table)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Plaid Items (represents user's connection to a bank)
CREATE TABLE plaid_items (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_id VARCHAR(255) UNIQUE NOT NULL,
    access_token TEXT NOT NULL,  -- ENCRYPT THIS IN PRODUCTION!
    institution_id VARCHAR(255),
    institution_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_synced_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',  -- active, login_required, error
    cursor TEXT,  -- For Transactions Sync API pagination

    CONSTRAINT unique_user_institution UNIQUE (user_id, institution_id)
);

-- Accounts (bank accounts associated with a Plaid Item)
CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    plaid_item_id INTEGER NOT NULL REFERENCES plaid_items(id) ON DELETE CASCADE,
    account_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    official_name VARCHAR(255),
    type VARCHAR(50),  -- depository, credit, loan, investment
    subtype VARCHAR(50),  -- checking, savings, credit card, etc.
    mask VARCHAR(10),  -- Last 4 digits
    current_balance DECIMAL(15, 2),
    available_balance DECIMAL(15, 2),
    currency_code VARCHAR(3) DEFAULT 'USD',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    date DATE NOT NULL,
    name VARCHAR(255) NOT NULL,
    amount DECIMAL(15, 2) NOT NULL,  -- Positive = money out, Negative = money in
    category_id VARCHAR(50),
    category TEXT[],  -- Array of categories
    pending BOOLEAN DEFAULT FALSE,
    merchant_name VARCHAR(255),
    payment_channel VARCHAR(50),  -- online, in store, other
    location_address VARCHAR(255),
    location_city VARCHAR(100),
    location_region VARCHAR(50),
    location_postal_code VARCHAR(20),
    location_country VARCHAR(2),
    iso_currency_code VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_plaid_items_user_id ON plaid_items(user_id);
CREATE INDEX idx_plaid_items_item_id ON plaid_items(item_id);
CREATE INDEX idx_accounts_plaid_item_id ON accounts(plaid_item_id);
CREATE INDEX idx_accounts_account_id ON accounts(account_id);
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_date ON transactions(date DESC);
CREATE INDEX idx_transactions_transaction_id ON transactions(transaction_id);
CREATE INDEX idx_transactions_pending ON transactions(pending);
```

### SQLAlchemy Models

**File**: `app/models.py`

```python
from sqlalchemy import Column, Integer, String, Text, DECIMAL, Boolean, TIMESTAMP, ForeignKey, ARRAY, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    plaid_items = relationship("PlaidItem", back_populates="user")


class PlaidItem(Base):
    __tablename__ = "plaid_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    item_id = Column(String(255), unique=True, nullable=False, index=True)
    access_token = Column(Text, nullable=False)  # TODO: Encrypt with Fernet or AWS KMS
    institution_id = Column(String(255))
    institution_name = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())
    last_synced_at = Column(TIMESTAMP)
    status = Column(String(50), default="active")
    cursor = Column(Text)  # For Transactions Sync API

    user = relationship("User", back_populates="plaid_items")
    accounts = relationship("Account", back_populates="plaid_item")


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    plaid_item_id = Column(Integer, ForeignKey("plaid_items.id", ondelete="CASCADE"), nullable=False)
    account_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    official_name = Column(String(255))
    type = Column(String(50))
    subtype = Column(String(50))
    mask = Column(String(10))
    current_balance = Column(DECIMAL(15, 2))
    available_balance = Column(DECIMAL(15, 2))
    currency_code = Column(String(3), default="USD")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    plaid_item = relationship("PlaidItem", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False)
    transaction_id = Column(String(255), unique=True, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    amount = Column(DECIMAL(15, 2), nullable=False)
    category_id = Column(String(50))
    category = Column(ARRAY(Text))
    pending = Column(Boolean, default=False, index=True)
    merchant_name = Column(String(255))
    payment_channel = Column(String(50))
    location_address = Column(String(255))
    location_city = Column(String(100))
    location_region = Column(String(50))
    location_postal_code = Column(String(20))
    location_country = Column(String(2))
    iso_currency_code = Column(String(3), default="USD")
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    account = relationship("Account", back_populates="transactions")
```

---

## Security Best Practices

### 1. Access Token Storage

**CRITICAL**: Never store access tokens in plaintext!

#### Encryption with Fernet (Python)

```bash
pip install cryptography
```

```python
from cryptography.fernet import Fernet
import os

# Generate and store encryption key (DO THIS ONCE, STORE IN ENV)
# encryption_key = Fernet.generate_key()
# Store in .env as ENCRYPTION_KEY=<key>

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY').encode()
cipher = Fernet(ENCRYPTION_KEY)

def encrypt_token(access_token: str) -> str:
    """Encrypt access token before storing in database."""
    return cipher.encrypt(access_token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypt access token when retrieving from database."""
    return cipher.decrypt(encrypted_token.encode()).decode()

# Usage:
# encrypted_access_token = encrypt_token(access_token)
# db.add(PlaidItem(..., access_token=encrypted_access_token))
```

### 2. Environment Variables

**Never commit secrets to version control!**

Add to `.gitignore`:
```
.env
.env.local
.env.*.local
```

Use environment-specific configs:
- `.env.development` - Sandbox environment
- `.env.production` - Production environment

### 3. HTTPS Only in Production

```python
# FastAPI HTTPS redirect middleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

if os.getenv('PLAID_ENV') == 'production':
    app.add_middleware(HTTPSRedirectMiddleware)
```

### 4. Rate Limiting

Implement rate limiting to prevent abuse:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/create_link_token")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def create_link_token(request: Request, user_id: str):
    # ... endpoint logic
```

### 5. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "https://yourdomain.com"  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 6. Input Validation

Use Pydantic models for strict validation:

```python
from pydantic import BaseModel, validator

class PublicTokenRequest(BaseModel):
    public_token: str

    @validator('public_token')
    def validate_public_token(cls, v):
        if not v.startswith('public-'):
            raise ValueError('Invalid public token format')
        return v
```

### 7. Secure Token Transmission

- Always use HTTPS in production
- Never log access tokens or public tokens
- Use secure, HTTP-only cookies for session tokens (if applicable)
- Implement CSRF protection

### 8. AWS Secrets Manager (Production Recommendation)

For production, use AWS Secrets Manager or similar:

```bash
pip install boto3
```

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage:
# plaid_secrets = get_secret('prod/plaid/credentials')
# PLAID_CLIENT_ID = plaid_secrets['client_id']
# PLAID_SECRET = plaid_secrets['secret']
```

---

## Wells Fargo & Venmo Support

### Wells Fargo

✅ **Fully Supported** by Plaid

- Uses **OAuth authentication flow** (requires `receivedRedirectUri` configuration)
- Supports **Auth** (account/routing numbers) and **Transactions** products
- Typically takes **2-3 minutes** for user to complete OAuth flow
- Plaid has 12,000+ supported institutions including all major US banks

### Venmo

✅ **Supported** via Plaid

- Venmo uses Plaid for instant bank verification
- When users connect Venmo to Wells Fargo, Plaid handles the connection
- Supports automatic balance checking and transaction verification
- Both instant verification (OAuth) and manual verification (micro-deposits) available

### OAuth Configuration for Wells Fargo

In your React component:

```jsx
const config = {
  token: linkToken,
  // REQUIRED for OAuth institutions like Wells Fargo
  receivedRedirectUri: window.location.href,
  onSuccess: onPlaidSuccess,
  onExit: onPlaidExit,
};
```

In your FastAPI backend:

```python
request = LinkTokenCreateRequest(
    products=[Products("auth"), Products("transactions")],
    client_name="Trader AI Dashboard",
    country_codes=[CountryCode("US")],
    language="en",
    user=LinkTokenCreateRequestUser(client_user_id=user_id),
    # REQUIRED for OAuth institutions
    redirect_uri="https://yourdomain.com/oauth-redirect",  # Production URL
)
```

### Testing Wells Fargo in Sandbox

Use these test credentials in Sandbox mode:
- **Institution**: Search for "Wells Fargo" in Plaid Link
- **Username**: `user_good`
- **Password**: `pass_good`

---

## Rate Limits & Development Environment

### Sandbox Environment (Free & Unlimited)

- **Cost**: FREE
- **Items**: Unlimited test Items
- **Institutions**: All test institutions available
- **Products**: All products (Auth, Transactions, Identity, etc.)
- **Link**: Full Plaid Link functionality

**Sandbox Rate Limits** (per minute):
- `/accounts/get`: 100/Item, 5,000/client
- `/account/balance/get`: 25/Item, 100/client
- `/auth/get`: 100/Item, 500/client
- `/identity/get`: 100/Item, 1,000/client
- `/institutions/get`: 10/client
- `/transactions/sync`: 100/Item, 1,000/client

### Development Mode (Limited Production)

- **Cost**: FREE for first 200 API calls
- **Items**: Up to 100 live Items
- **Use Case**: Testing with real bank accounts before full production

### Production Pricing (2025)

- **Free Tier**: First 200 API calls
- **Auth Product**: Pay-per-use after free tier
- **Transactions Product**: Pay-per-use after free tier

⚠️ **Note**: For trader-ai dashboard development, stay in **Sandbox** mode until ready to test with real banks.

---

## Common Gotchas & Solutions

### 1. OAuth Redirect Loop

**Problem**: User gets stuck in redirect loop after OAuth

**Solution**:
- Always store `link_token` in localStorage before opening Link
- Check for `oauth_state_id` in URL query params
- Use `receivedRedirectUri` with full URL including query params

```jsx
// Before opening Link
localStorage.setItem('plaid_link_token', linkToken);

// In OAuth redirect component
const isOAuthRedirect = window.location.href.includes('oauth_state_id');
const receivedRedirectUri = window.location.href;  // Full URL with params
```

### 2. Token Expiration

**Problem**: `link_token` expires after 4 hours, `public_token` expires after 30 minutes

**Solution**:
- Generate a fresh `link_token` each time user opens Plaid Link
- Exchange `public_token` immediately after receiving it
- Don't cache tokens on frontend

### 3. ITEM_LOGIN_REQUIRED Error

**Problem**: Access token returns `ITEM_LOGIN_REQUIRED` error

**Cause**: User changed password or revoked access at their bank

**Solution**:
- Detect this error code in your backend
- Generate a new `link_token` with `update` mode:

```python
request = LinkTokenCreateRequest(
    products=[Products("auth"), Products("transactions")],
    client_name="Trader AI Dashboard",
    country_codes=[CountryCode("US")],
    language="en",
    user=LinkTokenCreateRequestUser(client_user_id=user_id),
    access_token=stored_access_token  # Triggers update mode
)
```

- Prompt user to re-authenticate via Plaid Link

### 4. Transactions Not Syncing

**Problem**: `/transactions/sync` returns empty results

**Causes & Solutions**:
- **Cursor not stored**: Store `next_cursor` from each sync in database
- **First sync**: Start with empty cursor `""` for initial sync
- **Pagination**: Loop until `has_more` is `False`

```python
cursor = plaid_item.cursor or ""  # Start with stored cursor or empty string
has_more = True

while has_more:
    request = TransactionsSyncRequest(access_token=access_token, cursor=cursor)
    response = plaid_client.transactions_sync(request)

    # Process added/modified/removed transactions

    cursor = response.next_cursor
    has_more = response.has_more

# Store cursor for next sync
plaid_item.cursor = cursor
db.commit()
```

### 5. CORS Errors in Development

**Problem**: Browser blocks API requests with CORS error

**Solution**:
- Add CORS middleware to FastAPI:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 6. Webhooks Not Received

**Problem**: Not receiving webhook notifications for transaction updates

**Solution**:
- Configure webhook URL in Plaid Dashboard (not available in Sandbox)
- Use ngrok for local development webhook testing:

```bash
ngrok http 8000
# Use ngrok URL in Plaid Dashboard: https://abc123.ngrok.io/api/webhooks/plaid
```

```python
# Webhook endpoint
@router.post("/webhooks/plaid")
async def plaid_webhook(request: Request):
    payload = await request.json()
    webhook_type = payload.get('webhook_type')
    webhook_code = payload.get('webhook_code')
    item_id = payload.get('item_id')

    if webhook_type == 'TRANSACTIONS':
        if webhook_code in ['INITIAL_UPDATE', 'HISTORICAL_UPDATE', 'DEFAULT_UPDATE']:
            # Trigger transaction sync
            await sync_transactions_task(item_id)

    return {"status": "received"}
```

### 7. Database Connection Errors

**Problem**: Database connections exhausted under load

**Solution**:
- Use connection pooling with SQLAlchemy:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections before use
)
```

### 8. Encrypted Token Retrieval

**Problem**: Forgot to decrypt token when retrieving from database

**Solution**:
- Create helper functions:

```python
def get_access_token(db: Session, item_id: str) -> str:
    plaid_item = db.query(PlaidItem).filter(PlaidItem.item_id == item_id).first()
    if not plaid_item:
        raise ValueError("Item not found")
    return decrypt_token(plaid_item.access_token)
```

---

## Resources

### Official Documentation
- **Plaid Quickstart**: https://plaid.com/docs/quickstart/
- **Plaid API Reference**: https://plaid.com/docs/api/
- **Python SDK (plaid-python)**: https://github.com/plaid/plaid-python
- **React SDK (react-plaid-link)**: https://github.com/plaid/react-plaid-link
- **OAuth Guide**: https://plaid.com/docs/link/oauth/

### Code Examples
- **Plaid Quickstart (Python + React)**: https://github.com/plaid/quickstart
- **Plaid Pattern (PostgreSQL + React)**: https://github.com/plaid/pattern

### API Versions
- **Current Version**: 2020-09-14
- **SDK Version**: plaid-python 37.0.0+ (updated monthly)

### Rate Limits Documentation
- https://plaid.com/docs/errors/rate-limit-exceeded/

### Security Best Practices
- **AWS Secrets Manager**: https://aws.amazon.com/secrets-manager/
- **Encryption (Fernet)**: https://cryptography.io/en/latest/fernet/

### Support
- **Plaid Dashboard**: https://dashboard.plaid.com
- **Plaid Community**: https://community.plaid.com

---

## Quick Start Checklist

### Backend Setup
- [ ] Install `plaid-python` SDK
- [ ] Configure environment variables (`.env`)
- [ ] Create Plaid client configuration
- [ ] Implement `/create_link_token` endpoint
- [ ] Implement `/exchange_public_token` endpoint
- [ ] Implement `/accounts/{item_id}` endpoint
- [ ] Implement `/transactions/sync/{item_id}` endpoint
- [ ] Set up PostgreSQL database
- [ ] Create database schema (tables: `plaid_items`, `accounts`, `transactions`)
- [ ] Implement access token encryption

### Frontend Setup
- [ ] Install `react-plaid-link` package
- [ ] Create `PlaidLinkButton` component
- [ ] Implement OAuth redirect handling
- [ ] Store link token in localStorage before OAuth
- [ ] Configure `receivedRedirectUri` for OAuth
- [ ] Test with Sandbox credentials

### Security
- [ ] Enable HTTPS in production
- [ ] Encrypt access tokens at rest
- [ ] Use AWS Secrets Manager (production)
- [ ] Implement rate limiting
- [ ] Configure CORS properly
- [ ] Add input validation

### Testing
- [ ] Test in Sandbox with test credentials
- [ ] Test Wells Fargo OAuth flow
- [ ] Test transaction sync
- [ ] Test account balance retrieval
- [ ] Test error handling (invalid tokens, login required)

### Production Readiness
- [ ] Switch to Production environment
- [ ] Configure production redirect URIs
- [ ] Set up webhook endpoint (optional)
- [ ] Monitor rate limits
- [ ] Set up error logging (Sentry, CloudWatch)
- [ ] Load test API endpoints

---

**Last Updated**: November 7, 2025
**Plaid SDK Version**: plaid-python 37.0.0
**API Version**: 2020-09-14

For questions or issues, refer to the [Plaid Community](https://community.plaid.com) or [Plaid API Documentation](https://plaid.com/docs/).