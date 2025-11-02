# Example 2: REST API Integration Validation

## Scenario

A development team built a multi-component REST API for an e-commerce platform with separate microservices for authentication, product catalog, shopping cart, and payment processing. Each service has comprehensive unit tests that pass with 95%+ coverage. However, end-to-end workflows fail mysteriously when services interact - customers can't complete purchases even though each individual endpoint works correctly in isolation.

## Problem Statement

Individual API endpoints respond correctly when tested in isolation, but integration workflows fail unpredictably. The team needs systematic validation of the full integration stack to identify coordination issues between services that unit tests cannot detect.

**Architecture:**
```
Client → API Gateway → Auth Service
                    → Product Service
                    → Cart Service
                    → Payment Service
```

**Initial Symptom:**
```bash
curl -X POST https://api.example.com/checkout \
  -H "Authorization: Bearer <token>" \
  -d '{"cart_id": "abc123"}'

Response: 401 Unauthorized
# But the token validates correctly when tested directly!
```

## Audit Process

### Step 1: Setup Sandbox

Create E2B sandbox with full microservices stack:

```bash
# Initialize sandbox with complete environment
npx claude-flow@alpha hooks pre-task --description "Validate e-commerce API integration"

# Create docker-compose for all services
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: testpass
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  auth-service:
    build: ./services/auth
    environment:
      JWT_SECRET: test-secret-key
      JWT_EXPIRY: 3600
      DATABASE_URL: postgresql://postgres:testpass@postgres:5432/auth
    ports:
      - "3001:3000"
    depends_on:
      - postgres
      - redis

  product-service:
    build: ./services/products
    environment:
      DATABASE_URL: postgresql://postgres:testpass@postgres:5432/products
    ports:
      - "3002:3000"
    depends_on:
      - postgres

  cart-service:
    build: ./services/cart
    environment:
      REDIS_URL: redis://redis:6379
      AUTH_SERVICE_URL: http://auth-service:3000
    ports:
      - "3003:3000"
    depends_on:
      - redis
      - auth-service

  payment-service:
    build: ./services/payment
    environment:
      AUTH_SERVICE_URL: http://auth-service:3000
      CART_SERVICE_URL: http://cart-service:3000
    ports:
      - "3004:3000"
    depends_on:
      - auth-service
      - cart-service

  api-gateway:
    build: ./gateway
    environment:
      AUTH_SERVICE_URL: http://auth-service:3000
      PRODUCT_SERVICE_URL: http://product-service:3000
      CART_SERVICE_URL: http://cart-service:3000
      PAYMENT_SERVICE_URL: http://payment-service:3000
    ports:
      - "8080:8080"
    depends_on:
      - auth-service
      - product-service
      - cart-service
      - payment-service
EOF

# Start all services
docker-compose up -d

# Wait for health checks
sleep 10
```

### Step 2: Generate Test Cases

Create comprehensive integration test suite:

```python
# tests/integration/test_checkout_flow.py
import requests
import time
import pytest

BASE_URL = "http://localhost:8080"

class TestCheckoutIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test user and data."""
        self.user_email = f"test_{int(time.time())}@example.com"
        self.password = "TestPass123!"

        # Register user
        response = requests.post(f"{BASE_URL}/auth/register", json={
            "email": self.user_email,
            "password": self.password
        })
        assert response.status_code == 201

        # Login and get token
        response = requests.post(f"{BASE_URL}/auth/login", json={
            "email": self.user_email,
            "password": self.password
        })
        assert response.status_code == 200
        self.token = response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def test_complete_purchase_flow(self):
        """Test full purchase workflow: browse → cart → checkout."""

        # Step 1: Browse products
        response = requests.get(f"{BASE_URL}/products", headers=self.headers)
        assert response.status_code == 200
        products = response.json()
        assert len(products) > 0
        product_id = products[0]["id"]

        # Step 2: Add to cart
        response = requests.post(
            f"{BASE_URL}/cart/items",
            headers=self.headers,
            json={"product_id": product_id, "quantity": 2}
        )
        assert response.status_code == 201
        cart_id = response.json()["cart_id"]

        # Step 3: View cart
        response = requests.get(f"{BASE_URL}/cart/{cart_id}", headers=self.headers)
        assert response.status_code == 200
        cart = response.json()
        assert len(cart["items"]) == 1

        # Step 4: Checkout - THIS FAILS!
        response = requests.post(
            f"{BASE_URL}/checkout",
            headers=self.headers,
            json={"cart_id": cart_id, "payment_method": "credit_card"}
        )

        print(f"\n=== CHECKOUT REQUEST ===")
        print(f"URL: {BASE_URL}/checkout")
        print(f"Headers: {self.headers}")
        print(f"Body: {{'cart_id': '{cart_id}', 'payment_method': 'credit_card'}}")
        print(f"\n=== CHECKOUT RESPONSE ===")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")

        # EXPECTED: 200 OK with order_id
        # ACTUAL: 401 Unauthorized
        assert response.status_code == 200, f"Checkout failed: {response.text}"

    def test_token_validation_direct(self):
        """Verify token validates when tested directly."""
        response = requests.get(
            f"{BASE_URL}/auth/verify",
            headers=self.headers
        )
        assert response.status_code == 200
        assert response.json()["valid"] == True
```

### Step 3: Execute & Monitor

Run integration tests with request tracing:

```bash
# Enable debug logging on all services
docker-compose logs -f > service_logs.txt &

# Run integration tests
python -m pytest tests/integration/test_checkout_flow.py -vv -s

# Capture network traffic
docker run --rm --net=host \
  -v $(pwd):/data \
  nicolaka/netshoot \
  tcpdump -i any -w /data/api_traffic.pcap 'port 8080 or port 3001 or port 3003 or port 3004'
```

**Test Output:**
```
tests/integration/test_checkout_flow.py::TestCheckoutIntegration::test_complete_purchase_flow

=== CHECKOUT REQUEST ===
URL: http://localhost:8080/checkout
Headers: {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'}
Body: {'cart_id': 'cart_abc123', 'payment_method': 'credit_card'}

=== CHECKOUT RESPONSE ===
Status: 401
Body: {"error": "Invalid or expired token"}

FAILED - AssertionError: Checkout failed: {"error": "Invalid or expired token"}
```

### Step 4: Analyze Results

**Service Logs Analysis:**

```bash
# Extract relevant log entries
grep -A 5 "checkout" service_logs.txt

# auth-service.log
2024-01-15 10:23:45 [INFO] Token validation request from 172.18.0.6
2024-01-15 10:23:45 [INFO] Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
2024-01-15 10:23:45 [DEBUG] Token issued at: 1705318980
2024-01-15 10:23:45 [DEBUG] Token expires at: 1705322580
2024-01-15 10:23:45 [DEBUG] Current time: 1705322625
2024-01-15 10:23:45 [ERROR] Token expired (45 seconds ago)
2024-01-15 10:23:45 [INFO] Validation result: FAILED

# payment-service.log
2024-01-15 10:23:45 [INFO] Checkout request received
2024-01-15 10:23:45 [DEBUG] Validating token with auth service
2024-01-15 10:23:45 [ERROR] Auth service returned 401
2024-01-15 10:23:45 [INFO] Rejecting checkout - invalid token
```

**Timeline Analysis:**
```
10:23:00 - User registers (token issued)
10:23:05 - User logs in (same token reused - BUG #1)
10:23:10 - Browse products (token valid, 3590s remaining)
10:23:15 - Add to cart (token valid, 3585s remaining)
10:23:25 - View cart (token valid, 3575s remaining)
10:23:45 - Checkout (token EXPIRED - only had 3600s = 1 hour TTL)
```

**Root Cause Identified:**

1. **Token Expiry Issue**: JWT tokens have 1-hour expiry but test workflow takes ~45 seconds
2. **Service Coordination Issue**: Payment service validates token independently, not using gateway's validation
3. **Time Synchronization**: Services are checking token expiry against their own clocks
4. **No Token Refresh**: Gateway doesn't refresh tokens before forwarding to downstream services

**Trace Service-to-Service Calls:**

```python
# Add instrumentation to payment service
# payment-service/middleware/auth.py (BEFORE FIX)
def validate_token(token):
    # Each service validates independently
    response = requests.get(
        f"{AUTH_SERVICE_URL}/verify",
        headers={"Authorization": f"Bearer {token}"}
    )
    return response.status_code == 200
```

**Problems Found:**
1. Payment service validates token **after** gateway already validated it (redundant)
2. Time passes between gateway validation and payment validation
3. Token created during setup expires during multi-step workflow
4. No token refresh mechanism exists

### Step 5: Debug & Fix

**Fix 1: Gateway Token Refresh**

```javascript
// gateway/middleware/auth.js
async function authenticateRequest(req, res, next) {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
        return res.status(401).json({ error: 'No token provided' });
    }

    try {
        // Validate token
        const validation = await axios.get(`${AUTH_SERVICE_URL}/verify`, {
            headers: { Authorization: `Bearer ${token}` }
        });

        if (!validation.data.valid) {
            return res.status(401).json({ error: 'Invalid token' });
        }

        // NEW: Check if token expires soon (< 5 minutes remaining)
        const expiresIn = validation.data.expires_at - Date.now() / 1000;

        if (expiresIn < 300) {
            // Refresh token before forwarding request
            const refreshResponse = await axios.post(`${AUTH_SERVICE_URL}/refresh`, {
                token: token
            });

            const newToken = refreshResponse.data.token;

            // Forward with refreshed token
            req.headers.authorization = `Bearer ${newToken}`;

            // Also send refresh instruction to client
            res.setHeader('X-Token-Refreshed', newToken);
        }

        req.user = validation.data.user;
        next();

    } catch (error) {
        return res.status(401).json({ error: 'Token validation failed' });
    }
}
```

**Fix 2: Service-to-Service Trust**

```python
# payment-service/middleware/auth.py (AFTER FIX)
def validate_token(token):
    # Trust gateway's validation - just decode without re-validating
    # Gateway already checked expiry and refreshed if needed

    # Add gateway signature validation instead
    gateway_signature = request.headers.get('X-Gateway-Signature')

    if not verify_gateway_signature(gateway_signature, token):
        raise Unauthorized('Invalid gateway signature')

    # Decode token (don't validate expiry - gateway handled it)
    payload = jwt.decode(token, options={"verify_exp": False})
    return payload
```

**Fix 3: Clock Synchronization**

```yaml
# docker-compose.yml - Add NTP sync
version: '3.8'
services:
  auth-service:
    # ... existing config ...
    volumes:
      - /etc/localtime:/etc/localtime:ro  # Sync with host clock
    environment:
      TZ: UTC  # Use UTC everywhere
```

### Step 6: Verify Fix

Re-run integration tests with fixes deployed:

```bash
# Rebuild services with fixes
docker-compose down
docker-compose build
docker-compose up -d

# Run integration tests again
python -m pytest tests/integration/test_checkout_flow.py -vv -s
```

**Success Output:**
```
tests/integration/test_checkout_flow.py::TestCheckoutIntegration::test_complete_purchase_flow

=== CHECKOUT REQUEST ===
URL: http://localhost:8080/checkout
Headers: {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'}
Body: {'cart_id': 'cart_abc123', 'payment_method': 'credit_card'}

=== CHECKOUT RESPONSE ===
Status: 200
Headers: {'X-Token-Refreshed': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'}
Body: {"order_id": "ord_xyz789", "status": "processing", "amount": 129.99}

PASSED ✓
```

**Additional Stress Tests:**

```python
def test_long_running_workflow(self):
    """Test workflow that spans token expiry window."""
    # Add items to cart
    response = requests.post(f"{BASE_URL}/cart/items", ...)
    time.sleep(3660)  # Wait 61 minutes (token should expire)

    # Should still work due to token refresh
    response = requests.post(f"{BASE_URL}/checkout", ...)
    assert response.status_code == 200

def test_concurrent_checkouts(self):
    """Test multiple simultaneous checkouts."""
    import concurrent.futures

    def checkout_flow(user_id):
        # Complete checkout workflow
        return requests.post(f"{BASE_URL}/checkout", ...)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(checkout_flow, i) for i in range(10)]
        results = [f.result() for f in futures]

    # All should succeed
    assert all(r.status_code == 200 for r in results)
```

## Outcome

**What Was Discovered:**
- Services validated tokens independently, creating race conditions with token expiry
- Multi-step workflows caused tokens to expire mid-transaction
- No token refresh mechanism existed in gateway
- Service-to-service authentication was redundant and fragile

**How It Helped:**
1. **Full-stack sandbox** replicated production conditions exactly
2. **Integration tests** exposed coordination issues invisible to unit tests
3. **Request tracing** revealed the exact failure point in service chain
4. **Log analysis** showed timing issues between services
5. **Systematic fixes** addressed root causes, not symptoms

**Production Impact:**
- Checkout success rate increased from 73% to 99.7%
- Average checkout time reduced by 1.2 seconds (no redundant auth calls)
- Token expiry issues eliminated across all workflows
- System handles long-running user sessions reliably

## Key Takeaways

1. **Unit tests are insufficient**: Each service worked individually but integration failed
2. **Sandbox full stack testing is essential**: Only way to catch coordination issues
3. **Monitor service-to-service calls**: Hidden communication issues cause mysterious failures
4. **Time is a dependency**: Clock synchronization matters in distributed systems
5. **Trust but verify**: Gateway validation + service trust is better than redundant validation
6. **Test realistic workflows**: Short test flows missed expiry issues that real users hit

**When to Apply This Pattern:**
- After building microservices architecture
- When unit tests pass but system fails end-to-end
- For any distributed system with service-to-service communication
- Before deploying multi-step user workflows
- When debugging "works in isolation, fails integrated" issues
- For systems with authentication/authorization across services
