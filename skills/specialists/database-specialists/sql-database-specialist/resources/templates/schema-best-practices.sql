-- Database Schema Best Practices Template
-- PostgreSQL/MySQL compatible examples
-- This template demonstrates production-ready schema design

-- =============================================================================
-- 1. USERS TABLE - Best Practices for User Management
-- =============================================================================

CREATE TABLE users (
  -- Primary key: Use BIGSERIAL/BIGINT for future-proofing
  id BIGSERIAL PRIMARY KEY,

  -- Email: Indexed, unique, lowercase for case-insensitive search
  email VARCHAR(255) NOT NULL UNIQUE,

  -- Passwords: Never store plaintext (hash with bcrypt/Argon2)
  password_hash VARCHAR(255) NOT NULL,

  -- Profile fields
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  display_name VARCHAR(200),

  -- Status management
  status VARCHAR(20) NOT NULL DEFAULT 'active',
  email_verified BOOLEAN NOT NULL DEFAULT false,
  email_verified_at TIMESTAMPTZ,

  -- Timestamps: Always include for audit trail
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_login_at TIMESTAMPTZ,

  -- Soft delete (prefer over hard delete)
  deleted_at TIMESTAMPTZ,

  -- Constraints
  CONSTRAINT chk_users_status CHECK (status IN ('active', 'inactive', 'suspended', 'deleted')),
  CONSTRAINT chk_users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Indexes for common queries
CREATE INDEX idx_users_email_lower ON users (LOWER(email));
CREATE INDEX idx_users_status ON users (status) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created_at ON users (created_at DESC);

-- Full-text search on names (PostgreSQL)
CREATE INDEX idx_users_search ON users USING GIN (
  to_tsvector('english', first_name || ' ' || last_name)
);

COMMENT ON TABLE users IS 'User accounts with authentication and profile data';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt/Argon2 password hash (never plaintext)';
COMMENT ON COLUMN users.deleted_at IS 'Soft delete timestamp (NULL = active)';

-- =============================================================================
-- 2. ORDERS TABLE - E-commerce Best Practices
-- =============================================================================

CREATE TABLE orders (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL,

  -- Order details
  order_number VARCHAR(50) NOT NULL UNIQUE,  -- Human-readable order ID
  status VARCHAR(20) NOT NULL DEFAULT 'pending',

  -- Money: Use DECIMAL, never FLOAT
  subtotal DECIMAL(10, 2) NOT NULL CHECK (subtotal >= 0),
  tax DECIMAL(10, 2) NOT NULL DEFAULT 0 CHECK (tax >= 0),
  shipping DECIMAL(10, 2) NOT NULL DEFAULT 0 CHECK (shipping >= 0),
  total DECIMAL(10, 2) NOT NULL CHECK (total >= 0),

  -- Currency code (ISO 4217)
  currency VARCHAR(3) NOT NULL DEFAULT 'USD',

  -- Shipping address (denormalized for historical record)
  shipping_name VARCHAR(200),
  shipping_address_line1 VARCHAR(255),
  shipping_address_line2 VARCHAR(255),
  shipping_city VARCHAR(100),
  shipping_state VARCHAR(100),
  shipping_postal_code VARCHAR(20),
  shipping_country VARCHAR(2),  -- ISO 3166-1 alpha-2

  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,

  -- Foreign keys
  FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE RESTRICT,

  -- Constraints
  CONSTRAINT chk_orders_status CHECK (status IN ('pending', 'processing', 'completed', 'cancelled', 'refunded')),
  CONSTRAINT chk_orders_total CHECK (total = subtotal + tax + shipping)
);

-- Indexes
CREATE INDEX idx_orders_user_id ON orders (user_id);
CREATE INDEX idx_orders_status ON orders (status);
CREATE INDEX idx_orders_created_at ON orders (created_at DESC);
CREATE INDEX idx_orders_order_number ON orders (order_number);

-- Compound index for common query pattern
CREATE INDEX idx_orders_user_status_date ON orders (user_id, status, created_at DESC);

-- Partial index for active orders only
CREATE INDEX idx_orders_active ON orders (user_id, created_at DESC)
WHERE status IN ('pending', 'processing');

-- Covering index to avoid heap fetches
CREATE INDEX idx_orders_user_covering ON orders (user_id, created_at DESC)
INCLUDE (order_number, status, total);

COMMENT ON TABLE orders IS 'Customer orders with denormalized shipping data for historical accuracy';

-- =============================================================================
-- 3. ORDER_ITEMS TABLE - Join Table Best Practices
-- =============================================================================

CREATE TABLE order_items (
  id BIGSERIAL PRIMARY KEY,
  order_id BIGINT NOT NULL,
  product_id BIGINT NOT NULL,

  -- Denormalize product details for historical accuracy
  product_name VARCHAR(255) NOT NULL,
  product_sku VARCHAR(100),

  quantity INT NOT NULL CHECK (quantity > 0),
  unit_price DECIMAL(10, 2) NOT NULL CHECK (unit_price >= 0),
  subtotal DECIMAL(10, 2) NOT NULL CHECK (subtotal >= 0),

  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Foreign keys
  FOREIGN KEY (order_id) REFERENCES orders (id) ON DELETE CASCADE,
  FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE RESTRICT,

  -- Constraints
  CONSTRAINT chk_order_items_subtotal CHECK (subtotal = quantity * unit_price)
);

-- Indexes (foreign keys MUST be indexed)
CREATE INDEX idx_order_items_order_id ON order_items (order_id);
CREATE INDEX idx_order_items_product_id ON order_items (product_id);

COMMENT ON TABLE order_items IS 'Order line items with denormalized product data';

-- =============================================================================
-- 4. PRODUCTS TABLE - Inventory Management
-- =============================================================================

CREATE TABLE products (
  id BIGSERIAL PRIMARY KEY,

  -- Product identification
  name VARCHAR(255) NOT NULL,
  slug VARCHAR(255) NOT NULL UNIQUE,  -- URL-friendly identifier
  sku VARCHAR(100) UNIQUE,
  barcode VARCHAR(100),

  -- Description
  description TEXT,
  short_description VARCHAR(500),

  -- Pricing
  price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
  cost DECIMAL(10, 2) CHECK (cost >= 0),
  compare_at_price DECIMAL(10, 2) CHECK (compare_at_price >= price),

  -- Inventory
  stock_quantity INT NOT NULL DEFAULT 0 CHECK (stock_quantity >= 0),
  low_stock_threshold INT DEFAULT 10,

  -- Status
  is_active BOOLEAN NOT NULL DEFAULT true,
  is_featured BOOLEAN NOT NULL DEFAULT false,

  -- Metadata (JSONB for flexible attributes)
  metadata JSONB,

  -- Timestamps
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at TIMESTAMPTZ,

  -- Full-text search vector
  search_vector tsvector
);

-- Indexes
CREATE INDEX idx_products_slug ON products (slug);
CREATE INDEX idx_products_sku ON products (sku);
CREATE INDEX idx_products_is_active ON products (is_active) WHERE deleted_at IS NULL;
CREATE INDEX idx_products_stock ON products (stock_quantity) WHERE stock_quantity < 10;

-- GIN index for JSONB metadata queries
CREATE INDEX idx_products_metadata ON products USING GIN (metadata);

-- Full-text search
CREATE INDEX idx_products_search ON products USING GIN (search_vector);

-- Trigger to update search_vector
CREATE OR REPLACE FUNCTION products_search_trigger() RETURNS trigger AS $$
BEGIN
  NEW.search_vector := to_tsvector('english', COALESCE(NEW.name, '') || ' ' || COALESCE(NEW.description, ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER products_search_update
  BEFORE INSERT OR UPDATE ON products
  FOR EACH ROW EXECUTE FUNCTION products_search_trigger();

-- =============================================================================
-- 5. PARTITIONING EXAMPLE - Time-Series Data
-- =============================================================================

-- Events table partitioned by month
CREATE TABLE events (
  id BIGSERIAL,
  user_id BIGINT NOT NULL,
  event_type VARCHAR(50) NOT NULL,
  event_data JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (id, created_at)  -- Include partition key in PK
) PARTITION BY RANGE (created_at);

-- Create partitions
CREATE TABLE events_2024_01 PARTITION OF events
  FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE events_2024_02 PARTITION OF events
  FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Index on each partition
CREATE INDEX idx_events_2024_01_user_id ON events_2024_01 (user_id);
CREATE INDEX idx_events_2024_02_user_id ON events_2024_02 (user_id);

-- =============================================================================
-- 6. AUDIT LOG TABLE - Change Tracking
-- =============================================================================

CREATE TABLE audit_log (
  id BIGSERIAL PRIMARY KEY,
  table_name VARCHAR(100) NOT NULL,
  record_id BIGINT NOT NULL,
  action VARCHAR(10) NOT NULL,  -- INSERT, UPDATE, DELETE
  old_values JSONB,
  new_values JSONB,
  user_id BIGINT,
  ip_address INET,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT chk_audit_action CHECK (action IN ('INSERT', 'UPDATE', 'DELETE'))
);

-- Indexes
CREATE INDEX idx_audit_log_table_record ON audit_log (table_name, record_id);
CREATE INDEX idx_audit_log_user_id ON audit_log (user_id);
CREATE INDEX idx_audit_log_created_at ON audit_log (created_at DESC);

-- =============================================================================
-- 7. UPDATED_AT TRIGGER - Automatic Timestamp Updates
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to all tables with updated_at
CREATE TRIGGER users_updated_at BEFORE UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER orders_updated_at BEFORE UPDATE ON orders
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER products_updated_at BEFORE UPDATE ON products
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 8. VACUUM AND MAINTENANCE SCHEDULE
-- =============================================================================

-- PostgreSQL: Configure autovacuum
ALTER TABLE users SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE orders SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE events SET (autovacuum_vacuum_scale_factor = 0.2);

-- =============================================================================
-- BEST PRACTICES SUMMARY
-- =============================================================================

/*
1. ✅ Use BIGSERIAL/BIGINT for primary keys (future-proof)
2. ✅ Always index foreign keys
3. ✅ Use DECIMAL for money (never FLOAT)
4. ✅ Add timestamps (created_at, updated_at)
5. ✅ Implement soft deletes (deleted_at)
6. ✅ Add CHECK constraints for data validation
7. ✅ Use descriptive names (snake_case)
8. ✅ Denormalize for historical accuracy (orders, order_items)
9. ✅ Create covering indexes for hot queries
10. ✅ Use partial indexes for filtered queries
11. ✅ Add comments for documentation
12. ✅ Use JSONB for flexible metadata
13. ✅ Partition large time-series tables
14. ✅ Implement audit logging
15. ✅ Use triggers for automatic timestamps
*/
