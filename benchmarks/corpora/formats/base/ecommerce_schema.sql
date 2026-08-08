-- ecommerce_schema.sql
-- Database schema for the NovaMart e-commerce platform.
-- PostgreSQL 15+ compatible.

-- ============================================================
-- Customers table
-- Stores registered customer information.
-- ============================================================
CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    email         VARCHAR(255) NOT NULL UNIQUE,
    first_name    VARCHAR(100) NOT NULL,
    last_name     VARCHAR(100) NOT NULL,
    phone         VARCHAR(20),
    created_at    TIMESTAMP DEFAULT NOW(),
    loyalty_tier  VARCHAR(20) DEFAULT 'bronze'
        CHECK (loyalty_tier IN ('bronze', 'silver', 'gold', 'platinum'))
);

CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_loyalty ON customers(loyalty_tier);

-- ============================================================
-- Products table
-- Catalog of all items available for purchase.
-- ============================================================
CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    sku           VARCHAR(50) NOT NULL UNIQUE,
    name          VARCHAR(255) NOT NULL,
    description   TEXT,
    category      VARCHAR(100) NOT NULL,
    price_cents   INTEGER NOT NULL CHECK (price_cents >= 0),
    weight_grams  INTEGER,
    stock_count   INTEGER DEFAULT 0,
    is_active     BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE;

-- ============================================================
-- Orders table
-- Each order belongs to exactly one customer.
-- status tracks: pending -> confirmed -> shipped -> delivered
-- ============================================================
CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    status        VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled')),
    total_cents   INTEGER NOT NULL CHECK (total_cents >= 0),
    shipping_addr TEXT,
    placed_at     TIMESTAMP DEFAULT NOW(),
    shipped_at    TIMESTAMP,
    delivered_at  TIMESTAMP
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_placed ON orders(placed_at DESC);

-- ============================================================
-- Order items (join table)
-- Links orders to products with quantity and unit price.
-- ============================================================
CREATE TABLE order_items (
    item_id       SERIAL PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    INTEGER NOT NULL CHECK (unit_price >= 0),
    UNIQUE (order_id, product_id)
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- ============================================================
-- Reviews table
-- Customers can leave one review per product.
-- rating is 1-5 stars.
-- ============================================================
CREATE TABLE reviews (
    review_id     SERIAL PRIMARY KEY,
    customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    rating        SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title         VARCHAR(200),
    body          TEXT,
    created_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE (customer_id, product_id)
);

CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- ============================================================
-- Inventory log
-- Tracks stock changes (restock, sale, return, adjustment).
-- ============================================================
CREATE TABLE inventory_log (
    log_id        SERIAL PRIMARY KEY,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    change_type   VARCHAR(20) NOT NULL
        CHECK (change_type IN ('restock', 'sale', 'return', 'adjustment')),
    quantity      INTEGER NOT NULL,
    notes         TEXT,
    logged_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_inventory_product ON inventory_log(product_id);
CREATE INDEX idx_inventory_type ON inventory_log(change_type);
