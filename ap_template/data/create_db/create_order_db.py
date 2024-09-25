import os
import psycopg2
from psycopg2.extras import execute_values
from faker import Faker
import random

from dotenv import load_dotenv
from config.load_config import load_config

config_dict = load_config()

pg_config = config_dict["database"]
host = pg_config["pg_host"]
port = pg_config["pg_port"]
dbname = pg_config["pg_dbname"]
user = pg_config["pg_user"]

load_dotenv()
password = os.getenv("PG_PASSWORD")

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname=dbname, 
    user=user, 
    password=password, 
    host=host, 
    port=port
)
cur = conn.cursor()

# Step 1: Create the tables
create_tables_query = """
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    parent_category_id INT
);

CREATE TABLE IF NOT EXISTS countries (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS merchants (
    id SERIAL PRIMARY KEY,
    country_code INT REFERENCES countries(id),
    status VARCHAR(50),
    merchant_name VARCHAR(255),
    address TEXT,
    website_url VARCHAR(255),
    phone_number VARCHAR(50),
    email VARCHAR(255),
    logo_url VARCHAR(255),
    created_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    brand VARCHAR(255),
    color VARCHAR(50),
    weight DOUBLE PRECISION,
    dimensions VARCHAR(100),
    rating INT,
    merchant_id INT REFERENCES merchants(id),
    price DOUBLE PRECISION,
    created_at TIMESTAMP,
    category_id INT REFERENCES categories(id)
);

CREATE TABLE IF NOT EXISTS shipping_carriers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    tracking_url VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(255),
    email VARCHAR(255),
    username VARCHAR(255),
    phone_number VARCHAR(50),
    last_login_at TIMESTAMP,
    avatar_url TEXT,
    created_at TIMESTAMP,
    country_code INT REFERENCES countries(id)
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    status VARCHAR(50),
    created_at TIMESTAMP,
    total_sum DOUBLE PRECISION,
    shipping_address TEXT,
    billing_address TEXT,
    payment_method VARCHAR(50),
    payment_status VARCHAR(50),
    shipping_carrier_id INT REFERENCES shipping_carriers(id)
);

CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_id INT REFERENCES products(id),
    quantity INT,
    price DOUBLE PRECISION,
    sum DOUBLE PRECISION
);
"""

cur.execute(create_tables_query)
conn.commit()

# Step 2: Generate random data
fake = Faker()

# Insert random data into 'countries' table
countries = [(fake.country(),) for _ in range(10)]
insert_countries_query = "INSERT INTO countries (name) VALUES %s RETURNING id"
cur.execute("TRUNCATE TABLE countries RESTART IDENTITY CASCADE")
execute_values(cur, insert_countries_query, countries)
country_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'categories' table
categories = [(fake.word(), random.choice([None] + list(range(1, 6)))) for _ in range(10)]
insert_categories_query = "INSERT INTO categories (name, parent_category_id) VALUES %s RETURNING id"
cur.execute("TRUNCATE TABLE categories RESTART IDENTITY CASCADE")
execute_values(cur, insert_categories_query, categories)
category_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'merchants' table
merchants = []
for _ in range(20):
    merchant = (
        random.choice(country_ids),
        random.choice(['active', 'inactive']),
        fake.company(),
        fake.address(),
        fake.url(),
        fake.phone_number(),
        fake.email(),
        fake.image_url(),
        fake.date_time_this_decade()
    )
    merchants.append(merchant)
insert_merchants_query = """
INSERT INTO merchants (country_code, status, merchant_name, address, website_url, phone_number, email, logo_url, created_at)
VALUES %s RETURNING id
"""
cur.execute("TRUNCATE TABLE merchants RESTART IDENTITY CASCADE")
execute_values(cur, insert_merchants_query, merchants)
merchant_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'shipping_carriers' table
shipping_carriers = [(fake.company(), fake.url()) for _ in range(5)]
insert_shipping_carriers_query = "INSERT INTO shipping_carriers (name, tracking_url) VALUES %s RETURNING id"
cur.execute("TRUNCATE TABLE shipping_carriers RESTART IDENTITY CASCADE")
execute_values(cur, insert_shipping_carriers_query, shipping_carriers)
shipping_carrier_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'users' table
users = []
for _ in range(50):
    user = (
        fake.name(),
        fake.email(),
        fake.user_name(),
        fake.phone_number(),
        fake.date_time_this_decade(),
        fake.image_url(),
        fake.date_time_this_decade(),
        random.choice(country_ids)
    )
    users.append(user)
insert_users_query = """
INSERT INTO users (full_name, email, username, phone_number, last_login_at, avatar_url, created_at, country_code)
VALUES %s RETURNING id
"""
cur.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
execute_values(cur, insert_users_query, users)
user_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'products' table
products = []
for _ in range(100):
    product = (
        fake.word(),
        fake.text(),
        fake.company(),
        fake.color_name(),
        round(random.uniform(0.1, 10.0), 2),
        fake.random_element(elements=('10x10x10 cm', '15x15x15 cm', '20x20x20 cm')),
        random.randint(1, 5),
        random.choice(merchant_ids),
        round(random.uniform(10, 1000), 2),
        fake.date_time_this_year(),
        random.choice(category_ids)
    )
    products.append(product)
insert_products_query = """
INSERT INTO products (name, description, brand, color, weight, dimensions, rating, merchant_id, price, created_at, category_id)
VALUES %s RETURNING id
"""
cur.execute("TRUNCATE TABLE products RESTART IDENTITY CASCADE")
execute_values(cur, insert_products_query, products)
product_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'orders' table
orders = []
for _ in range(100):
    order = (
        random.choice(user_ids),
        random.choice(['pending', 'completed', 'cancelled']),
        fake.date_time_this_year(),
        round(random.uniform(20, 2000), 2),
        fake.address(),
        fake.address(),
        fake.random_element(elements=('credit card', 'paypal', 'bank transfer')),
        random.choice(['paid', 'unpaid']),
        random.choice(shipping_carrier_ids)
    )
    orders.append(order)
insert_orders_query = """
INSERT INTO orders (user_id, status, created_at, total_sum, shipping_address, billing_address, payment_method, payment_status, shipping_carrier_id)
VALUES %s RETURNING id
"""
cur.execute("TRUNCATE TABLE orders RESTART IDENTITY CASCADE")
execute_values(cur, insert_orders_query, orders)
order_ids = [row[0] for row in cur.fetchall()]

# Insert random data into 'order_items' table
order_items = []
for _ in range(200):
    order_item = (
        random.choice(order_ids),
        random.choice(product_ids),
        random.randint(1, 10),
        round(random.uniform(10, 1000), 2),
        round(random.uniform(20, 2000), 2)
    )
    order_items.append(order_item)
insert_order_items_query = """
INSERT INTO order_items (order_id, product_id, quantity, price, sum)
VALUES %s
"""
cur.execute("TRUNCATE TABLE order_items RESTART IDENTITY CASCADE")
execute_values(cur, insert_order_items_query, order_items)
conn.commit()

# check the tables
cur.execute("""SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'""")

for table in cur.fetchall():
    print(table)

# test select from one table
cur.execute("""SELECT * FROM countries LIMIT 10""")

for table in cur.fetchall():
    print(table)

# Close the connection
cur.close()
conn.close()
