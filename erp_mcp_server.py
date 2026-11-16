"""erp_mcp_server.py — MCP server that exposes your ERP database to the LLM.

The LLM calls these tools at runtime to fetch live company data, making
responses grounded in real data instead of just training knowledge.

Usage:
    pip install mcp sqlalchemy
    python erp_mcp_server.py --db sqlite:///erp.db
    python erp_mcp_server.py --db postgresql://user:pass@localhost/erp
    python erp_mcp_server.py --db mysql+pymysql://user:pass@localhost/erp

Environment variable alternative:
    ERP_DB_URL=sqlite:///erp.db  python erp_mcp_server.py

Adapt the table/column names to match your ERP schema.
"""
import argparse
import json
import os
from datetime import date, datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mcp = FastMCP("ERP Assistant")

_engine = None


def get_engine():
    return _engine


def query(sql: str, params: dict = None) -> list[dict]:
    """Run a SQL query and return rows as list of dicts."""
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        cols = list(result.keys())
        rows = []
        for row in result:
            rows.append({col: _serialize(val) for col, val in zip(cols, row)})
        return rows


def _serialize(val: Any) -> Any:
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    return val


# ---------------------------------------------------------------------------
# PURCHASE ORDERS
# ---------------------------------------------------------------------------
@mcp.tool()
def get_purchase_orders(
    vendor_id: str = None,
    status: str = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 20,
) -> str:
    """Fetch purchase orders from the ERP.

    Args:
        vendor_id:  Filter by vendor ID (optional)
        status:     Filter by status: OPEN, RECEIVED, INVOICED, CLOSED (optional)
        date_from:  Filter orders from this date (YYYY-MM-DD, optional)
        date_to:    Filter orders up to this date (YYYY-MM-DD, optional)
        limit:      Max rows to return (default 20)
    """
    where, params = ["1=1"], {}
    if vendor_id:
        where.append("vendor_id = :vendor_id")
        params["vendor_id"] = vendor_id
    if status:
        where.append("status = :status")
        params["status"] = status.upper()
    if date_from:
        where.append("order_date >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where.append("order_date <= :date_to")
        params["date_to"] = date_to
    params["limit"] = limit

    sql = f"""
        SELECT po_number, vendor_id, vendor_name, order_date,
               total_amount, currency, status, delivery_date
        FROM purchase_orders
        WHERE {' AND '.join(where)}
        ORDER BY order_date DESC
        LIMIT :limit
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No purchase orders found."


# ---------------------------------------------------------------------------
# INVOICES (AP + AR)
# ---------------------------------------------------------------------------
@mcp.tool()
def get_invoices(
    type: str = "AP",
    party_id: str = None,
    overdue_only: bool = False,
    limit: int = 20,
) -> str:
    """Fetch invoices from the ERP.

    Args:
        type:        AP (vendor invoices) or AR (customer invoices)
        party_id:    Vendor ID (for AP) or Customer ID (for AR)
        overdue_only: Return only overdue unpaid invoices
        limit:       Max rows to return (default 20)
    """
    where = ["type = :type"]
    params: dict = {"type": type.upper(), "limit": limit}

    if party_id:
        where.append("party_id = :party_id")
        params["party_id"] = party_id
    if overdue_only:
        where.append("due_date < CURRENT_DATE AND status != 'PAID'")

    sql = f"""
        SELECT invoice_number, type, party_id, party_name,
               invoice_date, due_date, amount, currency,
               amount_paid, status
        FROM invoices
        WHERE {' AND '.join(where)}
        ORDER BY due_date ASC
        LIMIT :limit
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No invoices found."


# ---------------------------------------------------------------------------
# INVENTORY
# ---------------------------------------------------------------------------
@mcp.tool()
def get_inventory(
    material_code: str = None,
    warehouse: str = None,
    low_stock_only: bool = False,
    limit: int = 30,
) -> str:
    """Check inventory stock levels.

    Args:
        material_code:  Filter by specific material/product code (optional)
        warehouse:      Filter by warehouse location (optional)
        low_stock_only: Return only items at or below reorder point
        limit:          Max rows (default 30)
    """
    where, params = ["1=1"], {}
    if material_code:
        where.append("material_code = :material_code")
        params["material_code"] = material_code
    if warehouse:
        where.append("warehouse = :warehouse")
        params["warehouse"] = warehouse
    if low_stock_only:
        where.append("qty_on_hand <= reorder_point")
    params["limit"] = limit

    sql = f"""
        SELECT material_code, description, warehouse,
               qty_on_hand, unit_of_measure, reorder_point,
               unit_cost, total_value, last_updated
        FROM inventory
        WHERE {' AND '.join(where)}
        ORDER BY qty_on_hand ASC
        LIMIT :limit
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No inventory records found."


# ---------------------------------------------------------------------------
# CUSTOMERS
# ---------------------------------------------------------------------------
@mcp.tool()
def get_customer(customer_id: str = None, name: str = None) -> str:
    """Look up customer master data.

    Args:
        customer_id: Exact customer ID (optional)
        name:        Partial customer name search (optional)
    """
    where, params = ["1=1"], {}
    if customer_id:
        where.append("customer_id = :customer_id")
        params["customer_id"] = customer_id
    if name:
        where.append("name ILIKE :name")
        params["name"] = f"%{name}%"

    sql = f"""
        SELECT customer_id, name, email, phone,
               address, credit_limit, payment_terms,
               account_balance, status
        FROM customers
        WHERE {' AND '.join(where)}
        LIMIT 10
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No customer found."


# ---------------------------------------------------------------------------
# VENDORS / SUPPLIERS
# ---------------------------------------------------------------------------
@mcp.tool()
def get_vendor(vendor_id: str = None, name: str = None) -> str:
    """Look up vendor/supplier master data.

    Args:
        vendor_id: Exact vendor ID (optional)
        name:      Partial vendor name search (optional)
    """
    where, params = ["1=1"], {}
    if vendor_id:
        where.append("vendor_id = :vendor_id")
        params["vendor_id"] = vendor_id
    if name:
        where.append("name ILIKE :name")
        params["name"] = f"%{name}%"

    sql = f"""
        SELECT vendor_id, name, email, phone,
               address, payment_terms, bank_account,
               tax_id, status
        FROM vendors
        WHERE {' AND '.join(where)}
        LIMIT 10
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No vendor found."


# ---------------------------------------------------------------------------
# GENERAL LEDGER
# ---------------------------------------------------------------------------
@mcp.tool()
def get_gl_balance(
    account_code: str = None,
    fiscal_year: int = None,
    period: int = None,
) -> str:
    """Get General Ledger account balances.

    Args:
        account_code: GL account number (optional, returns all if omitted)
        fiscal_year:  Year (optional, defaults to current year)
        period:       Accounting period 1-12 (optional)
    """
    where, params = ["1=1"], {}
    if account_code:
        where.append("account_code = :account_code")
        params["account_code"] = account_code
    if fiscal_year:
        where.append("fiscal_year = :fiscal_year")
        params["fiscal_year"] = fiscal_year
    if period:
        where.append("period = :period")
        params["period"] = period

    sql = f"""
        SELECT account_code, account_name, account_type,
               fiscal_year, period, debit_amount, credit_amount,
               (debit_amount - credit_amount) AS balance, currency
        FROM gl_balances
        WHERE {' AND '.join(where)}
        ORDER BY account_code
        LIMIT 50
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No GL balance records found."


# ---------------------------------------------------------------------------
# COST CENTERS
# ---------------------------------------------------------------------------
@mcp.tool()
def get_cost_center_spend(
    cost_center: str = None,
    fiscal_year: int = None,
    period: int = None,
) -> str:
    """Get actual vs budget spend by cost center.

    Args:
        cost_center: Cost center code (optional)
        fiscal_year: Year (optional)
        period:      Period 1-12 (optional)
    """
    where, params = ["1=1"], {}
    if cost_center:
        where.append("cost_center = :cost_center")
        params["cost_center"] = cost_center
    if fiscal_year:
        where.append("fiscal_year = :fiscal_year")
        params["fiscal_year"] = fiscal_year
    if period:
        where.append("period = :period")
        params["period"] = period

    sql = f"""
        SELECT cost_center, cost_center_name, fiscal_year, period,
               actual_amount, budget_amount,
               (actual_amount - budget_amount) AS variance,
               ROUND(100.0 * actual_amount / NULLIF(budget_amount, 0), 2) AS pct_used,
               currency
        FROM cost_center_actuals
        WHERE {' AND '.join(where)}
        ORDER BY cost_center, period
        LIMIT 50
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No cost center data found."


# ---------------------------------------------------------------------------
# SALES ORDERS
# ---------------------------------------------------------------------------
@mcp.tool()
def get_sales_orders(
    customer_id: str = None,
    status: str = None,
    date_from: str = None,
    limit: int = 20,
) -> str:
    """Fetch sales orders from the ERP.

    Args:
        customer_id: Filter by customer ID (optional)
        status:      OPEN, SHIPPED, INVOICED, CLOSED (optional)
        date_from:   Filter from date (YYYY-MM-DD, optional)
        limit:       Max rows (default 20)
    """
    where, params = ["1=1"], {}
    if customer_id:
        where.append("customer_id = :customer_id")
        params["customer_id"] = customer_id
    if status:
        where.append("status = :status")
        params["status"] = status.upper()
    if date_from:
        where.append("order_date >= :date_from")
        params["date_from"] = date_from
    params["limit"] = limit

    sql = f"""
        SELECT so_number, customer_id, customer_name, order_date,
               total_amount, currency, status, delivery_date
        FROM sales_orders
        WHERE {' AND '.join(where)}
        ORDER BY order_date DESC
        LIMIT :limit
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No sales orders found."


# ---------------------------------------------------------------------------
# EMPLOYEES / HR
# ---------------------------------------------------------------------------
@mcp.tool()
def get_employee(employee_id: str = None, name: str = None, department: str = None) -> str:
    """Look up employee information.

    Args:
        employee_id: Exact employee ID (optional)
        name:        Partial name search (optional)
        department:  Filter by department (optional)
    """
    where, params = ["1=1"], {}
    if employee_id:
        where.append("employee_id = :employee_id")
        params["employee_id"] = employee_id
    if name:
        where.append("full_name ILIKE :name")
        params["name"] = f"%{name}%"
    if department:
        where.append("department = :department")
        params["department"] = department

    sql = f"""
        SELECT employee_id, full_name, department, job_title,
               manager_id, hire_date, employment_status, work_email
        FROM employees
        WHERE {' AND '.join(where)}
        LIMIT 10
    """
    rows = query(sql, params)
    return json.dumps(rows, indent=2) if rows else "No employee found."


# ---------------------------------------------------------------------------
# FINANCIAL SUMMARY (dashboard-level)
# ---------------------------------------------------------------------------
@mcp.tool()
def get_financial_summary(fiscal_year: int = None, period: int = None) -> str:
    """Get a high-level financial summary (revenue, expenses, profit, AR, AP).

    Args:
        fiscal_year: Year (optional, defaults to current)
        period:      Period 1-12 (optional)
    """
    params: dict = {}
    year_filter = "AND fiscal_year = :fiscal_year" if fiscal_year else ""
    period_filter = "AND period = :period" if period else ""
    if fiscal_year:
        params["fiscal_year"] = fiscal_year
    if period:
        params["period"] = period

    sql = f"""
        SELECT
            SUM(CASE WHEN account_type = 'REVENUE'  THEN credit_amount - debit_amount ELSE 0 END) AS total_revenue,
            SUM(CASE WHEN account_type = 'EXPENSE'  THEN debit_amount - credit_amount ELSE 0 END) AS total_expenses,
            SUM(CASE WHEN account_type = 'REVENUE'  THEN credit_amount - debit_amount ELSE 0 END) -
            SUM(CASE WHEN account_type = 'EXPENSE'  THEN debit_amount - credit_amount ELSE 0 END) AS net_profit,
            MAX(currency) AS currency
        FROM gl_balances
        WHERE 1=1 {year_filter} {period_filter}
    """
    rows = query(sql, params)

    ar_sql = f"SELECT SUM(amount - amount_paid) AS open_ar FROM invoices WHERE type='AR' AND status != 'PAID'"
    ap_sql = f"SELECT SUM(amount - amount_paid) AS open_ap FROM invoices WHERE type='AP' AND status != 'PAID'"

    result = rows[0] if rows else {}
    try:
        result["open_ar"] = query(ar_sql)[0].get("open_ar", 0)
        result["open_ap"] = query(ap_sql)[0].get("open_ap", 0)
    except Exception:
        pass

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global _engine

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        default=os.environ.get("ERP_DB_URL", "sqlite:///erp.db"),
        help="SQLAlchemy database URL (default: sqlite:///erp.db)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse"],
        help="MCP transport: stdio (default) or sse (HTTP)",
    )
    args = parser.parse_args()

    print(f"Connecting to ERP database: {args.db}")
    _engine = create_engine(args.db, pool_pre_ping=True)

    print("Starting ERP MCP server...")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
