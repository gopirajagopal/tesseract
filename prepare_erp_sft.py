"""prepare_erp_sft.py — Build ERP-specific SFT (instruction fine-tuning) dataset.

Creates a rich Q&A dataset covering:
  - SAP modules (FI, CO, MM, SD, PP, HR, PM, QM, WM)
  - Oracle ERP Cloud / Oracle E-Business Suite
  - Microsoft Dynamics 365
  - Odoo (open source ERP)
  - General ERP concepts, business processes, transactions
  - Business process cycles: P2P, O2C, R2R, H2R, MtM

Output:
  data/erp_sft_train.jsonl
  data/erp_sft_val.jsonl

Usage:
    python prepare_erp_sft.py
"""
import json
import random
from pathlib import Path

OUT_TRAIN = Path("data/erp_sft_train.jsonl")
OUT_VAL   = Path("data/erp_sft_val.jsonl")

# ---------------------------------------------------------------------------
# Core ERP knowledge Q&A pairs
# ---------------------------------------------------------------------------
ERP_QA = [
    # ---- General ERP Concepts ----
    {
        "instruction": "What is an ERP system?",
        "response": "An ERP (Enterprise Resource Planning) system is integrated business management software that centralizes and automates core business processes across an organization — including finance, procurement, inventory, manufacturing, HR, and sales — in a single database. Examples include SAP S/4HANA, Oracle ERP Cloud, Microsoft Dynamics 365, and Odoo."
    },
    {
        "instruction": "What are the core modules of an ERP system?",
        "response": "Core ERP modules typically include:\n1. Financial Management (GL, AP, AR, Fixed Assets)\n2. Procurement (Purchase Orders, Vendor Management)\n3. Inventory & Warehouse Management\n4. Sales & Distribution (Order Management, Billing)\n5. Manufacturing / Production Planning\n6. Human Resources & Payroll\n7. Project Management\n8. Reporting & Analytics (Business Intelligence)"
    },
    {
        "instruction": "Explain the Procure-to-Pay (P2P) cycle.",
        "response": "The Procure-to-Pay (P2P) cycle covers all steps from identifying a purchasing need to paying the vendor:\n1. Purchase Requisition (PR) — internal request for goods/services\n2. Request for Quotation (RFQ) — sent to vendors\n3. Purchase Order (PO) — formal order issued to selected vendor\n4. Goods Receipt (GR) — physical receipt and inspection of goods\n5. Invoice Verification — three-way match (PO, GR, Invoice)\n6. Payment — vendor payment processed via AP\n7. Vendor Account Update — ledger entries posted"
    },
    {
        "instruction": "Explain the Order-to-Cash (O2C) cycle.",
        "response": "The Order-to-Cash (O2C) cycle covers all steps from a customer order to receiving payment:\n1. Sales Inquiry / Quotation — customer requests pricing\n2. Sales Order (SO) — confirmed order entered into the system\n3. Availability Check — inventory confirmed\n4. Delivery / Goods Issue — products shipped to customer\n5. Billing / Invoice — customer invoice generated\n6. Payment Receipt — customer payment collected\n7. Account Reconciliation — AR cleared and ledger updated"
    },
    {
        "instruction": "What is the Record-to-Report (R2R) process?",
        "response": "Record-to-Report (R2R) is the finance and accounting process that covers:\n1. Data Collection — capturing all financial transactions\n2. Journal Entry — recording transactions in the general ledger\n3. Reconciliation — matching sub-ledgers to GL\n4. Period-End Close — closing books at end of month/quarter/year\n5. Financial Statements — generating P&L, Balance Sheet, Cash Flow\n6. Reporting & Analysis — management and regulatory reporting\n7. Audit Support — providing documentation for auditors"
    },
    {
        "instruction": "What is a three-way match in procurement?",
        "response": "A three-way match is a validation process that compares three documents before approving vendor payment:\n1. Purchase Order (PO) — what was ordered and at what price\n2. Goods Receipt (GR) — what was actually received\n3. Vendor Invoice — what the vendor is charging\nAll three must match within tolerance. Discrepancies trigger exceptions requiring manual review. This prevents overpayment, fraud, and inventory errors."
    },
    {
        "instruction": "What is a Chart of Accounts?",
        "response": "A Chart of Accounts (CoA) is the complete listing of all general ledger accounts used by an organization, organized by account type:\n- Asset accounts (1000-1999)\n- Liability accounts (2000-2999)\n- Equity accounts (3000-3999)\n- Revenue accounts (4000-4999)\n- Expense accounts (5000-5999)\nEach account has a unique code, description, and account type. The CoA is the backbone of all financial reporting in an ERP system."
    },
    {
        "instruction": "What is the difference between AP and AR in ERP?",
        "response": "Accounts Payable (AP) and Accounts Receivable (AR) are mirror processes:\n\nAP (Accounts Payable):\n- Money the company OWES to vendors/suppliers\n- Processes: vendor invoices, payment runs, vendor master\n- Impacts: cash outflow, cost recognition\n\nAR (Accounts Receivable):\n- Money OWED to the company by customers\n- Processes: customer invoices, payment collection, dunning\n- Impacts: cash inflow, revenue recognition\n\nBoth feed into the General Ledger for financial reporting."
    },
    # ---- SAP Specific ----
    {
        "instruction": "What is SAP S/4HANA?",
        "response": "SAP S/4HANA is SAP's flagship next-generation ERP suite built on the SAP HANA in-memory database. Key features:\n- Real-time analytics embedded in transactional processes\n- Simplified data model (ACDOCA universal journal replaces multiple tables)\n- Fiori UX (role-based, responsive web interface)\n- Available on-premise, cloud (public/private), and hybrid\n- Successor to SAP ECC (ERP Central Component)\nCore modules: FI (Finance), CO (Controlling), MM (Materials Management), SD (Sales & Distribution), PP (Production Planning), HCM (Human Capital Management)."
    },
    {
        "instruction": "What T-code is used to create a Purchase Order in SAP?",
        "response": "ME21N is the transaction code (T-code) to create a Purchase Order in SAP MM (Materials Management). Related PO T-codes:\n- ME21N — Create PO\n- ME22N — Change PO\n- ME23N — Display PO\n- ME2M — PO list by material\n- ME2L — PO list by vendor\n- MIGO — Goods Receipt against PO\n- MIRO — Invoice Verification (Logistics Invoice Verification)"
    },
    {
        "instruction": "What are common SAP FI transaction codes?",
        "response": "Common SAP Financial Accounting (FI) T-codes:\n\nGeneral Ledger:\n- FB50 — Post GL document\n- FS10N — GL account balance\n- S_ALR_87012277 — GL account list\n\nAccounts Payable:\n- FB60 — Enter vendor invoice\n- F110 — Automatic Payment Run\n- FK01/FK02/FK03 — Create/Change/Display Vendor\n\nAccounts Receivable:\n- FB70 — Enter customer invoice\n- F-28 — Post customer payment\n- FD01/FD02/FD03 — Create/Change/Display Customer\n\nPeriod Close:\n- F.01 — Financial Statements\n- F-03 — Clear GL Account"
    },
    {
        "instruction": "What is a Material Master in SAP?",
        "response": "A Material Master in SAP is the central repository for all data related to a material (product, raw material, or service). It is organized into views by department:\n- Basic Data — description, unit of measure, weight\n- Purchasing — purchasing group, order unit, GR processing time\n- MRP — planning strategy, lot size, safety stock\n- Sales — sales org, distribution channel, pricing\n- Accounting — valuation class, standard price, moving average price\n- Storage — storage conditions, shelf life\nT-codes: MM01 (Create), MM02 (Change), MM03 (Display)."
    },
    {
        "instruction": "Explain SAP's organizational structure.",
        "response": "SAP organizational structure (hierarchy from top to bottom):\n1. Client — highest level, represents entire company group\n2. Company Code — legal entity (maps to a balance sheet)\n3. Plant — production/storage facility within a company code\n4. Storage Location — physical storage area within a plant\n5. Sales Organization — responsible for distributing goods\n6. Distribution Channel — how goods reach customers (retail, wholesale)\n7. Division — product line or service group\n8. Purchasing Organization — responsible for procurement activities\n9. Purchasing Group — buyer group within purchasing org"
    },
    {
        "instruction": "What is MIGO in SAP?",
        "response": "MIGO (Movement of Goods) is the central SAP transaction for all goods movements in Materials Management. Common movement types:\n- 101 — Goods Receipt against Purchase Order\n- 102 — Reversal of GR against PO\n- 201 — Goods Issue to Cost Center\n- 261 — Goods Issue to Production Order\n- 301 — Transfer Posting (plant to plant)\n- 311 — Transfer Posting (storage location to storage location)\n- 501 — Receipt without Purchase Order\nPath: Logistics → Materials Management → Inventory Management → Goods Movement → MIGO"
    },
    {
        "instruction": "What is SAP FICO?",
        "response": "SAP FICO refers to the combined Financial Accounting (FI) and Controlling (CO) modules in SAP ERP:\n\nFI (Financial Accounting) — external reporting:\n- General Ledger (GL)\n- Accounts Payable (AP)\n- Accounts Receivable (AR)\n- Asset Accounting (AA)\n- Bank Accounting\n\nCO (Controlling) — internal management accounting:\n- Cost Center Accounting\n- Profit Center Accounting\n- Internal Orders\n- Product Costing\n- Profitability Analysis (CO-PA)\n\nFI and CO are tightly integrated — cost postings in FI automatically update CO objects."
    },
    # ---- Oracle ERP ----
    {
        "instruction": "What is Oracle ERP Cloud?",
        "response": "Oracle ERP Cloud (Oracle Fusion Cloud ERP) is Oracle's cloud-based enterprise resource planning suite. Key modules:\n- Financial Management — GL, AP, AR, Fixed Assets, Cash Management\n- Procurement — Purchasing, Sourcing, Supplier Management\n- Project Management — Project Costing, Billing, Planning\n- Risk Management — Audit, Controls, Compliance\n- Supply Chain — Inventory, Order Management, Manufacturing\nKey advantages: real-time reporting with OTBI/BI Publisher, embedded AI/ML, frequent quarterly updates, mobile-first design."
    },
    {
        "instruction": "What is a Subledger in Oracle ERP?",
        "response": "A subledger in Oracle ERP is a detailed subsidiary ledger that records transactions for a specific area and summarizes them into the General Ledger:\n- AP Subledger — vendor invoices and payments\n- AR Subledger — customer invoices and receipts\n- Fixed Assets Subledger — asset acquisitions, depreciation, retirements\n- Inventory Subledger — inventory transactions and valuation\n- Project Subledger — project costs and revenue\nSubledger Accounting (SLA) in Oracle controls how subledger transactions are converted to journal entries in the GL using accounting rules."
    },
    {
        "instruction": "What are Oracle ERP's main financial reports?",
        "response": "Key financial reports in Oracle ERP Cloud:\n1. Trial Balance — lists all GL account balances\n2. Income Statement (P&L) — revenue vs expenses\n3. Balance Sheet — assets, liabilities, equity\n4. Cash Flow Statement — operating/investing/financing activities\n5. Aged Payables — outstanding vendor invoices by age bucket\n6. Aged Receivables — outstanding customer invoices by age bucket\n7. Budget vs Actual — variance analysis\n8. Cost Center Report — expenses by cost center\nReporting tools: OTBI (Oracle Transactional Business Intelligence), Financial Reporting Studio, SmartView for Excel."
    },
    # ---- Microsoft Dynamics 365 ----
    {
        "instruction": "What is Microsoft Dynamics 365 Finance?",
        "response": "Microsoft Dynamics 365 Finance is Microsoft's cloud ERP solution for financial management. Key capabilities:\n- General Ledger with multi-currency and multi-company support\n- Accounts Payable & Receivable automation\n- Budgeting & Forecasting\n- Fixed Asset management\n- Cash & Bank Management\n- Tax Management (multi-jurisdiction)\n- Financial Reporting (Management Reporter)\n- AI-powered insights and Cash Flow forecasting\nIntegrates natively with Dynamics 365 Supply Chain Management, Power BI, and Microsoft 365."
    },
    {
        "instruction": "What is a legal entity in Microsoft Dynamics 365?",
        "response": "In Microsoft Dynamics 365, a legal entity is an organization with a registered legal structure that can conduct business independently. It is the fundamental organizational unit:\n- Has its own chart of accounts, fiscal calendar, and currency\n- Maintains its own financial statements (balance sheet, P&L)\n- Maps to a company code in SAP terminology\n- Can be a corporation, LLC, partnership, or other legal structure\nMultiple legal entities can be grouped into a consolidation hierarchy for group reporting. Inter-company transactions are managed between legal entities."
    },
    # ---- Odoo ----
    {
        "instruction": "What is Odoo ERP?",
        "response": "Odoo is an open-source ERP platform with both community (free) and enterprise (paid) editions. Key features:\n- Modular architecture — install only needed modules\n- Main modules: Accounting, Sales, Purchase, Inventory, Manufacturing, HR, CRM, Project, E-commerce\n- Web-based, mobile-friendly interface\n- Strong integration between modules\n- Large community and marketplace for extensions\n- Available on-premise or as Odoo Online (SaaS)\nOdoo uses Python backend (Odoo/OpenERP framework) and PostgreSQL database. Customization via Python and XML."
    },
    {
        "instruction": "How does inventory valuation work in ERP systems?",
        "response": "ERP systems support multiple inventory valuation methods:\n\n1. FIFO (First In, First Out) — oldest inventory costs used first\n2. LIFO (Last In, First Out) — newest costs used first (not allowed under IFRS)\n3. Weighted Average Cost (WAC) — average cost of all units\n4. Standard Cost — predetermined cost set in advance; variances tracked\n5. Actual Cost / Specific Identification — each item tracked individually\n\nThe chosen method impacts:\n- Cost of Goods Sold (COGS)\n- Inventory asset value on Balance Sheet\n- Tax liability\n- Profit margins reported\n\nSAP uses valuation class + price control (V=moving average, S=standard price)."
    },
    # ---- Business Process ----
    {
        "instruction": "What is a cost center in ERP?",
        "response": "A cost center is an organizational unit that incurs costs but does not directly generate revenue. It is used for internal management accounting:\n- Examples: IT department, HR department, Facilities, Legal\n- Costs are posted to cost centers for budget control and reporting\n- Enables tracking of spending by department or function\n- Allows comparison of actual vs planned costs\n- Cost center managers are accountable for their budget\nIn SAP: managed in CO module, T-code KS01/KS02/KS03. In Oracle: mapped to cost center segment of the GL account structure."
    },
    {
        "instruction": "What is a profit center in ERP?",
        "response": "A profit center is an organizational unit that is held responsible for both revenues and costs, allowing profitability analysis:\n- Examples: product line, business unit, geographic region, store\n- Generates its own P&L statement\n- Management is accountable for both revenue and cost control\n- Enables profitability analysis by segment\nDifference from cost center: cost centers only track costs; profit centers track both revenue and costs.\nIn SAP: managed in EC-PCA (Profit Center Accounting), T-code KE51/KE52/KE53. In Oracle: mapped to business unit or segment."
    },
    {
        "instruction": "What is depreciation in ERP and how is it calculated?",
        "response": "Depreciation is the systematic allocation of a fixed asset's cost over its useful life. ERP systems support multiple depreciation methods:\n\n1. Straight-Line (SLN) — equal amount each period\n   Formula: (Cost - Salvage Value) / Useful Life\n\n2. Declining Balance — fixed percentage of book value\n   Formula: Book Value × Depreciation Rate\n\n3. Double Declining Balance — accelerated method\n   Formula: (2 / Useful Life) × Book Value\n\n4. Units of Production — based on actual usage\n   Formula: (Cost - Salvage) × (Units Used / Total Units)\n\n5. Sum of Years' Digits — accelerated method\nERP systems automatically post depreciation journals at period end to Fixed Asset and Depreciation Expense accounts."
    },
    {
        "instruction": "What is a purchase requisition vs purchase order?",
        "response": "Purchase Requisition (PR) and Purchase Order (PO) are sequential steps in the procurement process:\n\nPurchase Requisition (PR):\n- Internal document requesting goods/services\n- Created by the requesting department\n- Requires internal approval workflow\n- Does NOT create a legal obligation\n- SAP T-code: ME51N (create), ME52N (change)\n\nPurchase Order (PO):\n- External document sent to the vendor\n- Created by the Purchasing department\n- Legally binding contract with the vendor\n- Specifies price, quantity, delivery date\n- Can be created from approved PR (ME21N)\n- SAP T-code: ME21N (create), ME22N (change)"
    },
    {
        "instruction": "How does the payment run work in SAP F110?",
        "response": "F110 is the Automatic Payment Program in SAP that processes mass vendor payments:\n\nSteps to run F110:\n1. Enter Run Date and Identification (unique key)\n2. Parameters tab — enter company code, payment method, bank accounts, posting date, next payment date\n3. Proposal — system selects open invoices due for payment\n4. Review Proposal — check selected items, exceptions\n5. Edit Proposal — manually include/exclude items if needed\n6. Payment Run — system creates payment documents and clears open items\n7. Printout — generate payment media (checks, bank transfer files, SEPA)\n8. Post to Bank — payment file sent to bank\n\nPayment methods: Check (C), Wire Transfer (T), ACH, SEPA Credit Transfer."
    },
    {
        "instruction": "What is goods receipt in ERP?",
        "response": "A Goods Receipt (GR) is the formal recording in the ERP system of physical goods received from a vendor. Key aspects:\n\nProcess:\n1. Goods arrive at warehouse\n2. Warehouse staff verifies quantity and quality against PO\n3. GR is posted in ERP system\n4. Inventory stock increases\n5. GR/IR (Goods Receipt/Invoice Receipt) clearing account is credited\n\nAccounting entries at GR:\n- Debit: Inventory Account\n- Credit: GR/IR Clearing Account\n\nAt Invoice Verification (MIRO):\n- Debit: GR/IR Clearing Account\n- Credit: Vendor AP Account\n\nIn SAP: MIGO transaction, movement type 101."
    },
    {
        "instruction": "What is intercompany accounting in ERP?",
        "response": "Intercompany accounting manages financial transactions between different legal entities within the same corporate group:\n\nCommon intercompany transactions:\n- Intercompany sales (goods sold between subsidiaries)\n- Intercompany loans and interest\n- Shared services charges (IT, HR, Legal)\n- Management fee allocations\n\nProcess:\n1. Transaction posted in entity A\n2. Corresponding entry automatically created in entity B\n3. Both sides use intercompany GL accounts\n4. At consolidation, intercompany balances are eliminated\n\nIn SAP: configured via trading partner assignment and intercompany billing.\nKey requirement: intercompany receivable in entity A must equal intercompany payable in entity B (reconciliation)."
    },
    {
        "instruction": "What is master data in ERP?",
        "response": "Master data in ERP is the core reference data that is shared across the organization and used in transactions. It changes infrequently and must be managed carefully:\n\nTypes of master data:\n- Customer Master — name, address, payment terms, credit limit, account assignment\n- Vendor/Supplier Master — contact info, payment terms, bank details, tax info\n- Material Master — description, UoM, price, procurement data, MRP settings\n- GL Account Master — account number, description, account type, tax category\n- Employee Master — personal data, organizational assignment, payroll data\n- Asset Master — asset description, class, useful life, depreciation area\n\nMaster data governance is critical — poor quality master data leads to incorrect transactions, reporting errors, and compliance issues."
    },
    {
        "instruction": "What is a dunning process in ERP?",
        "response": "Dunning is the systematic process of sending payment reminders to customers with overdue invoices:\n\nDunning levels (typically 4 levels):\n1. Level 1 — polite reminder (e.g., 10 days overdue)\n2. Level 2 — second reminder with slight urgency (e.g., 20 days)\n3. Level 3 — formal demand letter (e.g., 30 days)\n4. Level 4 — final notice / collections referral (e.g., 45+ days)\n\nEach level can have different:\n- Letter text and tone\n- Interest charges applied\n- Minimum amount threshold\n- Responsible contact\n\nIn SAP: T-code F150 (Dunning Run).\nDunning history is tracked in the customer account to avoid sending lower-level notices after a higher level was already sent."
    },
    {
        "instruction": "What is the difference between ERP and CRM?",
        "response": "ERP (Enterprise Resource Planning) and CRM (Customer Relationship Management) serve different but complementary purposes:\n\nERP:\n- Focus: internal business operations\n- Core processes: finance, procurement, inventory, manufacturing, HR\n- Primary users: finance, operations, warehouse, HR teams\n- Drives: operational efficiency and financial reporting\n- Examples: SAP, Oracle ERP, Dynamics 365 Finance\n\nCRM:\n- Focus: customer-facing activities\n- Core processes: sales pipeline, marketing campaigns, customer service\n- Primary users: sales, marketing, customer support teams\n- Drives: revenue growth and customer retention\n- Examples: Salesforce, Microsoft Dynamics 365 Sales, HubSpot\n\nModern ERPs (SAP, Oracle, Dynamics 365) include CRM modules and integrate both seamlessly."
    },
    {
        "instruction": "What is an ERP implementation methodology?",
        "response": "ERP implementation methodologies provide a structured approach to deploying ERP systems:\n\nSAP Activate (for SAP S/4HANA):\n1. Discover — business case, system demo\n2. Prepare — project setup, learning\n3. Explore — fit-gap analysis, design\n4. Realize — configuration, development, testing\n5. Deploy — cutover, go-live\n6. Run — hypercare, operations\n\nOracle Cloud Implementation:\n1. Inception — scope and planning\n2. Elaboration — process design, configuration\n3. Construction — build and test\n4. Transition — UAT, training, go-live\n5. Production — post-go-live support\n\nKey success factors: executive sponsorship, clear scope, data migration quality, change management, and user training."
    },
    {
        "instruction": "What is a journal entry in ERP?",
        "response": "A journal entry is the fundamental accounting record in ERP that captures a financial transaction using double-entry bookkeeping:\n\nComponents:\n- Date — transaction date\n- Description / Reference — explanation of the entry\n- Debit lines — accounts to be debited\n- Credit lines — accounts to be credited\n- Total debits must equal total credits\n\nExample — Recording vendor invoice:\nDebit: Office Supplies Expense  $500\nCredit: Accounts Payable        $500\n\nTypes of journal entries:\n- Manual journals — entered by accountants\n- Automatic journals — system-generated (GR, invoice, payment)\n- Recurring journals — repeating entries (rent, depreciation)\n- Accrual journals — period-end estimates\n- Reversing journals — auto-reverse next period\n\nIn SAP: FB50 (GL journal), in Oracle: Create Journal page."
    },
]

# ---------------------------------------------------------------------------
# Additional synthetic ERP scenarios
# ---------------------------------------------------------------------------
ADDITIONAL_QA = [
    {
        "instruction": "How do you perform a bank reconciliation in ERP?",
        "response": "Bank reconciliation in ERP matches the company's bank account records with the bank statement:\n\nProcess:\n1. Import bank statement (manual upload or automated via bank connectivity)\n2. System auto-matches transactions based on amount, date, reference\n3. Review unmatched items\n4. Post adjustments for bank charges, interest, errors\n5. Confirm all items are cleared\n6. Generate reconciliation report\n\nCommon reconciling items:\n- Outstanding checks (issued but not yet cleared)\n- Deposits in transit (posted in system, not yet in bank)\n- Bank fees not yet recorded\n- NSF (non-sufficient funds) returned checks\n\nIn SAP: T-code FF67 (Bank Statement Processing).\nIn Oracle: Cash Management > Bank Statements and Reconciliation."
    },
    {
        "instruction": "What is an MRP run in SAP?",
        "response": "MRP (Material Requirements Planning) in SAP calculates what materials are needed, in what quantities, and when. The MRP run (T-code MD01/MD01N) processes:\n\nInputs:\n- Independent demand (sales orders, forecasts)\n- Dependent demand (bills of material)\n- Stock on hand\n- Open purchase orders and production orders\n- MRP parameters (lot size, safety stock, lead time)\n\nOutputs:\n- Planned orders (for production)\n- Purchase requisitions (for procurement)\n- Schedule lines (for subcontracting)\n\nMRP types:\n- MRP (consumption-based)\n- MPS (Master Production Scheduling) for critical items\n- MRPII (manufacturing resource planning)\n\nAfter MRP run, planners review the 'MRP List' (MD05) and convert planned orders."
    },
    {
        "instruction": "What are the key KPIs in an ERP system for finance?",
        "response": "Key financial KPIs tracked in ERP systems:\n\nLiquidity:\n- Current Ratio = Current Assets / Current Liabilities\n- Quick Ratio = (Cash + AR) / Current Liabilities\n- Days Sales Outstanding (DSO) = (AR / Revenue) × Days\n- Days Payable Outstanding (DPO) = (AP / COGS) × Days\n\nProfitability:\n- Gross Profit Margin = (Revenue - COGS) / Revenue\n- EBITDA Margin = EBITDA / Revenue\n- Return on Assets (ROA) = Net Income / Total Assets\n\nOperational:\n- Inventory Turnover = COGS / Average Inventory\n- Days Inventory Outstanding (DIO)\n- Purchase Order Cycle Time\n- Invoice Processing Time\n- Budget Variance = (Actual - Budget) / Budget"
    },
    {
        "instruction": "Explain the concept of period-end close in ERP.",
        "response": "Period-end close (month-end / quarter-end / year-end close) is the process of finalizing financial records for a period:\n\nTypical close checklist:\n1. Post all transactions — ensure all invoices, payments, expenses are entered\n2. Reconcile subledgers — verify AP, AR, Fixed Assets match GL\n3. Accrue expenses — record expenses incurred but not yet invoiced\n4. Defer revenue — move advance payments to deferred revenue\n5. Post depreciation — run fixed asset depreciation\n6. Reconcile bank accounts — complete bank reconciliation\n7. Allocate costs — distribute shared costs to cost centers\n8. Run foreign currency revaluation — update FX balances\n9. Intercompany elimination — reconcile IC balances\n10. Generate financial statements — P&L, Balance Sheet, Cash Flow\n11. Lock period — prevent further posting to closed period\n12. Variance analysis — investigate significant variances\n\nERP systems have period close checklists and task management tools to track progress."
    },
]

# ---------------------------------------------------------------------------
# Expanded ERP Q&A — accounting, modules, processes, concepts
# ---------------------------------------------------------------------------
EXPANDED_ERP_QA = [
    # --- Accounting fundamentals ---
    {
        "instruction": "What is double-entry bookkeeping?",
        "response": "Double-entry bookkeeping is the accounting system where every transaction affects at least two accounts: one debit and one credit, and the total debits always equal total credits.\n\nExample — cash sale of $1,000:\nDebit: Cash (Asset) $1,000\nCredit: Revenue $1,000\n\nThe accounting equation is always maintained: Assets = Liabilities + Equity.\n\nERP systems enforce double-entry automatically — every posting creates balanced journal entries. You cannot post an unbalanced document."
    },
    {
        "instruction": "What is the difference between debit and credit in accounting?",
        "response": "In accounting, debit (Dr) and credit (Cr) describe the direction of an entry in an account:\n\nDebits INCREASE: Assets, Expenses, Dividends\nDebits DECREASE: Liabilities, Equity, Revenue\n\nCredits INCREASE: Liabilities, Equity, Revenue\nCredits DECREASE: Assets, Expenses, Dividends\n\nMemory aid — DEAD CLIC:\n- Debit: Expenses, Assets, Dividends (increase)\n- Credit: Liabilities, Income, Capital (increase)\n\nExample:\n- Buy inventory for cash: Debit Inventory (asset up), Credit Cash (asset down)\n- Earn revenue: Debit Cash (asset up), Credit Revenue (income up)"
    },
    {
        "instruction": "What is a balance sheet?",
        "response": "A balance sheet (Statement of Financial Position) is a financial statement showing a company's assets, liabilities, and equity at a specific point in time.\n\nStructure:\nASSETS (what the company owns):\n- Current Assets: Cash, Accounts Receivable, Inventory, Prepaid Expenses\n- Non-Current Assets: Property/Plant/Equipment, Intangibles, Investments\n\nLIABILITIES (what the company owes):\n- Current Liabilities: Accounts Payable, Short-term loans, Accrued expenses\n- Non-Current Liabilities: Long-term debt, Deferred tax\n\nEQUITY (owners' stake):\n- Share Capital, Retained Earnings, Other Comprehensive Income\n\nFormula: Assets = Liabilities + Equity\n\nIn ERP, the balance sheet is generated from the General Ledger account balances."
    },
    {
        "instruction": "What is the income statement (P&L)?",
        "response": "The income statement (Profit and Loss / P&L statement) shows revenue, expenses, and profit over a period.\n\nStructure:\nRevenue\n- Cost of Goods Sold (COGS)\n= Gross Profit\n- Operating Expenses (SG&A, R&D, Depreciation)\n= Operating Income (EBIT)\n- Interest Expense\n+ Interest Income\n= Income Before Tax\n- Income Tax Expense\n= Net Income\n\nKey metrics:\n- Gross Margin = Gross Profit / Revenue\n- Operating Margin = Operating Income / Revenue\n- Net Margin = Net Income / Revenue\n\nIn ERP: Revenue accounts (credit balance) minus Expense accounts (debit balance) = Net Income."
    },
    {
        "instruction": "What is accrual accounting vs cash accounting?",
        "response": "Accrual Accounting: Revenue is recognized when earned and expenses when incurred, regardless of when cash is received or paid.\n\nExample: Invoice issued December 31, cash received January 15.\n- Accrual: Revenue recorded in December\n- Cash basis: Revenue recorded in January\n\nCash Accounting: Revenue recognized when cash is received; expenses when cash is paid.\n\nERP systems (SAP, Oracle, Dynamics) use accrual accounting as the default because:\n- Required by GAAP and IFRS for public companies\n- Gives more accurate picture of financial health\n- Matches revenue with related expenses in the same period\n\nAccrual entries in ERP: Post accrual at period end, reverse at start of next period."
    },
    {
        "instruction": "What is working capital?",
        "response": "Working capital measures a company's short-term liquidity — its ability to pay short-term obligations.\n\nFormula: Working Capital = Current Assets - Current Liabilities\n\nComponents tracked in ERP:\nCurrent Assets: Cash, Accounts Receivable, Inventory, Prepaid Expenses\nCurrent Liabilities: Accounts Payable, Accrued Liabilities, Short-term debt\n\nKey ratios:\n- Current Ratio = Current Assets / Current Liabilities (target: >1.0)\n- Quick Ratio = (Cash + AR) / Current Liabilities (target: >0.8)\n\nWorking capital management in ERP:\n- Reduce DSO (collect AR faster)\n- Extend DPO (negotiate longer payment terms with vendors)\n- Reduce inventory (improve turnover)\n\nERP systems provide real-time working capital dashboards."
    },
    # --- Accounts Payable ---
    {
        "instruction": "What is the accounts payable process in ERP?",
        "response": "The Accounts Payable (AP) process in ERP manages money owed to vendors:\n\n1. Vendor Master Setup — create vendor with bank details, payment terms\n2. Purchase Order (PO) — purchasing team creates PO\n3. Goods Receipt (GR) — warehouse receives goods, GR posted in system\n4. Invoice Receipt — vendor sends invoice; entered in ERP (MIRO in SAP)\n5. Three-way Match — system matches PO qty/price with GR and invoice\n6. Invoice Approval — workflow routes exceptions for approval\n7. Payment Run — automatic payment program selects due invoices\n8. Payment Output — bank file or check generated\n9. Clearing — open items cleared in vendor account\n\nKey documents: PO, GR, Invoice, Payment\nKey accounts: GR/IR Clearing, Vendor AP, Cash/Bank"
    },
    {
        "instruction": "What are payment terms in ERP?",
        "response": "Payment terms define when a vendor invoice is due and whether early payment discounts apply.\n\nCommon payment terms:\n- Net 30 — full amount due in 30 days\n- Net 60 — full amount due in 60 days\n- 2/10 Net 30 — 2% discount if paid in 10 days, otherwise due in 30\n- Due on Receipt — immediately payable\n- End of Month (EOM) — due at end of the month\n\nIn ERP, payment terms are configured with:\n- Baseline date (invoice date, document date, delivery date)\n- Discount percentage and days\n- Net due days\n\nPayment terms are assigned to vendor master records and can be overridden on individual invoices.\n\nSAP T-code: OBB8 (configure payment terms), FK02 (assign to vendor)."
    },
    {
        "instruction": "What is a credit memo in AP?",
        "response": "A credit memo in Accounts Payable is a document from a vendor that reduces the amount owed by the company — essentially the reverse of an invoice.\n\nCommon reasons for AP credit memos:\n- Goods returned to vendor (damaged, incorrect, or excess)\n- Vendor pricing error on original invoice\n- Agreed discount or rebate\n- Overpayment adjustment\n\nAccounting entry for AP credit memo:\nDebit: Accounts Payable (reduces the liability)\nCredit: Inventory / Expense (reverses original entry)\n\nIn ERP:\n- SAP: FB65 (post vendor credit memo), MIRO with credit memo indicator\n- Oracle: Create Credit Memo in AP module\n- The credit memo offsets against open invoices during payment run\n\nCredit memos can be automatically applied to future vendor invoices."
    },
    {
        "instruction": "What is a GR/IR account in SAP?",
        "response": "The GR/IR (Goods Receipt / Invoice Receipt) account is a clearing account in SAP that bridges the gap between receiving goods and receiving the vendor invoice.\n\nHow it works:\n\nStep 1 — Goods Receipt (MIGO):\nDebit: Inventory (Stock Account)\nCredit: GR/IR Clearing Account\n\nStep 2 — Invoice Receipt (MIRO):\nDebit: GR/IR Clearing Account\nCredit: Vendor Accounts Payable\n\nResult: GR/IR clears to zero when both GR and invoice exist for same PO/quantity.\n\nUncleared GR/IR items indicate:\n- GR posted but invoice not yet received (goods received, awaiting invoice)\n- Invoice posted but goods not yet received (invoice in advance of delivery)\n\nGR/IR reconciliation is a key period-end task. T-code MB5S shows open GR/IR items."
    },
    {
        "instruction": "What is a vendor master record?",
        "response": "A vendor master record is the central repository for all information about a supplier in the ERP system. It contains:\n\nGeneral Data (shared across company codes):\n- Vendor name, address, contact information\n- Tax number (VAT/GST ID)\n- Bank account details for payment\n\nCompany Code Data (financial):\n- Reconciliation account (AP GL account)\n- Payment terms\n- Payment method (check, wire, ACH)\n- Withholding tax settings\n- Currency\n\nPurchasing Organization Data (procurement):\n- Order currency\n- Incoterms\n- Minimum order quantity\n- GR-based invoice verification flag\n\nSAP T-codes: MK01 (create purchasing vendor), FK01 (create accounting vendor), XK01 (create both).\n\nVendor master data quality is critical — errors cause payment failures and compliance issues."
    },
    # --- Accounts Receivable ---
    {
        "instruction": "What is the accounts receivable process in ERP?",
        "response": "The Accounts Receivable (AR) process manages money owed to the company by customers:\n\n1. Customer Master Setup — create customer with credit terms, payment method\n2. Sales Order — customer order entered in system\n3. Delivery / Goods Issue — goods shipped, inventory reduced\n4. Billing / Invoice — customer invoice generated and sent\n5. Cash Application — customer payment received and applied to invoice\n6. Dunning — reminders sent for overdue invoices\n7. Dispute Management — handle invoice disputes and deductions\n8. Write-off — bad debts written off after collection exhausted\n\nKey metrics:\n- DSO (Days Sales Outstanding) — average days to collect AR\n- Aging Report — overdue invoices by time bucket (0-30, 31-60, 61-90, 90+ days)\n- Collection Rate — % of invoices collected on time\n\nSAP T-codes: VF01 (billing), F-28 (post payment), F150 (dunning)."
    },
    {
        "instruction": "What is Days Sales Outstanding (DSO)?",
        "response": "Days Sales Outstanding (DSO) measures the average number of days a company takes to collect payment after a sale.\n\nFormula: DSO = (Accounts Receivable / Total Credit Sales) × Number of Days\n\nExample:\n- AR balance: $500,000\n- Monthly credit sales: $1,000,000\n- DSO = (500,000 / 1,000,000) × 30 = 15 days\n\nInterpretation:\n- Lower DSO = faster collection = better cash flow\n- Higher DSO = slower collection = potential cash flow problems\n- Compare to payment terms: DSO should be close to your standard payment terms\n\nHow ERP helps reduce DSO:\n- Automated invoice delivery\n- Early payment discount offers\n- Automated dunning (payment reminders)\n- Real-time AR aging visibility\n- Credit limit enforcement to avoid new sales to poor payers"
    },
    {
        "instruction": "What is a customer master record in ERP?",
        "response": "A customer master record stores all information about a customer used across sales and finance processes:\n\nGeneral Data:\n- Customer name, address, contact information\n- Tax number (VAT registration)\n- Industry, customer group\n\nCompany Code Data (financial):\n- Reconciliation account (AR GL account)\n- Payment terms\n- Dunning procedure and level\n- Credit management settings\n- Tolerance for payment differences\n\nSales Area Data (commercial):\n- Sales organization, distribution channel, division\n- Pricing procedure\n- Shipping conditions, delivery priority\n- Account assignment group (for revenue posting)\n\nSAP T-codes: FD01 (create accounting customer), VD01 (create sales customer), XD01 (create both).\n\nAccurate customer master data ensures correct invoicing, tax calculation, and payment matching."
    },
    # --- Fixed Assets ---
    {
        "instruction": "What is fixed asset accounting in ERP?",
        "response": "Fixed Asset Accounting manages the lifecycle of long-term assets in the ERP system:\n\nAsset Lifecycle:\n1. Acquisition — purchase or creation of asset\n2. Capitalization — recording as asset (not expense)\n3. Depreciation — systematic cost allocation over useful life\n4. Transfer — move asset between cost centers or plants\n5. Retirement / Disposal — sell, scrap, or write off asset\n\nKey data per asset:\n- Asset description and category\n- Acquisition date and cost\n- Useful life and depreciation method\n- Location (plant, cost center)\n- Depreciation areas (GAAP, Tax, IFRS)\n\nAccounting at acquisition:\nDebit: Fixed Asset Account\nCredit: Vendor AP or Cash\n\nMonthly depreciation posting:\nDebit: Depreciation Expense\nCredit: Accumulated Depreciation\n\nSAP Module: FI-AA. T-codes: AS01 (create asset), AFAB (depreciation run)."
    },
    {
        "instruction": "What is the difference between capital expenditure and operating expenditure?",
        "response": "CapEx vs OpEx is a fundamental distinction in ERP and accounting:\n\nCapital Expenditure (CapEx):\n- Spending on long-term assets expected to provide value >1 year\n- Examples: buildings, machinery, vehicles, software, equipment\n- Recorded as an asset on the balance sheet\n- Cost spread over useful life via depreciation\n- ERP treatment: capitalize to fixed asset master record\n\nOperating Expenditure (OpEx):\n- Day-to-day business expenses for running operations\n- Examples: rent, utilities, salaries, office supplies, maintenance\n- Expensed immediately in the period incurred\n- Reduces profit directly\n- ERP treatment: post to expense GL account\n\nWhy it matters:\n- CapEx affects cash flow statement (investing activities)\n- OpEx affects income statement (operating expenses)\n- Tax treatment differs (OpEx fully deductible; CapEx deducted via depreciation)\n\nERP systems enforce capitalization thresholds — purchases below a limit are auto-expensed."
    },
    # --- Inventory ---
    {
        "instruction": "What is a bill of materials (BOM) in ERP?",
        "response": "A Bill of Materials (BOM) is a structured list of all components, materials, and quantities required to produce a finished product.\n\nBOM structure example (Bicycle):\n- 1x Frame\n- 2x Wheel assembly\n  - 1x Rim\n  - 1x Tire\n  - 32x Spoke\n- 1x Handlebar\n- 1x Saddle\n- 1x Chain\n- 2x Brake assembly\n\nBOM uses in ERP:\n- MRP (Material Requirements Planning) — calculates component demand\n- Production Orders — lists materials to be issued to production\n- Cost Estimation — calculates standard cost of finished goods\n- Sales Orders — configure-to-order products\n\nBOM types in SAP:\n- Production BOM (most common)\n- Sales BOM (packs items together for sales)\n- Engineering BOM\n\nT-codes: CS01 (create BOM), CS02 (change), CS03 (display), CS11 (BOM explosion)."
    },
    {
        "instruction": "What is a warehouse management system (WMS) in ERP?",
        "response": "Warehouse Management System (WMS) is the ERP module that manages physical storage locations and warehouse operations:\n\nCore Functions:\n- Bin/Storage Location Management — tracks exact physical location of stock\n- Inbound Processing — goods receipt, put-away strategies\n- Outbound Processing — pick, pack, ship\n- Transfer Orders — moves within warehouse\n- Inventory Counts — cycle counting and physical inventory\n- Slotting — optimal storage location assignment\n\nKey concepts:\n- Warehouse Number — top-level warehouse structure\n- Storage Type — area within warehouse (high-rack, bulk, picking)\n- Storage Section — subdivision of storage type\n- Bin — individual storage unit (shelf, location)\n\nSAP WM (LT0A, LT01 transfer orders) vs SAP Extended Warehouse Management (EWM) for complex warehouses.\n\nWMS integration: WMS connects to Inventory Management (MM-IM) for stock updates and SD for delivery processing."
    },
    {
        "instruction": "What is cycle counting in ERP?",
        "response": "Cycle counting is a continuous inventory auditing method where a subset of inventory is counted on a rotating schedule rather than doing one large annual physical inventory.\n\nAdvantages over annual physical inventory:\n- No need to shut down operations\n- Errors discovered and corrected faster\n- Higher accuracy — more frequent counting\n- Less disruptive to daily operations\n\nHow cycle counting works in ERP:\n1. Assign ABC classification to items (A=high value/fast moving, C=low value/slow)\n2. Schedule counting frequency: A-items monthly, B-items quarterly, C-items annually\n3. System generates count documents for selected items\n4. Warehouse staff counts physical stock\n5. Enter count in ERP and system calculates variance\n6. Approve or investigate variances above tolerance\n7. Post inventory adjustment if needed\n\nSAP: T-code MI01 (create inventory document), MI04 (enter count), MI07 (post differences)."
    },
    {
        "instruction": "What is safety stock in ERP?",
        "response": "Safety stock is a buffer quantity of inventory maintained to prevent stockouts caused by demand variability or supply delays.\n\nPurpose:\n- Protects against unexpected demand spikes\n- Covers for supplier delivery delays\n- Prevents production stoppages due to material shortages\n\nHow ERP calculates reorder points:\nReorder Point = (Average Daily Demand × Lead Time) + Safety Stock\n\nExample:\n- Average daily demand: 100 units\n- Lead time: 5 days\n- Safety stock: 200 units\n- Reorder Point = (100 × 5) + 200 = 700 units\n\nWhen stock drops to 700 units, ERP automatically generates a purchase requisition.\n\nFactors determining safety stock level:\n- Demand variability (standard deviation of demand)\n- Supply reliability (supplier lead time variability)\n- Desired service level (e.g., 95% fill rate)\n- Carrying cost vs. stockout cost\n\nSAP: Set in Material Master MRP view (MRP1 tab), field Safety Stock."
    },
    {
        "instruction": "What is a stock transfer order in ERP?",
        "response": "A Stock Transfer Order (STO) is an ERP document used to move materials between plants or storage locations within the same company.\n\nTypes of stock transfers:\n1. Storage Location to Storage Location — within same plant\n   (Movement type 311/312 in SAP)\n\n2. Plant to Plant — within same company code\n   (One-step: movement type 301; Two-step: 303/305)\n\n3. Company Code to Company Code — intercompany transfer\n   (Requires purchase order between company codes)\n\nSTO process (two-step plant-to-plant):\n1. Create STO (T-code ME21N with document type UB)\n2. Post goods issue at sending plant (MIGO, movement 351)\n3. Material in transit\n4. Post goods receipt at receiving plant (MIGO, movement 101)\n\nBenefits:\n- Balances inventory between locations\n- Enables central procurement with distribution\n- Tracks materials in transit accurately"
    },
    # --- Manufacturing ---
    {
        "instruction": "What is a production order in ERP?",
        "response": "A production order is an ERP document that authorizes and tracks the manufacture of a specific quantity of a product.\n\nProduction Order Lifecycle:\n1. Creation — system creates order (from MRP or manually) with BOM and routing\n2. Release — order released to production floor\n3. Goods Issue — components issued from stock to production order\n4. Confirmation — production quantities, times, and scrap reported\n5. Goods Receipt — finished goods received into stock\n6. Settlement — production variances settled to cost object\n7. Closure — order technically closed\n\nKey data in a production order:\n- Material to be produced and quantity\n- Production dates (start/finish)\n- Bill of Materials (component list)\n- Routing (work center operations and times)\n- Planned vs actual costs\n\nAccounting entries:\n- Goods Issue: Debit WIP, Credit Inventory\n- Goods Receipt: Debit Finished Goods, Credit WIP\n- Variance: Debit Variance Account, Credit WIP\n\nSAP T-codes: CO01 (create), CO02 (change), CO11N (confirm), COGI (reprocess errors)."
    },
    {
        "instruction": "What is a routing in manufacturing ERP?",
        "response": "A routing (also called a work center routing or operation list) defines the sequence of operations and work centers required to manufacture a product.\n\nRouting components:\n- Operations — individual manufacturing steps (e.g., cutting, welding, painting)\n- Work Centers — machines or labor pools where operations occur\n- Standard Times — setup time, machine time, labor time per operation\n- Control Keys — define how operations are confirmed and costed\n\nExample routing for a metal bracket:\nOperation 10: Laser Cutting — Work Center: Laser Machine — 0.5 hr\nOperation 20: Bending — Work Center: Press Brake — 0.25 hr\nOperation 30: Welding — Work Center: Welding Station — 1.0 hr\nOperation 40: Inspection — Work Center: QA Station — 0.25 hr\n\nRouting is used for:\n- Production scheduling (capacity planning)\n- Standard cost calculation (labor + machine costs)\n- Shop floor control and confirmation\n\nSAP T-codes: CA01 (create routing), CA02 (change), CA03 (display)."
    },
    # --- HR / Payroll ---
    {
        "instruction": "What is HR master data in ERP?",
        "response": "HR master data in ERP (called Infotypes in SAP HCM) stores all employee information across functional areas:\n\nPersonal Data:\n- Name, date of birth, gender, nationality\n- Contact information, emergency contacts\n- Tax information, social security number\n\nOrganizational Data:\n- Position, job title, department\n- Cost center (for payroll cost allocation)\n- Reporting line (manager)\n- Work schedule and shift\n\nPayroll Data:\n- Base salary, pay grade\n- Bank details for payment\n- Tax withholding setup\n- Benefits enrollment\n\nTime Management:\n- Leave entitlements and balances\n- Work schedule\n- Time recording method\n\nSAP HCM Infotypes: IT0000 (Actions), IT0001 (Org Assignment), IT0002 (Personal Data), IT0008 (Basic Pay), IT0009 (Bank Details).\n\nT-codes: PA30 (maintain HR master data), PA20 (display)."
    },
    {
        "instruction": "What is payroll processing in ERP?",
        "response": "Payroll processing in ERP calculates employee wages and generates payment and accounting entries:\n\nPayroll Process Steps:\n1. Time Data Collection — work hours, overtime, absences recorded\n2. Payroll Run — system calculates gross pay, deductions, net pay\n3. Payroll Results Check — verify calculations before posting\n4. Bank File Generation — payment file sent to bank\n5. Posting to FI — payroll costs posted to GL accounts\n6. Payslip Generation — employee pay statements produced\n\nPayroll components calculated:\n- Gross Pay: Base salary + overtime + bonuses + allowances\n- Statutory Deductions: Income tax, social security, pension\n- Voluntary Deductions: Health insurance, 401k, union dues\n- Net Pay: Gross Pay - All Deductions\n\nAccounting entries at payroll posting:\nDebit: Salary Expense (by cost center)\nCredit: Employee Payable (net pay)\nCredit: Tax Withholding Payable\nCredit: Benefits Payable\n\nSAP T-codes: PC00_Mxx_CALC (payroll run), PC00_Mxx_CIPE (post to FI)."
    },
    # --- Sales ---
    {
        "instruction": "What is a sales order in ERP?",
        "response": "A sales order (SO) is an ERP document that records a customer's request to purchase goods or services at agreed prices and quantities.\n\nSales Order creation process:\n1. Customer inquiry / quotation (optional)\n2. Sales Order created (with customer, materials, quantities, prices)\n3. System checks: credit limit, material availability\n4. Order confirmed to customer\n5. Delivery created from SO\n6. Goods issued (inventory reduced)\n7. Billing document (invoice) created\n8. Customer payment received\n\nKey fields in a sales order:\n- Sold-to party (customer)\n- Ship-to party (delivery address)\n- Bill-to party (invoice address)\n- Material, quantity, price\n- Requested delivery date\n- Shipping conditions\n- Payment terms\n\nSAP Sales & Distribution (SD):\n- T-code VA01 (create SO), VA02 (change), VA03 (display)\n- Document type: OR (standard order)\n\nSales orders integrate with MM (inventory check), PP (make-to-order), and FI (billing)."
    },
    {
        "instruction": "What is pricing in SAP SD?",
        "response": "Pricing in SAP SD uses a condition technique to automatically determine the price of a product for a customer.\n\nKey components:\n\n1. Condition Types — define types of price components:\n   - PR00: Base Price\n   - K007: Customer Discount\n   - K004: Material Discount\n   - MWST: Output Tax (VAT)\n   - RB00: Absolute Discount\n   - SKTO: Cash Discount\n\n2. Condition Tables — define key combinations (e.g., customer + material)\n\n3. Access Sequence — defines search order for pricing records\n\n4. Pricing Procedure — sequence of condition types applied to calculate final price:\n   Base Price\n   - Customer Discount\n   - Volume Discount\n   = Net Price\n   + Freight\n   + Tax\n   = Final Price\n\n5. Condition Records — actual price/discount values stored per customer/material/validity period\n\nT-codes: VK11 (create condition records), V/08 (pricing procedure), OVKK (assign pricing procedure)."
    },
    # --- Reporting ---
    {
        "instruction": "What is an aging report in ERP?",
        "response": "An aging report categorizes outstanding invoices by how long they have been unpaid, organized into time buckets.\n\nAR Aging Report (customer invoices):\nCustomer | 0-30 days | 31-60 days | 61-90 days | 90+ days | Total\nABC Corp |   $5,000  |   $2,000   |    $500    |    $0    | $7,500\nXYZ Ltd  |   $1,000  |     $0     |  $3,000    | $8,000   | $12,000\n\nAP Aging Report (vendor invoices — what you owe):\nSame structure but for vendor invoices\n\nWhy aging reports matter:\n- Identifies overdue accounts for collection action\n- Highlights customers with chronic late payment\n- Inputs for bad debt provision (accounts >90 days at risk)\n- Cash flow forecasting\n- Credit limit review\n\nIn SAP:\n- AR Aging: T-code FBL5N (customer line items) or S_ALR_87012178\n- AP Aging: T-code FBL1N (vendor line items) or S_ALR_87012082\n\nAging is a key report reviewed by CFOs and AR/AP managers daily."
    },
    {
        "instruction": "What is a trial balance in ERP?",
        "response": "A trial balance is a report listing all general ledger accounts and their debit or credit balances at a specific date.\n\nPurpose:\n- Verify that total debits equal total credits (balanced books)\n- Starting point for financial statement preparation\n- Identify posting errors\n\nStructure:\nAccount | Account Name | Debit Balance | Credit Balance\n1000    | Cash         | $50,000       |\n1100    | AR           | $120,000      |\n2000    | AP           |               | $80,000\n3000    | Equity       |               | $90,000\n4000    | Revenue      |               | $200,000\n5000    | Expenses     | $200,000      |\nTOTAL   |              | $370,000      | $370,000\n\nTypes:\n- Unadjusted TB — before period-end adjustments\n- Adjusted TB — after accruals, deferrals, depreciation\n- Post-closing TB — after closing entries\n\nIn SAP: T-code S_ALR_87012277 (GL Account Balances), FS10N (individual account)\nIn Oracle: Reports > Trial Balance"
    },
    # --- General ERP ---
    {
        "instruction": "What is an ERP system?",
        "response": "An ERP (Enterprise Resource Planning) system is integrated business management software that centralizes and automates core business processes across an organization in a single database.\n\nCore capabilities:\n- Finance and Accounting (GL, AP, AR, Fixed Assets)\n- Procurement and Supply Chain\n- Inventory and Warehouse Management\n- Manufacturing and Production Planning\n- Sales and Order Management\n- Human Resources and Payroll\n- Project Management\n- Business Intelligence and Reporting\n\nKey benefits:\n- Single source of truth — all departments share one database\n- Real-time visibility across the business\n- Automated workflows and approvals\n- Regulatory compliance and audit trails\n- Reduced manual data entry and errors\n\nLeading ERP vendors: SAP S/4HANA, Oracle ERP Cloud, Microsoft Dynamics 365, Odoo, NetSuite, Infor, Epicor."
    },
    {
        "instruction": "What is a general ledger in ERP?",
        "response": "The General Ledger (GL) is the central accounting record in ERP that contains all financial transactions of an organization.\n\nKey characteristics:\n- Complete record of all financial transactions\n- Organized by GL account and cost object\n- Uses double-entry bookkeeping (every entry balanced)\n- Source of all financial statements\n\nGL account types:\n- Balance Sheet accounts: Assets, Liabilities, Equity (carry balance forward)\n- P&L accounts: Revenue, Expenses (reset to zero at year-end)\n\nPostings to GL come from:\n- Accounts Payable (vendor invoices, payments)\n- Accounts Receivable (customer invoices, receipts)\n- Fixed Assets (depreciation, acquisitions)\n- Payroll (salary expense, tax liabilities)\n- Inventory (goods movements)\n- Manual journal entries\n\nKey GL reports:\n- Trial Balance\n- Financial Statements (P&L, Balance Sheet)\n- GL Account Statement (detailed transaction list)\n\nSAP T-codes: FS00 (GL master), FB50 (post GL document), FS10N (account balance)."
    },
    {
        "instruction": "What is accounts payable?",
        "response": "Accounts Payable (AP) is the money a company owes to its vendors and suppliers for goods and services received but not yet paid for.\n\nAP represents a current liability on the balance sheet.\n\nKey AP activities:\n- Recording vendor invoices\n- Matching invoices to purchase orders and goods receipts\n- Routing invoices for approval\n- Scheduling and executing vendor payments\n- Managing vendor master data\n- Reconciling vendor statements\n- Handling payment disputes and credit memos\n\nAccounting entry when invoice received:\nDebit: Expense or Inventory\nCredit: Accounts Payable\n\nAccounting entry when payment made:\nDebit: Accounts Payable\nCredit: Cash/Bank\n\nKey AP metrics:\n- Days Payable Outstanding (DPO): how long you take to pay vendors\n- Invoice processing time\n- Early payment discount capture rate\n- Vendor payment accuracy rate"
    },
    {
        "instruction": "What is accounts receivable?",
        "response": "Accounts Receivable (AR) is money owed to a company by its customers for goods or services delivered but not yet paid for.\n\nAR represents a current asset on the balance sheet.\n\nKey AR activities:\n- Generating customer invoices\n- Sending invoices and statements\n- Recording customer payments\n- Applying cash to open invoices\n- Managing credit limits\n- Collections and dunning for overdue accounts\n- Handling disputes and deductions\n- Writing off uncollectible bad debts\n\nAccounting entry when invoice issued:\nDebit: Accounts Receivable\nCredit: Revenue\n\nAccounting entry when payment received:\nDebit: Cash/Bank\nCredit: Accounts Receivable\n\nKey AR metrics:\n- Days Sales Outstanding (DSO): average days to collect payment\n- AR Aging: distribution of overdue balances\n- Bad Debt Rate: % of AR written off\n- Collection Effectiveness Index (CEI)"
    },
    {
        "instruction": "What is purchase order?",
        "response": "A Purchase Order (PO) is a formal commercial document issued by a buyer to a supplier, authorizing the purchase of goods or services at agreed terms.\n\nKey elements of a Purchase Order:\n- PO Number (unique identifier)\n- Vendor name and address\n- Delivery address\n- Line items: material/service, quantity, unit of measure, price\n- Delivery date\n- Payment terms\n- Terms and conditions\n\nTypes of Purchase Orders:\n- Standard PO — one-time purchase\n- Blanket PO / Framework Order — multiple deliveries over a period at fixed price\n- Contract — long-term price/quantity agreement\n- Scheduling Agreement — delivery schedule defined in advance\n\nPO lifecycle:\nPR → RFQ → Quotation → PO → GR → Invoice → Payment\n\nLegal significance: A signed PO is a legally binding contract.\n\nSAP T-codes: ME21N (create PO), ME22N (change), ME23N (display), ME2M (PO list by material)."
    },
    {
        "instruction": "What is inventory management in ERP?",
        "response": "Inventory Management in ERP tracks the quantity, value, and location of materials throughout their lifecycle.\n\nCore functions:\n- Goods Receipt — recording incoming materials\n- Goods Issue — recording material consumption or shipment\n- Stock Transfer — moving materials between locations\n- Physical Inventory — counting and reconciling stock\n- Valuation — maintaining material value (FIFO, standard cost, moving average)\n\nInventory types managed:\n- Unrestricted Stock — available for use/sale\n- Quality Inspection Stock — awaiting QC approval\n- Blocked Stock — rejected or restricted materials\n- Stock in Transit — transferred but not yet received\n- Consignment Stock — vendor-owned stock at your premises\n\nKey reports:\n- Stock Overview (MMBE in SAP) — current stock levels by plant/storage location\n- Material Document List (MB51) — all goods movements\n- Inventory Valuation Report — stock value by material/plant\n\nInventory accuracy = (Counted Quantity / System Quantity) × 100%. Target: >99%"
    },
    {
        "instruction": "How does an ERP help with financial reporting?",
        "response": "ERP systems automate and improve financial reporting through:\n\n1. Real-time Data — all transactions immediately available for reporting\n\n2. Standard Financial Statements:\n   - Income Statement (P&L)\n   - Balance Sheet\n   - Cash Flow Statement\n   - Statement of Changes in Equity\n\n3. Management Reports:\n   - Cost center reports (budget vs actual)\n   - Profit center P&L\n   - Project profitability\n   - Segment reporting\n\n4. Regulatory Compliance:\n   - GAAP / IFRS compliant ledgers\n   - Tax reporting (VAT, withholding tax)\n   - Statutory reporting by country\n\n5. Analytics and Dashboards:\n   - KPI dashboards for CFO and management\n   - Drill-down from summary to transaction level\n   - Variance analysis tools\n\n6. Audit Trail:\n   - Every transaction traceable to source document\n   - User stamps, timestamps, change history\n   - Supports internal and external audits\n\nReporting tools: SAP Analytics Cloud, Oracle OTBI, Dynamics 365 Power BI integration."
    },
    {
        "instruction": "What is a fiscal year in ERP?",
        "response": "A fiscal year is the 12-month accounting period a company uses for financial reporting and tax purposes. It does not have to align with the calendar year.\n\nCommon fiscal year types:\n- Calendar Year: January 1 – December 31\n- April – March: Common in India and UK\n- July – June: Common in Australia\n- October – September: Common in US government\n\nIn ERP, the fiscal year defines:\n- Accounting periods (usually 12 monthly periods + special periods)\n- Period open/close dates\n- Year-end closing procedures\n- Carryforward of balances\n\nFiscal year variants in SAP:\n- V3: April to March (4 special periods)\n- K4: Calendar year\n- Custom variants for non-standard year-ends\n\nPeriod status in ERP:\n- Open: Transactions can be posted\n- Closed: No posting allowed (locked after close)\n- Future: Can be opened in advance for next period\n\nT-code in SAP: OB29 (fiscal year variant), OB52 (open/close periods)."
    },
    {
        "instruction": "What is tax configuration in ERP?",
        "response": "Tax configuration in ERP ensures correct tax calculation, posting, and reporting for all transactions.\n\nTypes of taxes managed in ERP:\n- VAT / GST (Value Added Tax / Goods and Services Tax)\n- Sales Tax (US state and local)\n- Withholding Tax (WHT) — deducted from payments\n- Import / Customs Duty\n- Excise Duty\n\nVAT / GST in ERP:\n1. Tax codes defined (e.g., I1=Input VAT 20%, O1=Output VAT 20%)\n2. Tax code assigned to vendor/customer or transaction\n3. System automatically calculates tax amount\n4. Tax posted to separate GL account (Tax Payable / Tax Receivable)\n5. Periodic tax returns generated from ERP data\n\nAccounting entries (VAT on vendor invoice):\nDebit: Expense/Inventory (net amount)\nDebit: Input VAT Receivable (tax amount)\nCredit: Accounts Payable (gross amount)\n\nSAP T-codes: FTXP (configure tax codes), FK02 (assign to vendor).\nOracle: Tax Manager module for configuration."
    },
]

ALL_QA = ERP_QA + ADDITIONAL_QA + EXPANDED_ERP_QA


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)

    all_pairs = list(ALL_QA)
    random.shuffle(all_pairs)

    # Split 95/5 train/val
    cut = int(len(all_pairs) * 0.95)
    train_pairs = all_pairs[:cut]
    val_pairs   = all_pairs[cut:]

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✓ ERP SFT dataset ready")
    print(f"  Train: {len(train_pairs):,} pairs → {OUT_TRAIN}")
    print(f"  Val:   {len(val_pairs):,} pairs → {OUT_VAL}")
    print(f"\nNext step — SFT fine-tuning:")
    print("  python finetune.py --stage sft --config configs/erp.yaml \\")
    print("      --base-checkpoint checkpoints/general/step_00060000 \\")
    print("      --data data/erp_sft_train.jsonl --tokenizer tokenizer \\")
    print("      --checkpoint-dir checkpoints/erp_sft")


if __name__ == "__main__":
    main()
