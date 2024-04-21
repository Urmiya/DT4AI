# API Configuration
API_KEY = "72db1ad9bf86aa308c27cc9d148ca17c"
BASE_URL = "https://financialmodelingprep.com/api/v3/"
HISTORICAL_PRICE_URL = f"{BASE_URL}historical-price-full/{{}}?serietype=line&apikey={API_KEY}"

# Thresholds for price changes
STOCK_THRESHOLD = 2.5   # Threshold for stocks
ETF_THRESHOLD = 1.5     # Threshold for ETFs

SWISS_STOCKS = [
    {'symbol': 'NESN.SW', 'type': 'Stock'},  # Nestlé SA
    {'symbol': 'ROG.SW',  'type': 'Stock'},  # Roche Holding AG
    {'symbol': 'NOVN.SW', 'type': 'Stock'},  # Novartis AG
    {'symbol': 'ZURN.SW', 'type': 'Stock'},  # Zurich Insurance Group
    {'symbol': 'UBSG.SW', 'type': 'Stock'},  # UBS Group AG
    {'symbol': 'ABBN.SW', 'type': 'Stock'},  # ABB Ltd
    {'symbol': 'CFR.SW',  'type': 'Stock'},  # Compagnie Financière Richemont SA
    {'symbol': 'LONN.SW', 'type': 'Stock'},  # Lonza Group AG
    {'symbol': 'SGSN.SW', 'type': 'Stock'},  # SGS SA
    {'symbol': 'SREN.SW', 'type': 'Stock'},  # Swiss Re AG
    {'symbol': 'GIVN.SW', 'type': 'Stock'},  # Givaudan SA
    {'symbol': 'CSGN.SW', 'type': 'Stock'},  # Credit Suisse Group AG
    {'symbol': 'GEBN.SW', 'type': 'Stock'},  # Geberit AG
    {'symbol': 'SCMN.SW', 'type': 'Stock'},  # Swisscom AG
    {'symbol': 'SLHN.SW', 'type': 'Stock'},  # Swiss Life Holding AG
    {'symbol': 'SIKA.SW', 'type': 'Stock'},  # Sika AG
    {'symbol': 'ALC.SW',  'type': 'Stock'},  # Alcon Inc
    {'symbol': 'PGHN.SW', 'type': 'Stock'},  # Partners Group Holding AG
    {'symbol': 'ADEN.SW', 'type': 'Stock'}   # The Adecco Group
]

SWISS_MARKET_ASSETS = [
    # Stocks
    {'symbol': 'NESN.SW', 'type': 'Stock'},  # Nestle SA
    {'symbol': 'NOVN.SW', 'type': 'Stock'},  # Novartis AG
    {'symbol': 'ROG.SW', 'type': 'Stock'},   # Roche Holding AG
    {'symbol': 'ZURN.SW', 'type': 'Stock'},  # Zurich Insurance Group
    {'symbol': 'UBSG.SW', 'type': 'Stock'},  # UBS Group AG
    {'symbol': 'CSGN.SW', 'type': 'Stock'},  # Credit Suisse Group AG
    {'symbol': 'SCMN.SW', 'type': 'Stock'},  # Swisscom AG
    {'symbol': 'ABBN.SW', 'type': 'Stock'},  # ABB Ltd
    {'symbol': 'SGSN.SW', 'type': 'Stock'},  # SGS SA
    {'symbol': 'SREN.SW', 'type': 'Stock'},  # Swiss Re AG
    
    # ETFs 
    {'symbol': 'BND',     'type': 'ETF'},    # Vanguard Total Bond Market Index Fund
    {'symbol': 'JREE.SW', 'type': 'ETF'},    # JPMorgan ETFs (Ireland) ICAV - Europe Research Enhanced Index Equity (ESG) UCITS ETF
    {'symbol': 'FEMS',    'type': 'ETF'},    # First Trust Emerging Markets Small Cap AlphaDEX Fund
    {'symbol': 'PSYK',    'type': 'ETF'},    # PSYK ETF
    {'symbol': 'IUS3.DE', 'type': 'ETF'},    # iShares S&P Small Cap 600 UCITS ETF USD (Dist)
    {'symbol': '4UB1.F',  'type': 'ETF'},    # UBS (Irl) ETF plc - MSCI World Socially Responsible UCITS ETF
    {'symbol': 'XBCD.DE', 'type': 'ETF'},    # Xtrackers II - iBoxx Germany Covered Bond Swap UCITS ETF
    {'symbol': 'ISFU.L',  'type': 'ETF'},    # iShares Core FTSE 100 UCITS ETF GBP (Dist)
    {'symbol': 'IBMK',    'type': 'ETF'}     # iShares iBonds Dec 2022 Term Muni Bond ETF
]