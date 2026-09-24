"""Sector configuration.

Each stock maps to a sector. A sector either has an official NSE index with usable history on
Yahoo Finance (`SECTOR_INDEX_TICKERS`) or is represented by a leave-one-out, equal-weight
composite of its peers in the training universe (see `features.market.build_peer_sector_index`).

Only Bank Nifty, Nifty IT and Nifty Pharma currently have multi-year history on Yahoo Finance;
other sector symbols (^CNXAUTO, ^CNXFMCG, ...) return only the latest quote, so they use peers.
Add new sectors or index tickers here without touching the feature code.
"""

SECTOR_INDEX_TICKERS = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
}

SECTOR_MEMBERS = {
    "BANK": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS", "YESBANK.NS",
        "BANDHANBNK.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "AUBANK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
        "RBLBANK.NS", "UNIONBANK.NS", "MAHABANK.NS", "J&KBANK.NS", "KARURVYSYA.NS", "SOUTHBANK.NS",
    ],
    "FINANCE": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS", "ABCAPITAL.NS",
        "MFSL.NS", "LICHSGFIN.NS", "POONAWALLA.NS", "LTF.NS", "M&MFIN.NS", "MANAPPURAM.NS", "ANGELONE.NS",
        "MCX.NS", "BSE.NS", "CDSL.NS", "HUDCO.NS", "NAM-INDIA.NS", "HDFCAMC.NS",
    ],
    "IT": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS", "LTIM.NS", "MPHASIS.NS", "COFORGE.NS",
        "PERSISTENT.NS", "ZENSARTECH.NS", "KPITTECH.NS", "TATAELXSI.NS",
    ],
    "INTERNET": ["NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS", "INDIAMART.NS", "DELHIVERY.NS"],
    "PHARMA": [
        "SUNPHARMA.NS", "DIVISLAB.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "LUPIN.NS", "GLENMARK.NS",
        "ALKEM.NS", "SYNGENE.NS", "LAURUSLABS.NS", "JBCHEPHARM.NS", "ZYDUSLIFE.NS", "NATCOPHARM.NS",
    ],
    "AUTO": [
        "MARUTI.NS", "M&M.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS",
        "ESCORTS.NS", "BALKRISIND.NS", "MRF.NS", "APOLLOTYRE.NS", "EXIDEIND.NS", "TIINDIA.NS",
    ],
    "FMCG": [
        "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "VBL.NS", "GODREJCP.NS",
        "DABUR.NS", "COLPAL.NS", "RADICO.NS",
    ],
    "METALS": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "JINDALSTEL.NS", "SAIL.NS", "NMDC.NS",
        "NATIONALUM.NS", "VEDL.NS", "HINDZINC.NS",
    ],
    "OIL_GAS": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "GAIL.NS", "OIL.NS", "HINDPETRO.NS", "IOC.NS", "MRPL.NS",
        "CHENNPETRO.NS", "PETRONET.NS",
    ],
    "POWER": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "NHPC.NS", "SJVN.NS", "TORNTPOWER.NS", "CESC.NS"],
    "CAPITAL_GOODS": [
        "LT.NS", "SIEMENS.NS", "HAL.NS", "BEL.NS", "ABB.NS", "POLYCAB.NS", "BHEL.NS", "MAZDOCK.NS",
        "COCHINSHIP.NS", "CUMMINSIND.NS", "SKFINDIA.NS", "TIMKEN.NS", "CARBORUNIV.NS", "KEC.NS", "KPIL.NS",
        "KEI.NS", "KAYNES.NS",
    ],
    "CONSUMER_DURABLES": ["TITAN.NS", "HAVELLS.NS", "VOLTAS.NS", "BLUESTARCO.NS", "DIXON.NS", "AMBER.NS"],
    "CONSUMER_RETAIL": [
        "TRENT.NS", "PAGEIND.NS", "METROBRAND.NS", "BATAINDIA.NS", "RELAXO.NS", "ABFRL.NS", "JUBLFOOD.NS",
        "RAYMOND.NS",
    ],
    "MATERIALS": [
        "ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "SRF.NS",
        "UPL.NS", "TATACHEM.NS", "DEEPAKNTR.NS", "AARTIIND.NS", "PIIND.NS", "COROMANDEL.NS", "GNFC.NS",
        "ASTRAL.NS",
    ],
    "INFRA_LOGISTICS": ["ADANIENT.NS", "ADANIPORTS.NS", "CONCOR.NS", "IRCTC.NS", "RVNL.NS", "NBCC.NS", "IRB.NS"],
    "REALTY": ["DLF.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "GODREJPROP.NS", "BRIGADE.NS", "PRESTIGE.NS"],
    "TELECOM": ["BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS"],
}

STOCK_SECTOR = {stock: sector for sector, members in SECTOR_MEMBERS.items() for stock in members}


def get_sector(ticker):
    """Sector name for a ticker, or None when the ticker is not mapped (sector features become NaN)."""
    return STOCK_SECTOR.get(ticker)


def get_sector_peers(ticker):
    sector = get_sector(ticker)
    return [s for s in SECTOR_MEMBERS.get(sector, []) if s != ticker]
