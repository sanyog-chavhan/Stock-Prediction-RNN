import yfinance as yf

START_DATE = "2009-08-08"
END_DATE = "2020-08-13"

df = yf.download("AAPL", start=START_DATE, end=END_DATE, auto_adjust=True)

# Newer yfinance versions return MultiIndex columns; flatten them
df.columns = df.columns.get_level_values(0)

df.to_csv("aapl_prices.csv")
