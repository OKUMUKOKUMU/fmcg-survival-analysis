"""Generate synthetic new-SKU listing data for a survival (time-to-delisting) analysis.

Context: a Kenyan FMCG dairy & deli manufacturer launches about 15 new SKUs over two years
(yoghurt, cheese, ice cream, deli meats, butter & cream, flavoured milk). Each launch is
"listed" (ranged) in a few hundred retail outlets: supermarkets, dukas, wholesalers,
hotels/restaurants (HoReCa) and petrol-station shops. Some listings survive for years;
many are delisted within months when the outlet stops re-ordering or a range review drops them.

How the data is built (so there is real structure to discover):
  * 2,600 outlets, each with a channel, region, distance to the depot, merchandiser
    coverage and (for some) a closure date. When an outlet closes, every listing it
    holds is CENSORED at the closure date (we never see whether it would have delisted).
  * 15 SKU launches between Mar 2024 and Feb 2026. Each SKU rolls out to outlets over
    several weeks after launch, so listings enter the study at different calendar dates
    (STAGGERED ENTRY). Follow-up stops at the study cut-off, 30 Jun 2026
    (ADMINISTRATIVE CENSORING): late launches are observed for a much shorter time.
  * Listing contracts guarantee a 4-week minimum listing period, so no delisting happens
    before week 4. Early sell-through (% of the opening stock sold in the first 4 weeks)
    is therefore known for every listing before any delisting can happen.
  * The weekly delisting hazard follows a hump-shaped (log-logistic-type) baseline that
    peaks around the first quarterly range review, multiplied by exp(linear predictor):
        - early sell-through          strongly protective
        - trade promotion funding     protective ONLY while the launch promo runs
                                      (weeks 4-16), slightly harmful afterwards
                                      -> a genuinely NON-PROPORTIONAL effect
        - POS material / planogram    protective
        - merchandiser coverage       protective
        - price index vs category     premium pricing raises the hazard
        - retailer margin             higher margin protects the listing
        - distance to depot           remote outlets get patchier deliveries
        - shelf life (SKU level)      short shelf life -> expiries -> delisting
        - channel and category        e.g. petrol stations / dukas and ice cream
                                      (freezer space) delist faster
        - an unobserved SKU "appeal" frailty (some launches are simply stronger)
  * Trade promotion and POS material also raise early sell-through, so sell-through is
    partly a MEDIATOR of launch support (this matters for the business translation).

Output:
  data/sku_listings.csv   one row per outlet x new-SKU listing

All data is randomly generated. No real company, outlet or sales data is included.
Usage: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260630)
OUT = Path(__file__).parent
CUTOFF = pd.Timestamp("2026-06-30")
N_OUTLETS = 2_600
MIN_LISTING_WEEKS = 4
PROMO_WEEKS = 16           # launch promotion runs until week 16 of the listing

# ------------------------------------------------------------------------------------
# Outlets
# ------------------------------------------------------------------------------------
CHANNELS = {  # share, P(listed in a launch) weight, P(trade promo), P(POS), log-HR, retailer margin mean (%)
    "Supermarket":    (.14, 3.0, .65, .65, 0.00, 18),
    "Wholesaler":     (.10, 2.2, .50, .25, -0.20, 10),
    "Duka":           (.44, 0.6, .25, .20, 0.35, 16),
    "HoReCa":         (.17, 0.9, .30, .15, 0.10, 22),
    "Petrol station": (.15, 1.0, .35, .35, 0.45, 20),
}
REGIONS = {  # share, distance-to-depot range (km)
    "Nairobi": (.34, (5, 45)), "Central": (.18, (25, 140)), "Rift Valley": (.16, (70, 320)),
    "Western": (.10, (280, 420)), "Nyanza": (.10, (260, 400)), "Coast": (.12, (420, 530)),
}
ch_names, rg_names = list(CHANNELS), list(REGIONS)
outlets = pd.DataFrame({
    "outlet_id": [f"O{i:05d}" for i in range(1, N_OUTLETS + 1)],
    "channel": RNG.choice(ch_names, N_OUTLETS, p=[CHANNELS[c][0] for c in ch_names]),
    "region": RNG.choice(rg_names, N_OUTLETS, p=[REGIONS[r][0] for r in rg_names]),
})
lo = outlets.region.map(lambda r: REGIONS[r][1][0]).values
hi = outlets.region.map(lambda r: REGIONS[r][1][1]).values
outlets["distance_km"] = np.round(lo + (hi - lo) * RNG.beta(1.6, 1.6, N_OUTLETS)).astype(int)
p_merch = outlets.channel.map({"Supermarket": .80, "Wholesaler": .45, "Duka": .20,
                               "HoReCa": .30, "Petrol station": .45}).values
p_merch = p_merch * np.where(outlets.distance_km > 250, .6, 1.0)
outlets["has_merchandiser"] = (RNG.random(N_OUTLETS) < p_merch).astype(int)
# Outlet closures: ~6%/year for dukas, lower for formal trade. Closure date drawn from 2024 onward.
close_rate_yr = outlets.channel.map({"Supermarket": .02, "Wholesaler": .03, "Duka": .07,
                                     "HoReCa": .06, "Petrol station": .03}).values
years_to_close = RNG.exponential(1 / close_rate_yr)
outlets["closure_date"] = pd.Timestamp("2024-03-01") + pd.to_timedelta(np.round(years_to_close * 365.25), unit="D")
outlets.loc[outlets.closure_date > CUTOFF, "closure_date"] = pd.NaT

# ------------------------------------------------------------------------------------
# SKU launches
# ------------------------------------------------------------------------------------
SKUS = [  # sku name, category, shelf life (days), base price index, n listings target, launch date
    ("Greek-style yoghurt 450g",         "Yoghurt",          28, 104, 520, "2024-03-04"),
    ("Mature cheddar 200g",              "Cheese",          180, 108, 430, "2024-04-15"),
    ("Vanilla bean ice cream 1L",        "Ice cream",       365, 102, 470, "2024-06-03"),
    ("Drinking yoghurt mango 250ml",     "Yoghurt",          21,  97, 610, "2024-07-22"),
    ("Beef breakfast sausages 500g",     "Deli",             30,  99, 480, "2024-09-02"),
    ("Halloumi 250g",                    "Cheese",          120, 112, 360, "2024-10-14"),
    ("Salted butter 250g",               "Butter & cream",  120,  98, 520, "2024-12-02"),
    ("Strawberry milk 500ml",            "Flavoured milk",   21,  96, 560, "2025-01-27"),
    ("Choc-chip ice cream 500ml",        "Ice cream",       365, 105, 430, "2025-03-17"),
    ("Smoked chicken slices 150g",       "Deli",             21, 107, 390, "2025-05-05"),
    ("Feta 200g",                        "Cheese",           90, 103, 400, "2025-06-23"),
    ("Natural set yoghurt 1kg",          "Yoghurt",          35,  95, 540, "2025-08-11"),
    ("Fresh cream 250ml",                "Butter & cream",   14, 101, 420, "2025-10-06"),
    ("Fruit ice lolly 6-pack",           "Ice cream",       365,  99, 450, "2025-11-24"),
    ("Mozzarella 500g (food service)",   "Cheese",           60, 100, 380, "2026-02-09"),
]
CATEGORY_LOGHR = {"Yoghurt": 0.0, "Cheese": -0.15, "Ice cream": 0.35, "Deli": 0.15,
                  "Butter & cream": -0.25, "Flavoured milk": 0.05}
sku_appeal = RNG.normal(0, .22, len(SKUS))            # unobserved SKU frailty (log-HR scale)
sku_promo_budget = RNG.uniform(-.6, .6, len(SKUS))     # some launches had bigger promo budgets

rows = []
for k, (name, cat, shelf, price_base, n_target, launch) in enumerate(SKUS):
    launch = pd.Timestamp(launch)
    # outlets eligible: open at launch; listing propensity by channel (+ HoReCa loves food-service packs)
    open_at = outlets.closure_date.isna() | (outlets.closure_date > launch + pd.Timedelta(weeks=6))
    w = outlets.channel.map(lambda c: CHANNELS[c][1]).values * open_at.values
    if "food service" in name:
        w = w * np.where(outlets.channel == "HoReCa", 4, 1)
    n = int(RNG.normal(n_target, 25))
    idx = RNG.choice(N_OUTLETS, n, replace=False, p=w / w.sum())
    o = outlets.iloc[idx].reset_index(drop=True)
    d = pd.DataFrame({"sku_id": f"SKU{k + 1:02d}", "sku_name": name, "category": cat,
                      "launch_date": launch, "shelf_life_days": shelf}, index=range(n))
    d = pd.concat([d, o], axis=1)
    # roll-out: listings go live 0-14 weeks after launch (formal trade first)
    delay = RNG.gamma(1.5, 14, n) * np.where(d.channel.isin(["Supermarket", "Wholesaler"]), .6, 1.2)
    d["listing_date"] = launch + pd.to_timedelta(np.round(np.minimum(delay, 98)), unit="D")
    # launch support
    lp_promo = np.log(np.array([CHANNELS[c][2] for c in d.channel]) / (1 - np.array([CHANNELS[c][2] for c in d.channel])))
    d["trade_promo"] = (RNG.random(n) < 1 / (1 + np.exp(-(lp_promo + sku_promo_budget[k])))).astype(int)
    p_pos = np.array([CHANNELS[c][3] for c in d.channel]) + .20 * d.has_merchandiser.values
    d["pos_material"] = (RNG.random(n) < np.clip(p_pos, 0, .95)).astype(int)
    # commercial terms
    d["price_index"] = np.round(price_base + RNG.normal(0, 6, n)
                                + np.where(d.channel == "Duka", 3, 0) + np.where(d.channel == "Wholesaler", -4, 0), 1)
    d["retailer_margin_pct"] = np.round(np.clip(np.array([CHANNELS[c][5] for c in d.channel])
                                                + RNG.normal(0, 3, n), 4, 35), 1)
    # early sell-through: % of the opening stock sold in the first 4 weeks (a mediator of support)
    z = (0.30 + 0.55 * d.trade_promo + 0.35 * d.pos_material + 0.30 * d.has_merchandiser
         - 0.045 * (d.price_index - 100) - 2.2 * sku_appeal[k]
         + d.channel.map({"Supermarket": .3, "Wholesaler": .4, "Duka": -.2, "HoReCa": 0, "Petrol station": -.3}).values
         + RNG.normal(0, 1.0, n))
    d["sellthrough_4wk_pct"] = np.round(100 / (1 + np.exp(-(z - .4) * 1.1)), 1)
    d["_appeal"] = sku_appeal[k]
    rows.append(d)

L = pd.concat(rows, ignore_index=True)
L = L[L.listing_date <= CUTOFF - pd.Timedelta(weeks=MIN_LISTING_WEEKS + 1)].reset_index(drop=True)
L = L[L.closure_date.isna() | (L.closure_date > L.listing_date + pd.Timedelta(weeks=MIN_LISTING_WEEKS))].reset_index(drop=True)
n = len(L)

# ------------------------------------------------------------------------------------
# Delisting times: simulate a weekly hazard week by week (piecewise-constant hazard)
# ------------------------------------------------------------------------------------
lp_fixed = (L.channel.map(lambda c: CHANNELS[c][4]).values
            + L.category.map(CATEGORY_LOGHR).values
            - 0.032 * (L.sellthrough_4wk_pct.values - 50)              # very strong: HR ~0.73 per +10pp
            - 0.40 * L.pos_material.values
            - 0.40 * L.has_merchandiser.values
            + 0.026 * (L.price_index.values - 100)                     # HR ~1.30 per +10 index points
            - 0.035 * (L.retailer_margin_pct.values - 16)              # HR ~0.97 per +1pp margin
            + 0.0015 * L.distance_km.values                            # HR ~1.16 per +100 km
            - 0.20 * np.log2(L.shelf_life_days.values / 30)            # longer shelf life protects
            + L._appeal.values)
promo = L.trade_promo.values
MAX_W = int((CUTOFF - L.listing_date.min()).days / 7) + 2
a, b, scale = 18.0, 1.9, 0.95
event_week = np.full(n, np.inf)
alive = np.ones(n, bool)
for t in range(MIN_LISTING_WEEKS, MAX_W):
    base = scale * (b / a) * (t / a) ** (b - 1) / (1 + (t / a) ** b)
    promo_eff = np.where(promo == 1, -1.10 if t < PROMO_WEEKS else 0.15, 0.0)
    h = base * np.exp(lp_fixed + promo_eff - 0.9)
    hit = alive & (RNG.random(n) < 1 - np.exp(-h))
    event_week[hit] = t
    alive &= ~hit
event_day = event_week * 7 + RNG.integers(0, 7, n)          # day within the week

cutoff_day = (CUTOFF - L.listing_date).dt.days.values
closure_day = np.where(L.closure_date.isna(), np.inf, (L.closure_date - L.listing_date).dt.days.values)
exit_day = np.minimum.reduce([event_day, cutoff_day, closure_day]).astype(int)
L["delisted"] = (event_day <= np.minimum(cutoff_day, closure_day)).astype(int)
L["status"] = np.select([L.delisted == 1, closure_day <= cutoff_day],
                        ["Delisted", "Censored: outlet closed"], "Censored: still listed at cut-off")
L["exit_date"] = L.listing_date + pd.to_timedelta(exit_day, unit="D")
L["days_listed"] = exit_day
L["weeks_listed"] = np.round(exit_day / 7, 2)

L = L.sort_values(["listing_date", "sku_id", "outlet_id"]).reset_index(drop=True)
L.insert(0, "listing_id", [f"L{i:05d}" for i in range(1, len(L) + 1)])
cols = ["listing_id", "sku_id", "sku_name", "category", "launch_date", "listing_date", "outlet_id", "channel",
        "region", "distance_km", "has_merchandiser", "trade_promo", "pos_material", "price_index",
        "retailer_margin_pct", "shelf_life_days", "sellthrough_4wk_pct", "exit_date", "days_listed",
        "weeks_listed", "delisted", "status"]
for c in ["launch_date", "listing_date", "exit_date"]:
    L[c] = L[c].dt.strftime("%Y-%m-%d")
L[cols].to_csv(OUT / "sku_listings.csv", index=False)

print(f"{len(L):,} listings | {L.outlet_id.nunique():,} outlets | {L.sku_id.nunique()} SKUs | "
      f"listing dates {L.listing_date.min()} to {L.listing_date.max()} | cut-off {CUTOFF:%Y-%m-%d}")
print(L.status.value_counts().to_string())
print(f"Max follow-up {L.weeks_listed.max():.0f} weeks; median {L.weeks_listed.median():.0f} weeks")
