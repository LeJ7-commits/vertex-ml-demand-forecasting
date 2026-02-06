-- Feature table for demand forecasting
-- Source: demand_fcst.raw_demand
-- Output: demand_fcst.features_demand_daily
--
-- Keys: (store_id=ship-state, item_id=SKU, date)
-- Target: y = daily SUM(Qty)

CREATE OR REPLACE TABLE `vertex-demand-260203-13078.demand_fcst.features_demand_daily`
PARTITION BY date
CLUSTER BY store_id, item_id AS
WITH base AS (
  SELECT
    -- Date may be STRING or DATE depending on autodetect; handle both safely:
    COALESCE(
      SAFE_CAST(`Date` AS DATE),
      SAFE.PARSE_DATE('%m/%d/%Y', CAST(`Date` AS STRING))
    ) AS date,

    COALESCE(NULLIF(TRIM(CAST(`ship-state` AS STRING)), ''), 'UNKNOWN') AS store_id,
    COALESCE(NULLIF(TRIM(CAST(`SKU` AS STRING)), ''), 'UNKNOWN') AS item_id,
    SAFE_CAST(`Qty` AS FLOAT64) AS qty
  FROM `vertex-demand-260203-13078.demand_fcst.raw_demand`
  WHERE `Date` IS NOT NULL
),
daily AS (
  SELECT
    date,
    store_id,
    item_id,
    SUM(COALESCE(qty, 0)) AS y
  FROM base
  WHERE date IS NOT NULL
  GROUP BY 1,2,3
),
bounds AS (
  SELECT
    store_id,
    item_id,
    MIN(date) AS min_date,
    MAX(date) AS max_date
  FROM daily
  GROUP BY 1,2
),
spine AS (
  SELECT
    b.store_id,
    b.item_id,
    d AS date
  FROM bounds b,
  UNNEST(GENERATE_DATE_ARRAY(b.min_date, b.max_date)) AS d
),
series AS (
  SELECT
    s.date,
    s.store_id,
    s.item_id,
    COALESCE(d.y, 0) AS y
  FROM spine s
  LEFT JOIN daily d
    USING (date, store_id, item_id)
),
calendar AS (
  SELECT
    date,
    store_id,
    item_id,
    y,
    EXTRACT(DAYOFWEEK FROM date) AS dow,
    EXTRACT(WEEK FROM date)      AS week,
    EXTRACT(MONTH FROM date)     AS month,
    EXTRACT(QUARTER FROM date)   AS quarter,
    EXTRACT(YEAR FROM date)      AS year,
    SIN(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM date) / 365.0) AS sin_doy,
    COS(2 * ACOS(-1) * EXTRACT(DAYOFYEAR FROM date) / 365.0) AS cos_doy
  FROM series
),
lags AS (
  SELECT
    *,
    LAG(y, 1)  OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_1,
    LAG(y, 7)  OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_7,
    LAG(y, 14) OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_14,
    LAG(y, 28) OVER (PARTITION BY store_id, item_id ORDER BY date) AS lag_28
  FROM calendar
),
rollups AS (
  SELECT
    *,
    AVG(y) OVER (
      PARTITION BY store_id, item_id ORDER BY date
      ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_7,
    AVG(y) OVER (
      PARTITION BY store_id, item_id ORDER BY date
      ROWS BETWEEN 14 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_14,
    AVG(y) OVER (
      PARTITION BY store_id, item_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_mean_28,
    STDDEV_SAMP(y) OVER (
      PARTITION BY store_id, item_id ORDER BY date
      ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING
    ) AS roll_std_28
  FROM lags
)
SELECT
  date,
  store_id,
  item_id,
  y,
  dow, week, month, quarter, year,
  sin_doy, cos_doy,
  lag_1, lag_7, lag_14, lag_28,
  roll_mean_7, roll_mean_14, roll_mean_28,
  roll_std_28
FROM rollups
WHERE lag_28 IS NOT NULL;
