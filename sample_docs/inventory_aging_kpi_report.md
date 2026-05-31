# Inventory Aging KPI Report

Report owner: Supply Chain Analytics

Inventory aging measures the number of days inventory has remained on hand since the latest receipt date. It helps identify slow-moving stock, excess working capital, and items at risk of obsolescence.

KPI definition:
- Aging days = report date minus latest receipt date.
- Average inventory aging = average aging days across included SKUs.
- Aged inventory value = inventory value for SKUs with aging days greater than 90.

Interpretation guidance:
- 0 to 30 days: healthy flow for most fast-moving SKUs.
- 31 to 60 days: monitor demand and replenishment assumptions.
- 61 to 90 days: review promotion, transfer, or markdown options.
- More than 90 days: create an action plan with branch operations and demand planning.

The KPI should be reviewed weekly by branch operations leads. Finance uses aged inventory value to estimate working-capital exposure.

Known limitations: the metric does not explain why inventory is aging. Analysts should compare aging with demand forecast, stockout history, and open customer orders before recommending action.
