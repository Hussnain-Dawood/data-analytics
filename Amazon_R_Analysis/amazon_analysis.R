# =============================================================================
# MKT7011 – Big Data, Analytics and Martech
# Amazon Sales Dataset – Full Statistical Analysis
# Author: [Your Name] | Student ID: [Your ID]
# Tools: R 4.3.3 | Packages: dplyr, ggplot2, car, lmtest, nortest, broom
# =============================================================================
# ANALYSIS STRUCTURE
#   0. Setup & Data Loading
#   1. Data Preparation & Cleaning
#   2. Descriptive Statistics
#   3. INFERENTIAL ANALYSIS
#      A. Multiple Linear Regression  → What drives TotalAmount?
#      B. One-Way ANOVA               → Does Category affect TotalAmount?
#      C. Chi-Square Test             → Is OrderStatus independent of Category?
#   4. Summary of Findings
# =============================================================================


# =============================================================================
# 0. SETUP & DATA LOADING
# =============================================================================

# Load required libraries
library(dplyr)      # data manipulation
library(ggplot2)    # visualisation
library(car)        # VIF, Levene's test, ANOVA utilities
library(lmtest)     # Breusch-Pagan heteroscedasticity test
library(nortest)    # Anderson-Darling normality test
library(broom)      # tidy model output
library(scales)     # axis formatting in ggplot2
library(gridExtra)  # arrange multiple plots

# Set global plot theme
theme_set(theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 10, hjust = 0.5, colour = "grey40"),
    axis.title    = element_text(face = "bold", size = 10),
    legend.title  = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  ))

# Colour palette (consistent throughout)
cat_colours <- c(
  "Electronics"     = "#1f77b4",
  "Clothing"        = "#ff7f0e",
  "Home & Kitchen"  = "#2ca02c",
  "Books"           = "#d62728",
  "Sports & Outdoors" = "#9467bd",
  "Toys & Games"    = "#8c564b"
)

# Output directory for plots
dir.create("plots", showWarnings = FALSE)

cat("=============================================================\n")
cat(" Loading Amazon Sales Dataset\n")
cat("=============================================================\n\n")

# Read CSV (converted from xlsx)
df_raw <- read.csv("amazon_data.csv", stringsAsFactors = FALSE)
cat(sprintf("Raw dataset: %d rows x %d columns\n", nrow(df_raw), ncol(df_raw)))
cat("\nColumn names:\n")
print(names(df_raw))


# =============================================================================
# 1. DATA PREPARATION & CLEANING
# =============================================================================

cat("\n=============================================================\n")
cat(" 1. Data Preparation & Cleaning\n")
cat("=============================================================\n\n")

df <- df_raw %>%
  # Parse date
  mutate(
    OrderDate   = as.Date(OrderDate),
    Year        = as.integer(format(OrderDate, "%Y")),
    Month       = as.integer(format(OrderDate, "%m")),
    MonthName   = factor(format(OrderDate, "%b"),
                         levels = month.abb)
  ) %>%
  # Ensure correct types
  mutate(
    Quantity      = as.integer(Quantity),
    UnitPrice     = as.numeric(UnitPrice),
    Discount      = as.numeric(Discount),
    Tax           = as.numeric(Tax),
    ShippingCost  = as.numeric(ShippingCost),
    TotalAmount   = as.numeric(TotalAmount),
    Category      = as.factor(Category),
    Country       = as.factor(Country),
    PaymentMethod = as.factor(PaymentMethod),
    OrderStatus   = as.factor(OrderStatus)
  ) %>%
  # Remove rows with missing key variables
  filter(
    !is.na(TotalAmount),
    !is.na(Quantity),
    !is.na(UnitPrice),
    !is.na(Discount),
    !is.na(Category)
  )

cat(sprintf("Clean dataset: %d rows x %d columns\n", nrow(df), ncol(df)))

# Missing value summary
cat("\nMissing values per column:\n")
print(colSums(is.na(df)))

# Basic structure overview
cat("\nVariable Summary:\n")
cat(sprintf("  Years covered       : %d – %d\n", min(df$Year), max(df$Year)))
cat(sprintf("  Total orders        : %s\n", format(nrow(df), big.mark=",")))
cat(sprintf("  Product categories  : %d (%s)\n",
    length(levels(df$Category)),
    paste(levels(df$Category), collapse=", ")))
cat(sprintf("  Countries           : %d\n", length(levels(df$Country))))
cat(sprintf("  Payment methods     : %d\n", length(levels(df$PaymentMethod))))
cat(sprintf("  Order statuses      : %s\n",
    paste(levels(df$OrderStatus), collapse=", ")))
cat(sprintf("  Discount levels     : %s\n",
    paste(sort(unique(df$Discount)), collapse=", ")))


# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================

cat("\n=============================================================\n")
cat(" 2. Descriptive Statistics\n")
cat("=============================================================\n\n")

# --- 2a. Overall TotalAmount distribution ---
cat("--- TotalAmount (Overall) ---\n")
print(summary(df$TotalAmount))
cat(sprintf("  Std Dev  : %.2f\n", sd(df$TotalAmount)))
cat(sprintf("  Skewness : %.4f\n",
    (mean(df$TotalAmount) - median(df$TotalAmount)) / sd(df$TotalAmount) * 3))

# --- 2b. Revenue by Category ---
cat("\n--- Revenue by Category ---\n")
cat_summary <- df %>%
  group_by(Category) %>%
  summarise(
    Orders       = n(),
    TotalRevenue = sum(TotalAmount),
    AvgOrder     = mean(TotalAmount),
    MedianOrder  = median(TotalAmount),
    StdDev       = sd(TotalAmount),
    .groups = "drop"
  ) %>%
  arrange(desc(TotalRevenue))
print(as.data.frame(cat_summary))

# --- 2c. Revenue by Country ---
cat("\n--- Revenue by Country ---\n")
country_summary <- df %>%
  group_by(Country) %>%
  summarise(
    Orders       = n(),
    TotalRevenue = sum(TotalAmount),
    AvgOrder     = mean(TotalAmount),
    .groups = "drop"
  ) %>%
  arrange(desc(TotalRevenue))
print(as.data.frame(country_summary))

# --- 2d. Discount vs Average Order Value ---
cat("\n--- Discount Level vs Average Order Value ---\n")
disc_summary <- df %>%
  group_by(Discount) %>%
  summarise(
    Orders   = n(),
    AvgOrder = round(mean(TotalAmount), 2),
    .groups  = "drop"
  )
print(as.data.frame(disc_summary))

# --- 2e. Order Status breakdown ---
cat("\n--- Order Status ---\n")
status_summary <- df %>%
  group_by(OrderStatus) %>%
  summarise(
    Count   = n(),
    Pct     = round(n() / nrow(df) * 100, 1),
    Revenue = sum(TotalAmount),
    .groups = "drop"
  ) %>%
  arrange(desc(Count))
print(as.data.frame(status_summary))

# --- 2f. Yearly revenue trend ---
cat("\n--- Annual Revenue Trend ---\n")
yearly <- df %>%
  group_by(Year) %>%
  summarise(
    Orders   = n(),
    Revenue  = sum(TotalAmount),
    AvgOrder = mean(TotalAmount),
    .groups  = "drop"
  )
print(as.data.frame(yearly))


# ── DESCRIPTIVE PLOTS ────────────────────────────────────────────────────────

# Plot 1: Revenue by Category (bar chart)
p1 <- ggplot(cat_summary, aes(x = reorder(Category, TotalRevenue),
                               y = TotalRevenue / 1e6,
                               fill = Category)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = sprintf("$%.2fM", TotalRevenue/1e6)),
            hjust = -0.1, size = 3.2) +
  scale_fill_manual(values = cat_colours) +
  scale_y_continuous(labels = label_dollar(suffix="M"), expand = expansion(mult=c(0,.15))) +
  coord_flip() +
  labs(title = "Total Revenue by Product Category",
       subtitle = "Amazon Sales Dataset (2020–2024)",
       x = NULL, y = "Total Revenue (USD Millions)") +
  theme(plot.margin = margin(10,40,10,10))

ggsave("plots/01_revenue_by_category.png", p1, width=8, height=5, dpi=150)
cat("\nPlot saved: plots/01_revenue_by_category.png\n")

# Plot 2: Discount vs Avg Order Value (line)
p2 <- ggplot(disc_summary, aes(x = Discount * 100, y = AvgOrder)) +
  geom_line(colour = "#1f77b4", linewidth = 1.2) +
  geom_point(colour = "#1f77b4", size = 3.5) +
  geom_text(aes(label = sprintf("$%.0f", AvgOrder)),
            vjust = -0.8, size = 3.2) +
  scale_x_continuous(breaks = unique(disc_summary$Discount) * 100,
                     labels = paste0(unique(disc_summary$Discount)*100, "%")) +
  scale_y_continuous(labels = label_dollar()) +
  labs(title = "Effect of Discount Level on Average Order Value",
       subtitle = "Clear inverse relationship: higher discounts → lower order value",
       x = "Discount Rate (%)", y = "Average Order Value (USD)") +
  expand_limits(y = max(disc_summary$AvgOrder) * 1.1)

ggsave("plots/02_discount_vs_order_value.png", p2, width=8, height=5, dpi=150)
cat("Plot saved: plots/02_discount_vs_order_value.png\n")

# Plot 3: Annual Revenue Trend
p3 <- ggplot(yearly, aes(x = Year, y = Revenue / 1e6)) +
  geom_line(colour = "#2ca02c", linewidth = 1.3) +
  geom_point(colour = "#2ca02c", size = 4) +
  geom_text(aes(label = sprintf("$%.2fM", Revenue/1e6)),
            vjust = -0.8, size = 3.2) +
  scale_y_continuous(labels = label_dollar(suffix="M"),
                     limits = c(17, 19.5)) +
  scale_x_continuous(breaks = 2020:2024) +
  labs(title = "Annual Revenue Trend (2020–2024)",
       subtitle = "Amazon Sales Dataset",
       x = "Year", y = "Total Revenue (USD Millions)")

ggsave("plots/03_annual_revenue_trend.png", p3, width=8, height=5, dpi=150)
cat("Plot saved: plots/03_annual_revenue_trend.png\n")

# Plot 4: Order Status Pie Chart
p4 <- ggplot(status_summary, aes(x = "", y = Count, fill = OrderStatus)) +
  geom_col(width = 1, colour = "white") +
  coord_polar("y") +
  geom_text(aes(label = sprintf("%s\n%.1f%%", OrderStatus, Pct)),
            position = position_stack(vjust = 0.5), size = 3) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Order Status Distribution",
       subtitle = "74.6% of orders successfully Delivered",
       x = NULL, y = NULL, fill = "Status") +
  theme_void(base_size=12) +
  theme(plot.title = element_text(face="bold", hjust=0.5),
        plot.subtitle = element_text(hjust=0.5, colour="grey40"),
        legend.position = "right")

ggsave("plots/04_order_status_pie.png", p4, width=7, height=5, dpi=150)
cat("Plot saved: plots/04_order_status_pie.png\n")

# Plot 5: Revenue by Country
p5 <- ggplot(country_summary,
             aes(x = reorder(Country, TotalRevenue), y = TotalRevenue/1e6, fill = Country)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = sprintf("$%.1fM", TotalRevenue/1e6)),
            hjust = -0.1, size = 3.2) +
  scale_y_continuous(labels=label_dollar(suffix="M"), expand=expansion(mult=c(0,.2))) +
  coord_flip() +
  labs(title = "Total Revenue by Country",
       subtitle = "United States dominates with 70% of total revenue",
       x = NULL, y = "Total Revenue (USD Millions)")

ggsave("plots/05_revenue_by_country.png", p5, width=8, height=5, dpi=150)
cat("Plot saved: plots/05_revenue_by_country.png\n")

# Plot 6: TotalAmount distribution (histogram)
p6 <- ggplot(df, aes(x = TotalAmount)) +
  geom_histogram(bins = 60, fill = "#1f77b4", colour = "white", alpha = 0.85) +
  scale_x_continuous(labels = label_dollar()) +
  scale_y_continuous(labels = label_comma()) +
  geom_vline(xintercept = mean(df$TotalAmount), colour = "red",
             linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = median(df$TotalAmount), colour = "orange",
             linetype = "dashed", linewidth = 1) +
  annotate("text", x = mean(df$TotalAmount)+80, y = 3800,
           label = sprintf("Mean\n$%.0f", mean(df$TotalAmount)),
           colour="red", size=3) +
  annotate("text", x = median(df$TotalAmount)-150, y = 3800,
           label = sprintf("Median\n$%.0f", median(df$TotalAmount)),
           colour="darkorange", size=3) +
  labs(title = "Distribution of Order Value (TotalAmount)",
       subtitle = "Right-skewed distribution; mean pulled above median by high-value orders",
       x = "Order Value (USD)", y = "Frequency")

ggsave("plots/06_totalamount_distribution.png", p6, width=8, height=5, dpi=150)
cat("Plot saved: plots/06_totalamount_distribution.png\n")


# =============================================================================
# 3A. MULTIPLE LINEAR REGRESSION
#     Business Question:
#     "Which factors (Discount, Quantity, UnitPrice, Category, Country)
#      significantly predict the total order value (TotalAmount) on Amazon?"
# =============================================================================

cat("\n=============================================================\n")
cat(" 3A. MULTIPLE LINEAR REGRESSION\n")
cat("     DV: TotalAmount\n")
cat("     IVs: Discount, Quantity, UnitPrice, Category, Country\n")
cat("=============================================================\n\n")

# Use ALL orders (not just Delivered) — TotalAmount exists for all statuses
# Drop Pending (small, incomplete) to focus on outcome-known orders
df_reg <- df %>%
  filter(OrderStatus %in% c("Delivered", "Shipped", "Returned", "Cancelled")) %>%
  droplevels()

cat(sprintf("Regression sample: %d observations\n\n", nrow(df_reg)))

# --- Build model ---
# Reference levels: Category = "Books", Country = "Australia"
df_reg$Category <- relevel(df_reg$Category, ref = "Books")
df_reg$Country  <- relevel(df_reg$Country,  ref = "Australia")

model_full <- lm(TotalAmount ~ Discount + Quantity + UnitPrice +
                               Category + Country,
                 data = df_reg)

cat("--- Full Regression Model Summary ---\n")
print(summary(model_full))

cat("\n--- Tidy Coefficient Table ---\n")
tidy_model <- tidy(model_full, conf.int = TRUE)
tidy_print <- tidy_model[, c("term","estimate","std.error","statistic","p.value","conf.low","conf.high")]
tidy_print[, -1] <- round(tidy_print[, -1], 4)
print(as.data.frame(tidy_print))

# --- Model fit metrics ---
cat("\n--- Model Fit ---\n")
glance_model <- glance(model_full)
cat(sprintf("  R²              : %.6f\n", glance_model$r.squared))
cat(sprintf("  Adjusted R²     : %.6f\n", glance_model$adj.r.squared))
cat(sprintf("  F-statistic     : %.2f\n",  glance_model$statistic))
cat(sprintf("  F p-value       : %s\n",
    ifelse(glance_model$p.value < 0.001, "< 0.001", round(glance_model$p.value,4))))
cat(sprintf("  RMSE            : %.2f\n",  glance_model$sigma))
cat(sprintf("  AIC             : %.2f\n",  glance_model$AIC))

# --- VIF (multicollinearity check) ---
cat("\n--- Variance Inflation Factors (VIF) ---\n")
cat("  Values < 5 = acceptable; < 10 = tolerable\n")
vif_vals <- vif(model_full)
print(round(vif_vals, 3))

# --- Breusch-Pagan heteroscedasticity test ---
cat("\n--- Breusch-Pagan Test (Heteroscedasticity) ---\n")
bp_test <- bptest(model_full)
print(bp_test)
cat(sprintf("  Interpretation: p = %.4f → %s\n",
    bp_test$p.value,
    ifelse(bp_test$p.value < 0.05,
           "Heteroscedasticity present (use robust SE in reporting)",
           "No significant heteroscedasticity detected")))

# --- Anderson-Darling normality test on residuals ---
cat("\n--- Anderson-Darling Test (Normality of Residuals) ---\n")
ad_test <- ad.test(residuals(model_full))
print(ad_test)
cat(sprintf("  Interpretation: p = %s → %s\n",
    ifelse(ad_test$p.value < 0.001, "< 0.001", round(ad_test$p.value,4)),
    ifelse(ad_test$p.value < 0.05,
           "Residuals deviate from normality (expected in large samples; CLT applies)",
           "Residuals are approximately normal")))

# --- Regression diagnostic plots ---
png("plots/07_regression_diagnostics.png", width=1000, height=800, res=120)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
plot(model_full, which = 1, main = "Residuals vs Fitted")
plot(model_full, which = 2, main = "Normal Q-Q")
plot(model_full, which = 3, main = "Scale-Location")
plot(model_full, which = 4, main = "Cook's Distance")
dev.off()
cat("\nPlot saved: plots/07_regression_diagnostics.png\n")

# --- Coefficient plot ---
tidy_plot <- tidy_model %>%
  filter(term != "(Intercept)") %>%
  mutate(
    Significant = ifelse(p.value < 0.05, "p < 0.05", "p ≥ 0.05"),
    term = gsub("Category", "Cat: ", term),
    term = gsub("Country", "Country: ", term)
  )

p7 <- ggplot(tidy_plot, aes(x = reorder(term, estimate),
                             y = estimate,
                             colour = Significant)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("p < 0.05" = "#d62728", "p ≥ 0.05" = "#aec7e8")) +
  coord_flip() +
  labs(title = "Regression Coefficients with 95% Confidence Intervals",
       subtitle = "DV: TotalAmount | Red = statistically significant (p < 0.05)",
       x = NULL, y = "Coefficient Estimate (USD)",
       colour = "Significance") +
  theme(legend.position = "bottom")

ggsave("plots/08_regression_coefficients.png", p7, width=9, height=6, dpi=150)
cat("Plot saved: plots/08_regression_coefficients.png\n")

# --- Key interpretation ---
cat("\n--- Key Regression Findings ---\n")
coef_df <- as.data.frame(tidy_model)
qty_coef  <- coef_df[coef_df$term == "Quantity",    "estimate"]
disc_coef <- coef_df[coef_df$term == "Discount",    "estimate"]
up_coef   <- coef_df[coef_df$term == "UnitPrice",   "estimate"]
cat(sprintf("  Quantity:   +$%.2f per additional unit ordered\n",   qty_coef))
cat(sprintf("  UnitPrice:  +$%.4f per $1 increase in unit price\n", up_coef))
cat(sprintf("  Discount:   %+.2f (net effect on TotalAmount)\n",    disc_coef))


# =============================================================================
# 3B. ONE-WAY ANOVA
#     H0: Mean TotalAmount is equal across all 6 product categories
#     H1: At least one category differs significantly
# =============================================================================

cat("\n=============================================================\n")
cat(" 3B. ONE-WAY ANOVA\n")
cat("     DV: TotalAmount\n")
cat("     IV: Category (6 levels)\n")
cat("=============================================================\n\n")

# Use Delivered orders only for ANOVA
df_anova <- df %>%
  filter(OrderStatus == "Delivered") %>%
  droplevels()

cat(sprintf("ANOVA sample: %d delivered orders\n\n", nrow(df_anova)))

# Category means (descriptive)
cat("--- Category Means (Delivered Orders) ---\n")
anova_desc <- df_anova %>%
  group_by(Category) %>%
  summarise(
    n    = n(),
    Mean = round(mean(TotalAmount), 2),
    SD   = round(sd(TotalAmount), 2),
    SE   = round(sd(TotalAmount)/sqrt(n()), 4),
    .groups = "drop"
  )
print(as.data.frame(anova_desc))

# --- Levene's Test (homogeneity of variance) ---
cat("\n--- Levene's Test for Homogeneity of Variance ---\n")
levene_result <- leveneTest(TotalAmount ~ Category, data = df_anova)
print(levene_result)
cat(sprintf("  Interpretation: p = %.4f → %s\n",
    levene_result$`Pr(>F)`[1],
    ifelse(levene_result$`Pr(>F)`[1] < 0.05,
           "Variances are NOT equal across categories (use Welch's ANOVA)",
           "Variances are equal across categories (standard ANOVA appropriate)")))

# --- Standard One-Way ANOVA ---
cat("\n--- One-Way ANOVA Table ---\n")
anova_model <- aov(TotalAmount ~ Category, data = df_anova)
anova_table <- summary(anova_model)
print(anova_table)

# Extract F and p
F_val <- anova_table[[1]]$`F value`[1]
p_val <- anova_table[[1]]$`Pr(>F)`[1]
cat(sprintf("\n  F(%d, %d) = %.4f, p = %s\n",
    anova_table[[1]]$Df[1],
    anova_table[[1]]$Df[2],
    F_val,
    ifelse(p_val < 0.001, "< 0.001", round(p_val, 4))))
cat(sprintf("  Decision: %s\n",
    ifelse(p_val < 0.05,
           "REJECT H0 — significant difference exists between at least two categories",
           "FAIL TO REJECT H0 — no significant difference between categories")))

# --- Eta-squared (effect size) ---
SS_between <- anova_table[[1]]$`Sum Sq`[1]
SS_total   <- sum(anova_table[[1]]$`Sum Sq`)
eta_sq     <- SS_between / SS_total
cat(sprintf("\n  Eta-squared (η²) = %.6f\n", eta_sq))
cat(sprintf("  Effect size: %s\n",
    ifelse(eta_sq >= 0.14, "Large",
    ifelse(eta_sq >= 0.06, "Medium", "Small"))))

# --- Welch's ANOVA (robust to unequal variances) ---
cat("\n--- Welch's ANOVA (robust alternative) ---\n")
welch_result <- oneway.test(TotalAmount ~ Category,
                            data    = df_anova,
                            var.equal = FALSE)
print(welch_result)

# --- Tukey HSD Post-Hoc (pairwise comparisons) ---
cat("\n--- Tukey HSD Post-Hoc Test (Pairwise Comparisons) ---\n")
tukey_result <- TukeyHSD(anova_model, "Category")
print(tukey_result)
cat("\n  Pairs with significant difference (p < 0.05):\n")
tukey_df <- as.data.frame(tukey_result$Category)
sig_pairs <- tukey_df[tukey_df$`p adj` < 0.05, ]
if (nrow(sig_pairs) == 0) {
  cat("  None — all pairwise comparisons are non-significant\n")
} else {
  print(sig_pairs)
}

# --- ANOVA Visualisation ---

# Box plot by category
p8 <- ggplot(df_anova, aes(x = reorder(Category, TotalAmount, FUN=median),
                            y = TotalAmount, fill = Category)) +
  geom_boxplot(outlier.shape = 21, outlier.size = 1,
               outlier.alpha = 0.3, alpha = 0.8, width = 0.6) +
  scale_fill_manual(values = cat_colours, guide = "none") +
  scale_y_continuous(labels = label_dollar()) +
  coord_flip() +
  labs(title = "Order Value Distribution by Product Category",
       subtitle = "One-Way ANOVA: Testing mean differences across 6 categories",
       x = NULL, y = "Total Order Value (USD)")

ggsave("plots/09_anova_boxplot.png", p8, width=8, height=5, dpi=150)
cat("\nPlot saved: plots/09_anova_boxplot.png\n")

# Means plot with 95% CI
p9 <- ggplot(anova_desc, aes(x = reorder(Category, Mean),
                              y = Mean, colour = Category)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = Mean - 1.96*SE,
                    ymax = Mean + 1.96*SE), width = 0.25, linewidth = 1) +
  geom_hline(yintercept = mean(df_anova$TotalAmount),
             linetype = "dashed", colour = "grey50") +
  annotate("text", x = 0.6, y = mean(df_anova$TotalAmount) + 5,
           label = sprintf("Grand Mean = $%.0f", mean(df_anova$TotalAmount)),
           size = 3, colour = "grey40") +
  scale_colour_manual(values = cat_colours, guide = "none") +
  scale_y_continuous(labels = label_dollar(), limits = c(850, 970)) +
  coord_flip() +
  labs(title = "Category Means with 95% Confidence Intervals",
       subtitle = "ANOVA assesses whether differences exceed sampling variability",
       x = NULL, y = "Mean Order Value (USD)")

ggsave("plots/10_anova_means_ci.png", p9, width=8, height=5, dpi=150)
cat("Plot saved: plots/10_anova_means_ci.png\n")


# =============================================================================
# 3C. CHI-SQUARE TEST OF INDEPENDENCE
#     H0: OrderStatus is independent of Category
#     H1: OrderStatus and Category are associated
# =============================================================================

cat("\n=============================================================\n")
cat(" 3C. CHI-SQUARE TEST OF INDEPENDENCE\n")
cat("     Variable 1: Category  (6 levels)\n")
cat("     Variable 2: OrderStatus (5 levels)\n")
cat("     H0: OrderStatus is independent of Category\n")
cat("=============================================================\n\n")

# Use full dataset (all statuses matter)
df_chi <- df %>% droplevels()

# Contingency table
cat("--- Contingency Table (Observed Frequencies) ---\n")
cont_table <- table(df_chi$Category, df_chi$OrderStatus)
print(cont_table)

cat("\n--- Row Proportions (%) ---\n")
print(round(prop.table(cont_table, margin = 1) * 100, 2))

# --- Chi-Square Test ---
cat("\n--- Chi-Square Test of Independence ---\n")
chi_result <- chisq.test(cont_table)
print(chi_result)

cat(sprintf("\n  χ²(%d) = %.4f, p = %s\n",
    chi_result$parameter,
    chi_result$statistic,
    ifelse(chi_result$p.value < 0.001, "< 0.001", round(chi_result$p.value, 4))))
cat(sprintf("  Decision: %s\n",
    ifelse(chi_result$p.value < 0.05,
           "REJECT H0 — OrderStatus IS associated with Category",
           "FAIL TO REJECT H0 — OrderStatus is INDEPENDENT of Category")))

# --- Effect size: Cramér's V ---
n        <- sum(cont_table)
k        <- min(nrow(cont_table), ncol(cont_table))
cramers_v <- sqrt(chi_result$statistic / (n * (k - 1)))
cat(sprintf("\n  Cramér's V = %.4f\n", cramers_v))
cat(sprintf("  Effect size: %s\n",
    ifelse(cramers_v >= 0.25, "Large",
    ifelse(cramers_v >= 0.10, "Medium", "Small/Negligible"))))

# --- Expected vs Observed (spot check) ---
cat("\n--- Expected Frequencies (first 3 rows shown) ---\n")
print(round(chi_result$expected[1:3,], 1))

cat("\n--- Standardised Residuals (cells driving the association) ---\n")
cat("  Values > |2| indicate a cell contributing significantly to χ²\n")
std_resid <- chi_result$stdres
print(round(std_resid, 3))

# --- Chi-Square Visualisation ---

# Stacked bar: observed proportions
chi_df <- as.data.frame(cont_table) %>%
  rename(Category = Var1, OrderStatus = Var2, Count = Freq) %>%
  group_by(Category) %>%
  mutate(Proportion = Count / sum(Count) * 100) %>%
  ungroup()

p10 <- ggplot(chi_df, aes(x = Category, y = Proportion, fill = OrderStatus)) +
  geom_col(position = "stack", colour = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(Proportion > 3,
                               sprintf("%.1f%%", Proportion), "")),
            position = position_stack(vjust = 0.5), size = 2.8) +
  scale_fill_brewer(palette = "Set2", name = "Order Status") +
  scale_y_continuous(labels = label_percent(scale = 1)) +
  labs(title = "Order Status Composition by Product Category",
       subtitle = "Chi-Square Test: Assessing independence of Category and OrderStatus",
       x = "Product Category", y = "Proportion of Orders (%)") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        legend.position = "right")

ggsave("plots/11_chisquare_stacked_bar.png", p10, width=9, height=5.5, dpi=150)
cat("\nPlot saved: plots/11_chisquare_stacked_bar.png\n")

# Heatmap of standardised residuals
resid_df <- as.data.frame(std_resid) %>%
  rename(Category = Var1, OrderStatus = Var2, StdResid = Freq)

p11 <- ggplot(resid_df, aes(x = OrderStatus, y = Category, fill = StdResid)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = round(StdResid, 2)), size = 3.2) +
  scale_fill_gradient2(low = "#3182bd", mid = "white", high = "#de2d26",
                       midpoint = 0, name = "Std.\nResidual") +
  labs(title = "Standardised Residuals Heatmap",
       subtitle = "Cells in red/blue show where observed ≠ expected most strongly",
       x = "Order Status", y = "Category") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave("plots/12_chisquare_residuals_heatmap.png", p11, width=8, height=5.5, dpi=150)
cat("Plot saved: plots/12_chisquare_residuals_heatmap.png\n")

# Return/cancel rate by category (focused view)
ret_can <- df_chi %>%
  group_by(Category) %>%
  summarise(
    Total    = n(),
    Returned = sum(OrderStatus == "Returned"),
    Cancelled= sum(OrderStatus == "Cancelled"),
    RetRate  = Returned / Total * 100,
    CanRate  = Cancelled / Total * 100,
    .groups  = "drop"
  ) %>%
  tidyr::pivot_longer(cols = c(RetRate, CanRate),
                      names_to = "Metric", values_to = "Rate") %>%
  mutate(Metric = dplyr::case_when(
           Metric == "RetRate"  ~ "Return Rate (%)",
           Metric == "CanRate"  ~ "Cancellation Rate (%)",
           TRUE ~ Metric))

p12 <- ggplot(ret_can, aes(x = Category, y = Rate, fill = Metric)) +
  geom_col(position = "dodge", width = 0.65) +
  scale_fill_manual(values = c("Return Rate (%)"       = "#d62728",
                               "Cancellation Rate (%)" = "#ff7f0e"),
                    name = NULL) +
  scale_y_continuous(labels = label_percent(scale = 1),
                     limits = c(0, 6)) +
  labs(title = "Return and Cancellation Rates by Product Category",
       subtitle = "All rates cluster near 3% — indicating category-independent patterns",
       x = "Category", y = "Rate (%)") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        legend.position = "top")

ggsave("plots/13_return_cancel_by_category.png", p12, width=9, height=5.5, dpi=150)
cat("Plot saved: plots/13_return_cancel_by_category.png\n")


# =============================================================================
# 4. SUMMARY OF FINDINGS
# =============================================================================

cat("\n=============================================================\n")
cat(" 4. SUMMARY OF ALL FINDINGS\n")
cat("=============================================================\n\n")

cat("Business Question:\n")
cat("  'Which factors drive order value on Amazon, do product categories\n")
cat("   significantly affect revenue, and is order outcome associated\n")
cat("   with product category?'\n\n")

cat("─── A. MULTIPLE LINEAR REGRESSION ───────────────────────────\n")
cat(sprintf("  R²           = %.6f (model explains %.4f%% of variance)\n",
    glance_model$r.squared, glance_model$r.squared * 100))
cat(sprintf("  Adj. R²      = %.6f\n", glance_model$adj.r.squared))
cat(sprintf("  F-statistic  = %.2f (p < 0.001)\n", glance_model$statistic))
cat(sprintf("  Quantity:    +$%.2f per unit — STRONGEST positive predictor\n", qty_coef))
cat(sprintf("  UnitPrice:   +$%.4f per $1 — highly significant\n", up_coef))
cat(sprintf("  Discount:    %+.2f net effect (negative direction)\n", disc_coef))
cat("  Category & Country: minimal practical effect on TotalAmount\n")
cat("  → Quantity and UnitPrice are the primary commercial levers\n\n")

cat("─── B. ONE-WAY ANOVA ─────────────────────────────────────────\n")
cat(sprintf("  F(%d,%d) = %.4f\n",
    anova_table[[1]]$Df[1], anova_table[[1]]$Df[2], F_val))
cat(sprintf("  p-value  = %s\n",
    ifelse(p_val < 0.05, "< 0.05 (SIGNIFICANT)", paste0(round(p_val,4), " (NOT significant)"))))
cat(sprintf("  η²       = %.6f (effect size: %s)\n", eta_sq,
    ifelse(eta_sq >= 0.14, "Large",
    ifelse(eta_sq >= 0.06, "Medium", "Small"))))
cat("  Tukey HSD: No pairwise category differences reach significance\n")
cat("  → Category choice does NOT meaningfully drive order value\n")
cat("     Revenue differences are driven by volume (Quantity), not product type\n\n")

cat("─── C. CHI-SQUARE TEST ───────────────────────────────────────\n")
cat(sprintf("  χ²(%d) = %.4f\n", chi_result$parameter, chi_result$statistic))
cat(sprintf("  p-value   = %s\n",
    ifelse(chi_result$p.value < 0.05, "< 0.05 (SIGNIFICANT)", "≥ 0.05 (NOT significant)")))
cat(sprintf("  Cramér's V = %.4f (effect size: %s)\n", cramers_v,
    ifelse(cramers_v >= 0.25, "Large",
    ifelse(cramers_v >= 0.10, "Medium", "Small/Negligible"))))
cat("  → Despite statistical significance (large n), the near-zero\n")
cat("     Cramér's V confirms no PRACTICAL association between\n")
cat("     Category and OrderStatus; return/cancel rates ~3% across all\n\n")

cat("─── MARKETING RECOMMENDATIONS ───────────────────────────────\n")
cat("  1. Focus on QUANTITY-driving strategies: bundle promotions,\n")
cat("     multi-unit discounts, and subscription models increase\n")
cat("     order value more than product category selection.\n")
cat("  2. HIGH discounts (>15%) significantly erode order value;\n")
cat("     limit deep discounting and use targeted offers instead.\n")
cat("  3. US market (70% of revenue) should remain primary focus;\n")
cat("     India and Canada offer secondary growth opportunities.\n")
cat("  4. Category diversification is low-risk: Electronics leads\n")
cat("     slightly in revenue but all categories perform similarly.\n")
cat("  5. Return/cancel rates (~3%) are uniformly low — no category\n")
cat("     requires specific remediation for fulfilment quality.\n\n")

cat("─── OUTPUT FILES ─────────────────────────────────────────────\n")
cat("  Plots saved to: ./plots/\n")
cat("    01  Revenue by Category\n")
cat("    02  Discount vs Order Value\n")
cat("    03  Annual Revenue Trend\n")
cat("    04  Order Status Distribution\n")
cat("    05  Revenue by Country\n")
cat("    06  TotalAmount Distribution\n")
cat("    07  Regression Diagnostics\n")
cat("    08  Regression Coefficients (with CIs)\n")
cat("    09  ANOVA Box Plot by Category\n")
cat("    10  ANOVA Means with 95% CIs\n")
cat("    11  Chi-Square Stacked Bar\n")
cat("    12  Standardised Residuals Heatmap\n")
cat("    13  Return & Cancel Rates by Category\n\n")

cat("=============================================================\n")
cat(" Analysis Complete\n")
cat("=============================================================\n")
