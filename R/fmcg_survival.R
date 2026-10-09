# =====================================================================================
# New-SKU listing survival analysis in R (base R + the `survival` package)
# Author: Fordrane Albert Okumu
#
# PURPOSE
#   An independent rebuild of python/sku_listing_survival.ipynb with R's reference
#   survival-analysis package. If lifelines (Python) and survival (R) give the same
#   Kaplan-Meier estimates, hazard ratios and AFT coefficients from the same data and the
#   same design matrix, the analysis is right.
#
# STEPS
#   1. Load data, build the same design matrix (same units, same reference levels)
#   2. Kaplan-Meier (survfit): median, S(t) at 13/26/52/104 weeks, RMST(52)
#   3. Log-rank tests (survdiff)
#   4. Cox PH (coxph, ties = "efron" - lifelines also uses Efron's method for ties)
#   5. PH checks (cox.zph) and the two fixes: time split at week 16 (survSplit) and strata()
#   6. Parametric AFT models (survreg): Weibull, log-normal, log-logistic, compared by AIC
#   7. Cross-check against the Python outputs + auto-generated Markdown report
#
# Run from the repository root (after the notebook, for the cross-check):
#   Rscript R/fmcg_survival.R
# =====================================================================================

suppressPackageStartupMessages(library(survival))
out_dir <- file.path("outputs", "r"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
TEAL <- "#0f766e"; AMBER <- "#b45309"; GREY <- "#94a3b8"; RED <- "#b91c1c"
PROMO_WEEKS <- 16; TAU <- 52

# -------------------------------------------------------------------------------------
# 1. Data and design matrix (identical coding to design() in the notebook)
# -------------------------------------------------------------------------------------
df <- read.csv("data/sku_listings.csv", stringsAsFactors = FALSE)
cat(sprintf("%d listings | %d delisted | %d censored (still listed) | %d censored (outlet closed)\n",
            nrow(df), sum(df$delisted), sum(df$status == "Censored: still listed at cut-off"),
            sum(df$status == "Censored: outlet closed")))

CHANNELS   <- c("Wholesaler", "Duka", "HoReCa", "Petrol station")                 # reference: Supermarket
CATEGORIES <- c("Cheese", "Ice cream", "Deli", "Butter & cream", "Flavoured milk") # reference: Yoghurt
clean <- function(s) gsub(" ", "_", gsub(" & ", "_", s))

design <- function(d) {
  X <- data.frame(sellthrough_per10pp = d$sellthrough_4wk_pct / 10,
                  trade_promo = d$trade_promo, pos_material = d$pos_material,
                  has_merchandiser = d$has_merchandiser,
                  price_index_per10 = (d$price_index - 100) / 10,
                  retailer_margin_pp = d$retailer_margin_pct,
                  distance_per100km = d$distance_km / 100,
                  log2_shelf_life = log2(d$shelf_life_days))
  for (c in CHANNELS)   X[[paste0("channel_", clean(c))]]  <- as.integer(d$channel == c)
  for (c in CATEGORIES) X[[paste0("category_", clean(c))]] <- as.integer(d$category == c)
  X
}
D <- cbind(design(df), weeks_listed = df$weeks_listed, delisted = df$delisted)

# -------------------------------------------------------------------------------------
# 2. Kaplan-Meier
# -------------------------------------------------------------------------------------
km <- survfit(Surv(weeks_listed, delisted) ~ 1, data = df, conf.type = "log-log")
km_tab <- summary(km, times = c(13, 26, 52, 104))
km_med <- quantile(km, probs = .5)
rm_all <- summary(km, rmean = TAU)$table
cat(sprintf("KM median %.1f weeks (95%% CI %.1f-%.1f) | RMST(%d) = %.2f\n",
            km_med$quantile, km_med$lower, km_med$upper, TAU, rm_all["rmean"]))

km_promo <- survfit(Surv(weeks_listed, delisted) ~ trade_promo, data = df, conf.type = "log-log")
rm_promo <- summary(km_promo, rmean = TAU)$table
med_promo <- quantile(km_promo, probs = .5)$quantile

png(file.path(out_dir, "R_01_km_by_promo.png"), width = 1600, height = 1000, res = 170)
par(mar = c(4.5, 4.5, 3, 1))
plot(km_promo, col = c(GREY, TEAL), lwd = 2, conf.int = TRUE, xlab = "Weeks since listing",
     ylab = "Still listed  S(t)", main = "Kaplan-Meier by trade promotion (R, survfit)", xlim = c(0, 122))
abline(v = PROMO_WEEKS, lty = 3, col = "grey40"); abline(h = .5, lty = 3, col = "grey60")
legend("bottomleft", c("No promo", "Promo funded"), col = c(GREY, TEAL), lwd = 2, bty = "n")
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 3. Log-rank tests
# -------------------------------------------------------------------------------------
lr <- function(f) { s <- survdiff(f, data = df); c(chi2 = unname(s$chisq), df = length(s$n) - 1,
                                                   p = pchisq(s$chisq, length(s$n) - 1, lower.tail = FALSE)) }
logrank <- rbind(trade_promo = lr(Surv(weeks_listed, delisted) ~ trade_promo),
                 pos_material = lr(Surv(weeks_listed, delisted) ~ pos_material),
                 has_merchandiser = lr(Surv(weeks_listed, delisted) ~ has_merchandiser),
                 channel = lr(Surv(weeks_listed, delisted) ~ channel),
                 category = lr(Surv(weeks_listed, delisted) ~ category))

# -------------------------------------------------------------------------------------
# 4. Cox PH (Efron ties, same as lifelines)
# -------------------------------------------------------------------------------------
cox <- coxph(Surv(weeks_listed, delisted) ~ ., data = D, ties = "efron")
cs <- summary(cox)
hr <- data.frame(feature = rownames(cs$coefficients), coef = cs$coefficients[, "coef"],
                 hazard_ratio = cs$conf.int[, "exp(coef)"], ci_low = cs$conf.int[, "lower .95"],
                 ci_high = cs$conf.int[, "upper .95"], p_value = cs$coefficients[, "Pr(>|z|)"], row.names = NULL)
hr <- hr[order(hr$hazard_ratio), ]
write.csv(hr, file.path(out_dir, "cox_hazard_ratios_R.csv"), row.names = FALSE)
cat(sprintf("Cox PH: concordance %.3f, log partial likelihood %.1f\n", cox$concordance["concordance"], cox$loglik[2]))

png(file.path(out_dir, "R_02_cox_forest.png"), width = 1500, height = 1200, res = 170)
par(mar = c(4.5, 12, 3, 2))
k <- nrow(hr); colr <- ifelse(hr$p_value < .05, ifelse(hr$hazard_ratio > 1, RED, TEAL), GREY)
plot(hr$hazard_ratio, 1:k, log = "x", xlim = range(c(hr$ci_low, hr$ci_high)), pch = 19, col = colr, yaxt = "n",
     ylab = "", xlab = "Hazard ratio (95% CI, log scale)", main = "Cox PH hazard ratios (R, coxph)")
segments(hr$ci_low, 1:k, hr$ci_high, 1:k, col = colr, lwd = 2); abline(v = 1, lty = 2)
axis(2, 1:k, hr$feature, las = 1, cex.axis = .7)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# 5. Proportional hazards: cox.zph, then fix by time split and by stratification
# -------------------------------------------------------------------------------------
zph <- cox.zph(cox, transform = "km")
zt <- zph$table
png(file.path(out_dir, "R_03_schoenfeld_promo.png"), width = 1500, height = 1000, res = 170)
par(mar = c(4.5, 4.5, 3, 1))
plot(zph[which(rownames(zt) == "trade_promo")], resid = FALSE, col = TEAL, lwd = 2,
     xlab = "Weeks since listing (KM-transformed scale)", ylab = "Beta(t) for trade_promo",
     main = "cox.zph: time-varying log-HR of trade promotion")
abline(h = coef(cox)["trade_promo"], col = AMBER, lty = 2); abline(h = 0, lty = 3)
invisible(dev.off())

Dl <- survSplit(Surv(weeks_listed, delisted) ~ ., data = D, cut = PROMO_WEEKS, episode = "ep", start = "tstart")
Dl$promo_wk0_16 <- Dl$trade_promo * (Dl$ep == 1)
Dl$promo_after_16 <- Dl$trade_promo * (Dl$ep == 2)
covs <- setdiff(names(D), c("weeks_listed", "delisted", "trade_promo"))
f_split <- as.formula(paste("Surv(tstart, weeks_listed, delisted) ~", paste(c(covs, "promo_wk0_16", "promo_after_16"), collapse = " + ")))
cox_split <- coxph(f_split, data = Dl, ties = "efron")
ss <- summary(cox_split)
hr_split <- data.frame(feature = rownames(ss$coefficients), hazard_ratio = ss$conf.int[, "exp(coef)"],
                       ci_low = ss$conf.int[, "lower .95"], ci_high = ss$conf.int[, "upper .95"],
                       p_value = ss$coefficients[, "Pr(>|z|)"], row.names = NULL)
lr_split <- 2 * (cox_split$loglik[2] - cox$loglik[2])

f_strat <- as.formula(paste("Surv(weeks_listed, delisted) ~", paste(c(covs, "strata(trade_promo)"), collapse = " + ")))
cox_strat <- coxph(f_strat, data = D, ties = "efron")
zph_strat <- cox.zph(cox_strat, transform = "km")$table

# -------------------------------------------------------------------------------------
# 6. Parametric AFT models
# -------------------------------------------------------------------------------------
dists <- c(Weibull = "weibull", `Log-normal` = "lognormal", `Log-logistic` = "loglogistic")
aft <- lapply(dists, function(d) survreg(Surv(weeks_listed, delisted) ~ ., data = D, dist = d))
aic <- data.frame(model = names(aft), AIC = sapply(aft, AIC), log_likelihood = sapply(aft, function(m) m$loglik[2]), row.names = NULL)
aic$delta_AIC <- aic$AIC - min(aic$AIC)
tr_weib <- exp(coef(aft$Weibull)[-1])
weib_shape <- 1 / aft$Weibull$scale

# -------------------------------------------------------------------------------------
# 7. Cross-check against Python and write the report
# -------------------------------------------------------------------------------------
rel <- function(a, b) max(abs(a - b) / abs(b))
checks <- c()
py <- function(f) file.path("outputs", "python", f)
if (file.exists(py("cox_hazard_ratios.csv"))) {
  p <- read.csv(py("cox_hazard_ratios.csv")); m <- merge(p, hr, by = "feature", suffixes = c("_py", "_r"))
  checks <- c(checks, sprintf("Cox PH hazard ratios: %d compared, max relative difference **%.1e**", nrow(m), rel(m$hazard_ratio_r, m$hazard_ratio_py)))
  p2 <- read.csv(py("cox_timesplit_hazard_ratios.csv")); m2 <- merge(p2, hr_split, by = "feature", suffixes = c("_py", "_r"))
  checks <- c(checks, sprintf("Time-split Cox hazard ratios (counting process): %d compared, max relative difference **%.1e**",
                              nrow(m2), rel(m2$hazard_ratio_r, m2$hazard_ratio_py)))
  p3 <- read.csv(py("aft_time_ratios.csv")); p3 <- p3[p3$model == "Weibull", ]
  m3 <- merge(p3, data.frame(feature = names(tr_weib), tr_r = tr_weib), by = "feature")
  checks <- c(checks, sprintf("Weibull AFT time ratios: %d compared, max relative difference **%.1e**", nrow(m3), rel(m3$tr_r, m3$time_ratio)))
  p4 <- read.csv(py("aft_model_comparison.csv")); names(p4)[1] <- "model"; m4 <- merge(p4, aic, by = "model", suffixes = c("_py", "_r"))
  checks <- c(checks, sprintf("AFT AIC (Weibull / log-normal / log-logistic): max absolute difference **%.2g**", max(abs(m4$AIC_py - m4$AIC_r))))
  p5 <- read.csv(py("rmst_comparisons.csv"))
  r_rm <- unname(rm_promo[, "rmean"])
  checks <- c(checks, sprintf("RMST(52) promo vs no promo: R %.2f vs %.2f weeks | Python %.2f vs %.2f weeks",
                              r_rm[2], r_rm[1], p5$rmst_a[1], p5$rmst_b[1]))
  p6 <- read.csv(py("ph_tests.csv"))
  checks <- c(checks, sprintf("PH test for trade_promo: R cox.zph chi2 = %.1f (p = %.1e) | lifelines chi2 = %.1f. Both flag it; the statistics differ because survival >= 3.0 uses an exact score test and lifelines an approximation",
                              zt["trade_promo", "chisq"], zt["trade_promo", "p"], p6$chi2_km[p6$feature == "trade_promo"]))
} else checks <- "Python outputs not found - run the notebook first to enable the cross-check."
cat("\nCROSS-CHECK:\n", paste("-", gsub("\\*\\*", "", checks), collapse = "\n"), "\n")

md_table <- function(df) c(paste("|", paste(names(df), collapse = " | "), "|"),
                           paste("|", paste(rep("---", ncol(df)), collapse = " | "), "|"),
                           apply(df, 1, function(r) paste("|", paste(r, collapse = " | "), "|")))
f2 <- function(x, d = 2) formatC(x, format = "f", digits = d)
fp <- function(x) formatC(x, format = "g", digits = 2)
report <- c(
  "# New-SKU listing survival — R results (auto-generated by `R/fmcg_survival.R`)", "",
  sprintf("*%d outlet x SKU listings · %d delistings · %d censored (%d still listed at cut-off, %d outlet closed) · R `survival` %s*",
          nrow(df), sum(df$delisted), sum(df$delisted == 0), sum(df$status == "Censored: still listed at cut-off"),
          sum(df$status == "Censored: outlet closed"), as.character(packageVersion("survival"))), "",
  "## Cross-check with the Python notebook (lifelines)", "", paste("-", checks), "",
  "Both implementations use Efron's method for tied event times (`coxph(..., ties = \"efron\")`; lifelines' default). With Breslow ties the R hazard ratios would differ slightly, because weekly-recorded durations contain many ties.", "",
  "## Kaplan-Meier (survfit)", "",
  sprintf("- Median time to delisting: **%s weeks** (95%% CI %s–%s)", f2(km_med$quantile, 1), f2(km_med$lower, 1), f2(km_med$upper, 1)),
  sprintf("- RMST(%d): **%s weeks** (SE %s)", TAU, f2(rm_all["rmean"]), f2(rm_all["se(rmean)"])), "",
  md_table(data.frame(week = km_tab$time, still_listed = f2(km_tab$surv, 3), ci = paste(f2(km_tab$lower, 3), "–", f2(km_tab$upper, 3)),
                      at_risk = km_tab$n.risk)), "",
  md_table(data.frame(group = c("No promo", "Promo funded"), median_wks = f2(as.numeric(med_promo), 1),
                      rmst_52 = f2(rm_promo[, "rmean"]), se = f2(rm_promo[, "se(rmean)"]))), "",
  "![KM by promo](R_01_km_by_promo.png)", "",
  "## Log-rank tests (survdiff)", "",
  md_table(data.frame(variable = rownames(logrank), chi2 = f2(logrank[, "chi2"], 1), df = logrank[, "df"], p_value = fp(logrank[, "p"]))), "",
  "## Cox PH (coxph, Efron ties)", "",
  sprintf("Concordance %.3f · log partial likelihood %.1f", cox$concordance["concordance"], cox$loglik[2]), "",
  md_table(data.frame(feature = hr$feature, HR = f2(hr$hazard_ratio, 3), `95% CI` = paste(f2(hr$ci_low), "–", f2(hr$ci_high)),
                      p_value = fp(hr$p_value), check.names = FALSE)), "",
  "![Forest plot](R_02_cox_forest.png)", "",
  "## Proportional hazards (cox.zph, KM transform)", "",
  md_table(data.frame(term = rownames(zt), chi2 = f2(zt[, "chisq"], 1), p_value = fp(zt[, "p"]))), "",
  "![Schoenfeld promo](R_03_schoenfeld_promo.png)", "",
  sprintf("**Fix 1 — time split at week %d (survSplit, counting process):** likelihood-ratio chi2 vs single promo term = %.1f on 1 df.", PROMO_WEEKS, lr_split), "",
  md_table(data.frame(feature = hr_split$feature, HR = f2(hr_split$hazard_ratio, 3),
                      `95% CI` = paste(f2(hr_split$ci_low), "–", f2(hr_split$ci_high)), p_value = fp(hr_split$p_value), check.names = FALSE)[
                        hr_split$feature %in% c("promo_wk0_16", "promo_after_16", "sellthrough_per10pp", "pos_material", "has_merchandiser"), ]), "",
  sprintf("**Fix 2 — strata(trade_promo):** remaining terms with PH p < 0.01: %s (global test p = %s).",
          ifelse(any(zph_strat[rownames(zph_strat) != "GLOBAL", "p"] < .01),
                 paste(rownames(zph_strat)[zph_strat[, "p"] < .01 & rownames(zph_strat) != "GLOBAL"], collapse = ", "), "none"),
          fp(zph_strat["GLOBAL", "p"])), "",
  "## Parametric AFT models (survreg)", "",
  md_table(data.frame(model = aic$model, AIC = f2(aic$AIC, 1), log_likelihood = f2(aic$log_likelihood, 1), delta_AIC = f2(aic$delta_AIC, 1))), "",
  sprintf("Weibull shape (1/scale) = %.3f. Selected Weibull time ratios (>1 = listing lasts longer):", weib_shape), "",
  md_table(data.frame(feature = names(tr_weib), time_ratio = f2(tr_weib, 3))[order(-tr_weib), ]))
writeLines(report, file.path(out_dir, "R_results.md"))
cat("Wrote outputs/r/R_results.md, CSV and charts\n")
