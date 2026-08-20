# ============================================================
# HEATWAVE DETECTION (per site) using heatwaveR
# Definition used here:
# - Heatwave = ≥ minDuration consecutive days above a percentile threshold
#   of a reference climatology (e.g., 90th percentile).
#
# Inputs required in `data`:
#   - site: site/region identifier
#   - date_col: date column name (Date)
#   - var_col: temperature variable column name (numeric)
# Notes:
#   - For robust results, dates should be daily and continuous (no gaps),
#     or at least consistently sampled across years.
# ============================================================

# List of packages required for the script (To be adjusted according to what you are using)

required_packages <- c("dplyr", "ggplot2", "lubridate", "tidyr","purrr","readxl","heatwaveR")

# Function to check and install missing packages
install_if_missing <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE)) {
      install.packages(package)
      library(package, character.only = TRUE)
      }
  }
}


install_if_missing(required_packages)
# ============================================================
# Heatwaves (heatwaveR) + Rainfall/Temp anomalies vs baseline
# ============================================================

# -------------------- CONFIG --------------------
infile <- "./era5land_weather_zoning_fr100101to250922.xlsx"

site_keep <- c("Bouarfa", "Missour", "North Maatarka", "South Maatarka")

clim_period <- c("2010-01-01", "2024-12-31")   # climatologyPeriod for ts2clm()
baseline_years <- 2010:2024                   # baseline years for anomalies

pctile <- 90
minDuration <- 3

# If your file uses other names, change these:
site_col_in <- "working_region"
precip_col_in <- "precipitation_daily_utc"    # used later in yearly sums

# -------------------- HELPERS --------------------

# auto_rename_cols(): Auto-detects and renames date/Tmean/Tmin/Tmax columns from the raw file; 
#stops if missing/ambiguous.

auto_rename_cols <- function(df) {
  nms <- names(df)
  
  pick1 <- function(pattern, label) {
    hit <- nms[grepl(pattern, nms, ignore.case = TRUE)]
    if (length(hit) == 0) stop("No match for ", label, " using pattern: ", pattern)
    if (length(hit) > 1)  stop("Ambiguous ", label, ". Matches: ", paste(hit, collapse = ", "))
    hit
  }
  
  df %>%
    dplyr::rename(
      date  = dplyr::all_of(pick1("^date$|standard time|time$|timestamp|datetime", "date")),
      Tmean = dplyr::all_of(pick1("tmean|t2m.*mean|temp.*mean|mean.*temp", "Tmean")),
      Tmin  = dplyr::all_of(pick1("tmin|t2m.*min|min(imum)?|temp.*min|min.*temp", "Tmin")),
      Tmax  = dplyr::all_of(pick1("tmax|t2m.*max|max(imum)?|temp.*max|max.*temp", "Tmax"))
    )
}

# detect_hw_by_site(): Computes climatology (ts2clm) and detects heatwave events (detect_event) 
#per site for a chosen temperature variable.

detect_hw_by_site <- function(df, var_col,
                              climatologyPeriod = clim_period,
                              pctile = 90,
                              minDuration = 3) {
  df %>%
    group_by(site) %>%
    nest() %>%
    mutate(
      ts = map(data, ~{
        x <- .x %>% transmute(t = date, temp = .data[[var_col]])
        ts2clm(x, climatologyPeriod = climatologyPeriod, pctile = pctile)
      }),
      full_event = map(ts, ~detect_event(.x, minDuration = minDuration)),
      events = map(full_event, "event")
    ) %>%
    select(site, full_event, events)
}

# extract_events(): Extracts (unnests) the per-event table from a heatwaveR object and 
#tags it with the variable name.
extract_events <- function(hw_tbl, var_name) {
  hw_tbl %>%
    transmute(site, variable = var_name, event = map(full_event, "event")) %>%
    unnest(event)
}

# compute_anoms(): Computes site-specific baseline means over baseline_years and adds 
#anomaly columns as observed − baseline (plus optional % anomaly for rainfall).
compute_anoms <- function(df, value_cols, baseline_years, year_col = "year") {
  
  base <- df %>%
    dplyr::filter(.data[[year_col]] %in% baseline_years) %>%
    dplyr::group_by(site) %>%
    dplyr::summarise(
      dplyr::across(dplyr::all_of(value_cols), ~mean(.x, na.rm = TRUE),
                    .names = "clim_{.col}"),
      .groups = "drop"
    )
  
  out <- df %>%
    dplyr::left_join(base, by = "site")
  
  # Create anomaly columns (no cur_column)
  for (v in value_cols) {
    clim_v <- paste0("clim_", v)
    anom_v <- paste0("anom_", v)
    out[[anom_v]] <- out[[v]] - out[[clim_v]]
  }
  
  # Optional: percent anomaly for precipitation
  if ("prcp_sum" %in% value_cols) {
    out$anom_prcp_pct <- 100 * out$anom_prcp_sum / pmax(out$clim_prcp_sum, 1e-6)
  }
  
  out
}

# save_hw_plots(): Generates and saves event_line plots for each site’s heatwave object to PNG files.
save_hw_plots <- function(hw_obj, out_dir = "heatwave_plots", prefix = "Tmax") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  for (i in seq_len(nrow(hw_obj))) {
    site_name <- hw_obj$site[i]
    p <- event_line(hw_obj$full_event[[i]], spread=1000)#change spread to define the period you want to plot
    
    ggsave(
      filename = file.path(out_dir, paste0(prefix, "_", site_name, ".png")),
      plot = p, width = 10, height = 4, dpi = 300
    )
  }
}



# -------------------- LOAD + PREP --------------------
clim_daily <- read_excel(infile) %>%
  auto_rename_cols() %>%
  rename(site = all_of(site_col_in)) %>%
  mutate(date = as.Date(date)) %>%                         # adapt if your date col differs
  filter(!is.na(date), site %in% site_keep) %>%
  mutate(year = year(date))

# -------------------- HEATWAVES --------------------
study <- clim_daily %>%
  filter(date >= as.Date(clim_period[1]), date <= as.Date(clim_period[2]))

hw_tmean <- detect_hw_by_site(study, "Tmean", clim_period, pctile, minDuration)
hw_tmin  <- detect_hw_by_site(study, "Tmin",  clim_period, pctile, minDuration)
hw_tmax  <- detect_hw_by_site(study, "Tmax",  clim_period, pctile, minDuration)

events_tmean <- extract_events(hw_tmean, "Tmean")
events_tmin  <- extract_events(hw_tmin,  "Tmin")
events_tmax  <- extract_events(hw_tmax,  "Tmax")
heatwave_events_all <- bind_rows(events_tmean, events_tmin, events_tmax)


#plots_automatically to be adjusted:
save_hw_plots(hw_tmean, out_dir = "heatwave_plots", prefix = "Tmean")
save_hw_plots(hw_tmin,  out_dir = "heatwave_plots", prefix = "Tmin")
save_hw_plots(hw_tmax,  out_dir = "heatwave_plots", prefix = "Tmax")


# Quick single-site  plot
i <- which(hw_tmax$site == "Missour")[1]
event_line(hw_tmax$full_event[[i]], start_date = "2015-05-01", end_date = "2015-05-30", category = TRUE)
event_line(hw_tmax$full_event[[i]], spread=1000)



# Heatwave count heatmap (example using Tmean events)
ev_counts <- events_tmean %>%
  mutate(
    year  = year(date_start),
    month = factor(month(date_start, label = TRUE, abbr = TRUE),
                   levels = month.abb, ordered = TRUE)
  ) %>%
  count(site, year, month, name = "n_events") %>%
  ungroup() %>%                               # <- IMPORTANT
  complete(site, year, month, fill = list(n_events = 0))


ggplot(ev_counts, aes(month, year, fill = n_events)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_wrap(~site) +
  labs(x = "Month", y = "Year", fill = "Nb events") +
  scale_y_continuous(breaks = 2016:2025) +
  scale_fill_gradientn(
    colours = c("white", "yellow", "orange"),
    name = "Nb. events"
  ) +
  labs(x = "Month", y = "Year") +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    axis.title.x = element_text(size = 16, face = "bold", color="black"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text.x  = element_text(size = 16, angle = 45, hjust = 1,colour = "black",face="bold"),
    axis.text.y  = element_text(size = 16,colour = "black",face="bold"),
    strip.text   = element_text(size = 16, face = "bold"),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 16)
  ) 

# Mean intensity heatmap (example using Tmax events)
ev_intensity <- events_tmax %>%
  mutate(year = year(date_start),
         month = factor(month(date_start, label = TRUE, abbr = TRUE),
                        levels = month.abb, ordered = TRUE)) %>%
  group_by(site, year, month) %>%
  summarise(mean_intensity = mean(intensity_mean, na.rm = TRUE), .groups = "drop") %>%
  ungroup() %>%                               # <- IMPORTANT
  complete(site, year, month, fill = list(mean_intensity = 0))

ggplot(ev_intensity, aes(month, year, fill = mean_intensity)) +
  geom_tile(color = "white", linewidth = 0.3) +
  facet_wrap(~site) +
  labs(x = "Month", y = "Year", fill = "Mean intensity") +
  scale_y_continuous(breaks = 2010:2024) +
  scale_fill_gradientn(
    colours = c("white", "yellow", "orange"),
    name = "Mean Intesity"
  ) +
  labs(x = "Month", y = "Year") +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    axis.title.x = element_text(size = 16, face = "bold", color="black"),
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.text.x  = element_text(size = 16, angle = 45, hjust = 1,colour = "black",face="bold"),
    axis.text.y  = element_text(size = 16,colour = "black",face="bold"),
    strip.text   = element_text(size = 16, face = "bold"),
    legend.title = element_text(size = 16, face = "bold"),
    legend.text  = element_text(size = 16)
  ) 



# ============================================================
# ANOMALIES VS BASELINE (temperature + precipitation)
#
# Concept:
# For each site, and for a given time unit (month OR season OR year):
# - Compute baseline mean ("climatology") over baseline_years
# - Anomaly = observed - climatology
#
# Inputs:
# - df: aggregated table (monthly, seasonal, or yearly)
# - key_cols: time unit columns used for matching (e.g., "month" or "season")
# - value_cols: variables to compute anomalies for (e.g., t_mean, prcp_sum)
# - baseline_years: reference years used to define the baseline climatology
# - year_col: which column represents the year for filtering baseline years
#
# Output:
# - adds columns clim_* (baseline mean) and anom_* (absolute anomaly)
# - optional: anom_prcp_pct for precipitation (% anomaly)
# ============================================================

# -------------------- YEARLY ANOMALIES (TEMP + RAIN) --------------------
clim_yearly <- clim_daily %>%
  group_by(site, year) %>%
  summarise(
    t_mean  = mean(Tmean, na.rm = TRUE),
    t_min   = mean(Tmin,  na.rm = TRUE),
    t_max   = mean(Tmax,  na.rm = TRUE),
    prcp_sum = sum(.data[[precip_col_in]], na.rm = TRUE),
    n_days  = n(),
    .groups = "drop"
  )

yearly_anoms <- compute_anoms(
  df = clim_yearly,
  value_cols = c("t_mean", "t_min", "t_max", "prcp_sum"),
  baseline_years = baseline_years
)

ggplot(filter(yearly_anoms, year %in% baseline_years), aes(year, anom_prcp_sum)) +
  geom_hline(yintercept = 0, linetype = 2) +
  geom_col() +
  geom_col(fill = "steelblue") +
  facet_wrap(~site, scales = "free_y") +
  labs(title = "Yearly precipitation anomaly vs baseline", x = "Year", y = "Anomaly (mm)") +
  theme_minimal()


#------------------------------Easiest way in case you have/interested in one site only --------------------------------

#Isolate one site
missour_Data <- clim_daily %>%
  filter(site == "Missour") %>%
  transmute(
    t    = as.Date(date),     # date column
    temp = as.numeric(Tmax)   # temperature column (or your real Tmax/Tmin/Tmean column)
  ) %>%
  arrange(t)

#will generate a dataframe with threshold per day and the mean and actual temperature
ts <- ts2clm(missour_Data, climatologyPeriod = c("2010-01-01", "2024-12-31"), pctile = 90)
res <- detect_event(ts)

event_line(res)