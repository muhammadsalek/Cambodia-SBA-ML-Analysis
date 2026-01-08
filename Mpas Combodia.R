



#### Vector of package names####
packages <- c(
  "readxl",      # Read Excel files
  "sf",          # Handle shapefiles
  "dplyr",       # Data manipulation
  "tmap",        # Thematic mapping
  "ggplot2",     # High-quality plots
  "ggthemes",    # Themes for ggplot2
  "viridis",     # Color scales
  "RColorBrewer" # More palettes
)

# Install all packages at once
install.packages(packages)

# Load all packages
lapply(packages, library, character.only = TRUE)






# Load libraries
library(readxl)
library(sf)
library(dplyr)
library(tmap)
library(ggplot2)
library(ggthemes)
library(viridis)
library(RColorBrewer)








# Load required packages
library(readxl)
library(sf)
library(dplyr)







# Load necessary libraries
library(sf)         # for shapefiles
library(readxl)     # for Excel
library(dplyr)      # data manipulation
library(ggplot2)    # plotting
library(stringr)    # string manipulation

#-----------------------------
# 1. Load Excel data
#-----------------------------
sba_data <- read_excel("D:/Research/BDHS Research/Nepal/SBA/Combodia/Spatial/division_global_SBA_percentages.xlsx")

# Check the first few rows
head(sba_data)
names(sba_data)
#-----------------------------
# 2. Load Shapefile
#-----------------------------
shp <- st_read("D:/Research/BDHS Research/Nepal/SBA/Combodia/Spatial/khm_admin_boundaries.shp/khm_admin1.shp")

# Check the province column name
names(shp)




library(sf)
library(readxl)
library(stringr)

# Load Excel data
sba_data <- read_excel("D:/Research/BDHS Research/Nepal/SBA/Combodia/Spatial/division_global_SBA_percentages.xlsx")
excel_provinces <- str_trim(str_to_lower(sba_data$province))
cat("Provinces in Excel:\n")
print(excel_provinces)

# Load Shapefile
shp <- st_read("D:/Research/BDHS Research/Nepal/SBA/Combodia/Spatial/khm_admin_boundaries.shp/khm_admin1.shp")
shp_provinces <- str_trim(str_to_lower(shp$adm1_name))
cat("Provinces in Shapefile:\n")
print(shp_provinces)




# Correct mismatched province names in Excel to match shapefile
sba_data <- sba_data %>%
  mutate(province = case_when(
    str_to_lower(province) == "otdar meanchey" ~ "oddar meanchey",
    TRUE ~ province
  ))













library(ggplot2)
library(RColorBrewer)
library(dplyr)
library(sf)
library(stringr)

# Standardize province names for merging
sba_data <- sba_data %>%
  mutate(province_clean = str_trim(str_to_lower(province)))

shp <- shp %>%
  mutate(province_clean = str_trim(str_to_lower(adm1_name)))

# Merge Excel data with shapefile
shp_merged <- shp %>%
  left_join(sba_data, by = c("province_clean" = "province_clean"))

# Plot spatial map
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct)) +  # pick the column you want
  scale_fill_gradientn(colors = brewer.pal(9, "YlOrRd"), na.value = "grey90") +
  theme_minimal() +
  labs(title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
       fill = "SBA (%)")




































library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)   # for compass & scale bar




library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)

# Ensure province names are standardized
sba_data <- sba_data %>%
  mutate(province_clean = str_trim(str_to_lower(province)))

shp <- shp %>%
  mutate(province_clean = str_trim(str_to_lower(adm1_name))) %>%
  st_make_valid()  # fix invalid geometries

# Merge Excel with shapefile
shp_merged <- shp %>%
  left_join(sba_data, by = c("province_clean" = "province_clean"))

# Compute centroids safely
centroids <- st_centroid(shp_merged, of_largest_polygon = TRUE)  # choose largest polygon if multi-part

# Plot polished map
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct), color = "black", size = 0.3) +
  geom_sf_text(data = centroids, aes(label = adm1_name), size = 3, fontface = "bold") +
  scale_fill_gradientn(colors = brewer.pal(9, "YlOrRd"), na.value = "grey90") +
  theme_void() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  ) +
  labs(
    title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
    fill = "SBA (%)"
  ) +
  annotation_scale(location = "bl", width_hint = 0.4) +
  annotation_north_arrow(location = "tl", which_north = "true", style = north_arrow_fancy_orienteering()) +
  coord_sf(expand = FALSE)



















library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)



# Plot polished map
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct), color = "white", size = 0.4) +  # bright borders
  geom_sf_text(data = centroids, aes(label = adm1_name), size = 3, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"),  # bright red palette
    na.value = "grey95"
  ) +
  theme_minimal(base_family = "Arial") +
  theme(
    legend.position = "left",       # move legend to left
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.margin = margin(10,10,10,10)
  ) +
  labs(
    title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
    fill = "SBA (%)"
  ) +
  annotation_scale(location = "bl", width_hint = 0.4, line_width = 0.5) +
  annotation_north_arrow(location = "tl", which_north = "true",
                         style = north_arrow_fancy_orienteering()) +
  coord_sf(expand = FALSE)

















library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)

# Standardize province names
sba_data <- sba_data %>%
  mutate(province_clean = str_trim(str_to_lower(province)))

shp <- shp %>%
  mutate(province_clean = str_trim(str_to_lower(adm1_name))) %>%
  st_make_valid()

# Merge Excel with shapefile
shp_merged <- shp %>%
  left_join(sba_data, by = c("province_clean" = "province_clean"))

# Compute centroids safely
centroids <- st_centroid(shp_merged, of_largest_polygon = TRUE)

# Plot map without latitude/longitude, with frame border
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct), color = "white", size = 0.4) +
  geom_sf_text(data = centroids, aes(label = adm1_name), size = 3, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"),
    na.value = "grey95"
  ) +
  theme_void() +  # remove axes, ticks, lat/lon
  theme(
    legend.position = "left",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.margin = margin(10,10,10,10),
    panel.border = element_rect(color = "black", fill = NA, size = 1)  # frame border
  ) +
  labs(
    title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
    fill = "SBA (%)"
  ) +
  annotation_scale(location = "bl", width_hint = 0.4, line_width = 0.5) +
  annotation_north_arrow(location = "tl", which_north = "true",
                         style = north_arrow_fancy_orienteering()) +
  coord_sf(expand = FALSE)



















library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)

# Standardize province names
sba_data <- sba_data %>%
  mutate(province_clean = str_trim(str_to_lower(province)))

shp <- shp %>%
  mutate(province_clean = str_trim(str_to_lower(adm1_name))) %>%
  st_make_valid()

# Merge Excel with shapefile
shp_merged <- shp %>%
  left_join(sba_data, by = c("province_clean" = "province_clean"))

# Compute centroids safely
centroids <- st_centroid(shp_merged, of_largest_polygon = TRUE)

# Plot map without latitude/longitude and without scale bar
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct), color = "white", size = 0.4) +
  geom_sf_text(data = centroids, aes(label = adm1_name), size = 3, fontface = "bold") +
  scale_fill_gradientn(
    colors = c("#fee5d9","#fcae91","#fb6a4a","#de2d26","#a50f15"),
    na.value = "grey95"
  ) +
  theme_void() +  # remove axes, ticks, lat/lon
  theme(
    legend.position = "left",
    legend.title = element_text(size = 12, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    plot.margin = margin(10,10,10,10),
    panel.border = element_rect(color = "black", fill = NA, size = 1)  # frame border
  ) +
  labs(
    title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
    fill = "SBA (%)"
  ) +
  annotation_north_arrow(location = "tl", which_north = "true",
                         style = north_arrow_fancy_orienteering()) +
  coord_sf(expand = FALSE)


















library(sf)
library(ggplot2)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(ggspatial)
library(ggtext)  # for enhanced label styling

# Standardize province names
sba_data <- sba_data %>%
  mutate(province_clean = str_trim(str_to_lower(province)))

shp <- shp %>%
  mutate(province_clean = str_trim(str_to_lower(adm1_name))) %>%
  st_make_valid()

# Merge Excel with shapefile
shp_merged <- shp %>%
  left_join(sba_data, by = c("province_clean" = "province_clean"))

# Compute centroids safely
centroids <- st_centroid(shp_merged, of_largest_polygon = TRUE)

# Stylish map
ggplot(shp_merged) +
  geom_sf(aes(fill = Skilled_global_pct), color = "white", size = 0.3) +  # thin white borders
  geom_sf_text(
    data = centroids,
    aes(label = adm1_name),
    size = 3.5,
    fontface = "bold",
    color = "black",
    check_overlap = TRUE
  ) +
  scale_fill_gradientn(
    colors = c("#fde0dd", "#fa9fb5", "#f768a1", "#c51b8a"),  # soft pink-magenta gradient
    na.value = "grey95"
  ) +
  theme_void() +
  theme(
    legend.position = "left",
    legend.title = element_text(size = 13, face = "bold"),
    legend.text = element_text(size = 10),
    plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
    plot.margin = margin(15,15,15,15),
    panel.border = element_rect(color = "black", fill = NA, size = 1.2)
  ) +
  labs(
    title = "Skilled Birth Attendance (SBA) by Province in Cambodia",
    fill = "SBA (%)"
  ) +
  annotation_north_arrow(location = "tl", which_north = "true",
                         style = north_arrow_fancy_orienteering()) +
  coord_sf(expand = FALSE)










