Challenge



1. Model

2. Mapping

The main function of the R file is to extract sleep survey response data from the CDC and ACS data from the US Census using the tidycensus library. Subsequently, the data is merged based on Census Tract numbers.

The sleep data is sourced from the CDC (https://data.cdc.gov/500-Cities-Places/500-Cities-Sleeping-less-than-7-hours-among-adults/eqbn-8mpz/data_preview). Locations with less than 50 data points are disregarded as they do not provide sufficient sleep information.

Next, additional information is obtained from the US Census in two ways:
a. You can visit the US Census ACS website and query the data.
b. You can access it through an API.

In this project, we did the second method as we are familiar with the US Census database and have our own API code. The `tidycensus` library is utilized, and available variables can be explored through the library. We primarily used TABLE B19013_001 for median income information and TABLE B24011_002, B24011_029, B24011_033 for occupation information.

For visualization, we employed `geom_sf()` from the `ggplot2` library. State FIPS codes were obtained from USPS/US Census information, and coordinates were sourced from Google Maps. The city center was used as the focal point, and the map size was adjusted accordingly.

3. Web Application
