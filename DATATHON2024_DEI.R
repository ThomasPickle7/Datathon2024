#RICE DATATHON 2024
#theodore kim
# 2024-01-19

#Purpose:
#the main purpose of this file is to get the sleep data from CDC
#and get ACS data from US census through tidycensus library.
#After that, I need to merge by Census Tract number
#

#Library
library(tidycensus) #us census data access
library(dplyr)
library(ggplot2) #graph
library(tidyr) #data processing
library(sf) #drawing map
library(psych) #correlation
library(ggcorrplot)

getwd()
setwd("C://Users//tk46//Desktop//DATATHON")

#sleep data is from:
#https://data.cdc.gov/500-Cities-Places/500-Cities-Sleeping-less-than-7-hours-among-adults/eqbn-8mpz/data_preview
tract_sleep<-read.csv('censustractsleep.csv',
                      colClasses=c("TractFIPS"="character"))

colnames(tract_sleep)
head(tract_sleep)

#subsetting
tract_sleep_sample<-tract_sleep%>%dplyr::select(c("Year","StateAbbr","CityName",
                                                  "Data_Value","Low_Confidence_Limit","High_Confidence_Limit",
                                                  "PopulationCount","GeoLocation","CityFIPS","TractFIPS"))%>%
  filter(PopulationCount>50)
# a location with less than 50 shows no value
#so, I decide to remove those first.


#get some other information from US Census
#There is two ways to get the data.
#1. you can go to the US Census ACS website and query the data.
#2. you can access through API
#Here, I do number 2 since I am familiar with the US Census database : Theodore Kim

my_api_key="Your_API_Key_Goes_HERE"
#this is my personal API code. Theodore Kim
#if you need your API code, there is a website in US Census website.

options(tigris_use_cache = TRUE)

census_api_key(my_api_key)

#I need:
#Year 2016, 2017
#TABLE B19013_001: Median Household Income
#TABLE B03002_001: Demographic information: Race
#TABLE B24011_001: Occupation

acs17 <- load_variables(2017, "acs5", cache = TRUE)
#available variables can be seen in acs17.
#this can be done by the library(tidycensus)

STATEABB<-c("AL","AK","AZ","AR","CA",
            "CO","CT","DE","FL","GA",
            "HI","ID","IL","IN","IA",
            "KS","KY","LA","ME","MD",
            "MA","MI","MN","MS","MO",
            "MT","NE","NV","NH","NJ",
            "NM","NY","NC","ND","OH",
            "OK","OR","PA","RI","SC",
            "SD","TN","TX","UT","VT",
            "VA","WA","WV","WI","WY")
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


tract_income<-get_acs(geography="tract",
        variables=c(medianincome="B19013_001"),
        state=c(STATEABB),
        survey="acs5",
        year=2017,geometry=TRUE)
head(tract_income)
colnames(tract_income)

#save this for later
tract_geometry<-tract_income%>%select(c(GEOID,NAME,geometry))

#make it wide form
tract_income<-get_acs(geography="tract",
                      variables=c(medianincome="B19013_001"),
                      state=c(STATEABB),
                      survey="acs5",
                      year=2017)
tract_income<-tract_income%>%select(-c(NAME,moe))
tract_income_w<-
  spread(tract_income, key=variable, value=estimate)


#GEOID reference: US census
#https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm

#checking whether the census tract code dtype is identical
head(tract_income$GEOID)
head(tract_sleep_sample$TractFIPS)


tract_all<-merge(tract_sleep_sample,tract_income_w,by.x="TractFIPS",by.y="GEOID",
                 all.x=TRUE)

tract_map<-merge(tract_geometry,tract_all,by.x="GEOID",by.y="TractFIPS",all.x=TRUE)
#tract_map will be our motherboard

############
#length(unique(tract_all$TractFIPS))==nrow(tract_all%>%filter(Year==2016))
#
#dataframe: tract_map is the motherboard
#in other word, merge allthings to the tract_map(all.x)

###########
#tract_poverty<-get_acs(geography="tract",year=2017,state=c(STATEABB),survey="acs5",geometry = TRUE,
#                       )
tract_job<-get_acs(geography="tract",year=2017,state=c(STATEABB),
                   survey="acs5",
                   variables=c(mgmt="B24011_002",
                               construction="B24011_029",
                               trans="B24011_033"))
#This is the job count in each tract.
#I picked three categories here. There is more room to improve.

tract_job<-tract_job%>%select(-c(NAME,moe))
tract_job<-
  spread(tract_job,key=variable, value=estimate)

#I decide to make white collar job and blue collar job.
#Because blue collar labor may start work earlier in the morning.
tract_job$whitecollar<-tract_job$mgmt
tract_job$bluecollar<-tract_job$construction+tract_job$trans

tract_job_w<-tract_job%>%select(c(GEOID,whitecollar,bluecollar))

#I also try to incorporate the demographic information in tract
tract_race<-get_acs(geography="tract",year=2017,state=c(STATEABB),
                    survey="acs5",
                    variables=c(WHITE="B03002_001",
                                AFRICANAMERICAN="B03002_004",
                                HISPANIC="B03002_012",
                                ASIAN="B03002_006"))
tract_race<-tract_race%>%select(-c(NAME,moe))
tract_race<-
  spread(tract_race, key=variable, value=estimate)

#citation: usage of tidycensus calling data
#https://www.rdocumentation.org/packages/tidycensus/versions/1.5/topics/get_acs

###########
#merging and correlation
colnames(tract_map)

#merge do this only once
tract_map<-merge(tract_map,tract_job_w,by="GEOID",all.x=TRUE)

#check merge status
colnames(tract_map)

correlation_test<-data.frame(
  TroubleSleep=tract_map$Data_Value,
  Income=tract_map$medianincome,
  Totalpop=tract_map$PopulationCount,
  WCjob=tract_map$whitecollar,
  BCjob=tract_map$bluecollar)

correlation_coefficients<-round(cor(correlation_test,use="complete.obs"),digits=2)

pvalue_corr <- corr.test(correlation_test)$p    # Apply corr.test function
pvalue_corr

ggcorrplot(correlation_coefficients,
           title = "Correlation Chart",
           legend.title = "Coeff",
           type=c("upper"),
           show.diag = TRUE,
           colors = c("blue", "white", "red"),
           outline.color = "black",
           sig.level = 0.01,
           p.mat=pvalue_corr)

#x indicates "non-significant"

#ggcorrplot documentation reference
#https://www.rdocumentation.org/packages/ggcorrplot/versions/0.1.4.1/topics/ggcorrplot

###########
#here I am manually making breaks because I want to visualize more simpler and inituitive.
#the gradual change of color sometimes confuse the audience
#when they wanna do side by side comparison
quantile(tract_ggplot$Data_Value,probs=seq(0,1,1/4),na.rm=TRUE)
min(tract_ggplot$Data_Value,na.rm=TRUE)
max(tract_ggplot$Data_Value,na.rm=TRUE)
breaks_sleep=c(16.1,32.3,36.1,40.5,55.4)
my_color_choice<-c("#8abcff",
                   "#2380fe",
                   "#0355c3",
                   "#023478")
#color hex code: https://html-color.codes/

state_mapping<-get_acs(geography="state",survey="acs5",
                       variables="B19013_001",
                       year=2017,geometry=TRUE)%>%
  select(c(GEOID,geometry))%>%filter(!grepl("^(02|11|15|72|76)",GEOID))

#AL,HI,DC,PR,VI is removed here
#This is continental United States
tract_ggplot<-tract_map%>%filter(!grepl("^(02|11|15|72|76)",GEOID))
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#this will show entire continental united states
ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot,aes(fill=Data_Value),color="transparent")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf()+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="The United States Map",
       subtitle="% Less than 7 Hours of Sleep (Adults aged >=18 years)",
       caption="Data Source:  500 Cities: Local Data for Better Health, 2019 release")
#drawing map geom_sf()
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#separate maps
sf_use_s2(use_s2=FALSE)
#citation:
#https://www.rdocumentation.org/packages/sf/versions/1.0-15/topics/s2
#turn back TRUE is the default
#this allows me to set latitude and longtitude in decimal places

#TEXAS testing
ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot,aes(fill=Data_Value),color="transparent")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(-107, -93), ylim = c(25.7, 36.7), expand = FALSE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="% Less than 7 Hours of Sleep",
       subtitle="Adults, Age>=18",
       caption="Data Source:  CDC 2019 release")
#drawing map geom_sf()+coord_sf()
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


######################
#Houston Area
#state code: 48
#Houston location:
point_to=c(-95.36,29.76)
#city hall
#checking the coordination using GoogleMap

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^48",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.6, point_to[1]+0.6),
           ylim = c(point_to[2]-0.3,point_to[2]+0.3), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="Houston, TX",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#NEW YORK CITY
#state code: 36
#new jersey is 34
#NYC location Manhattan
point_to=c( -73.98,40.74)
#empire state building
#checking the coordination using GoogleMap

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(36|34)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.2, point_to[1]+0.2),
           ylim = c(point_to[2]-0.1,point_to[2]+0.1), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="NYC, NY",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#SEATTLE
#state code is 53
#Seattle WA Downtown location
point_to=c(-122.33,47.6)
#amazon company buildings

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(53)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.3, point_to[1]+0.3),
           ylim = c(point_to[2]-0.125,point_to[2]+0.125), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="Seattle, WA",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#CHICAGO
#state code is 17
#chicago location is
point_to=c(-87.6,41.8)
#riverwalk

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(17)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.2, point_to[1]+0.2),
           ylim = c(point_to[2]-0.1,point_to[2]+0.1), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="Chicago, IL",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#ATLANTA
#STATE CODE IS 13
#LOCATION IS:
point_to=c(-84.40,33.75)
#mercedes-benz stadium

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(13)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.2, point_to[1]+0.2),
           ylim = c(point_to[2]-0.1,point_to[2]+0.15), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="Atlanta, GA",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm



#LA
#STATE CODE IS 06
#LOCATION IS:
point_to=c(-118.2,34.1)
#dodger stadium


ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(06)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.4, point_to[1]+0.4),
           ylim = c(point_to[2]-0.2,point_to[2]+0.2), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="Los Angeles, CA",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm


#SAN FRANCISCO
#STATE CODE IS 06
#LOCATION IS:
point_to=c(-122.4,37.76)
#San Bruno Mountain

ggplot()+
  geom_sf(data=state_mapping,color="black",fill="#d1ffe1",
          linewidth=0.1,inherit.aes=FALSE)+
  geom_sf(data=tract_ggplot%>%filter(grepl("^(06)",GEOID)),aes(fill=Data_Value),color="white")+
  scale_fill_stepsn(colours = my_color_choice,
                    breaks = breaks_sleep,
                    name = "Percentage",na.value = "transparent")+
  coord_sf(xlim = c(point_to[1]-0.2, point_to[1]+0.2),
           ylim = c(point_to[2]-0.1,point_to[2]+0.1), expand=TRUE)+
  theme(panel.background = element_rect(fill = 'lightblue'),
        panel.grid.major = element_line(color="#f5f6f6"))+
  labs(title="San Francisco, CA",
       subtitle="% Less than 7 Hours of Sleep",
       caption="Data Source:  CDC 2019 release")
#documentation: https://ggplot2.tidyverse.org/reference/ggsf.html
#color hex code: https://html-color.codes/
#FIPS code: USPS State FIPS code list
#https://www.bls.gov/respondents/mwr/electronic-data-interchange/appendix-d-usps-state-abbreviations-and-fips-codes.htm



#investigation
#noise pollution
#from
#https://data-usdot.opendata.arcgis.com/documents/usdot::2016-noise-data/about
#this data provides ArcGIS data
