![Image](https://github.com/user-attachments/assets/e51af996-3264-48e0-8579-8cd860eb4b38)

## Data-driven Insights and Predictions to Improve TTC Transit Reliability

# Description:

This project is a comprehensive transit delay analysis that takes in raw datasets from multiple transportation modes—including bus, streetcar, and subway into visual disgrams and readable data. It integrates external weather data, GTFS route information, and time-based features such as rolling averages and lag metrics to predict transit delays. It also employs machine learning techniques, using XGBoost for both classification (predicting the occurrence of delays) and regression (estimating the delay duration), while also offering visualizations and interpretability tools like SHAP to analyze model performance and feature importance.


# Tech used: 

Python along with tools below
1) Data Handling & Manipulation:
Pandas(1.5.3): For reading CSV files, merging datasets, and performing data cleaning and transformation.
NumPy(1.23.5): For efficient numerical operations and array handling.
2) Datetime Processing:
Pandas & Python’s datetime module: To parse, combine, and manipulate date and time information.
3) APIs & External Data:
Requests: For fetching weather data from the Open-Meteo Archive API.
4) Feature Engineering:
Rolling Windows: Using Pandas to calculate lag features like rolling averages and counts.
RapidFuzz: For fuzzy matching in GTFS route mapping, ensuring stop names from different datasets are correctly aligned.
5) Visualization:
Matplotlib(3.6.2): For plotting graphs like confusion matrices, regression error plots, and time series charts.
Folium(0.12.1): For creating interactive maps to visualize geographical risk and route based data.
6) Machine Learning & Model Training:
XGBoost(1.6.2): For both classification and regression tasks, providing robust predictive modeling capabilities.
Scikit-learn(1.2.0): For data splitting, label encoding, and evaluation metrics such as classification reports and mean absolute error.
7) Model Interpretability:
SHAP(0.41.0): To generate summary plots that help explain the impact of features on model predictions.

# USER INSTRUCTION:

Data Preparation:

1) Ensure your raw transit data files (bus, streetcar, and subway CSVs) and the supplementary files (subway delay codes, GTFS stops, etc.) are placed in the correct directory structure (e.g., data/, data/external/).
2) Update any file paths in the code if your data is stored in a different location.

Preprocessing and Feature Engineering:

3) Run the main preprocessing script (for example, by executing python main_preprocessing.py) to load, clean, and merge the datasets. This script creates an enriched dataset by integrating weather data, time-based features, and GTFS route information.
4) The output will be an enriched CSV file (e.g., enriched_data.csv) that consolidates all the transit data with additional features.

Visualization:

5) Use the provided plotting functions (such as plot_confusion_matrix, regression_error_plot, or the Folium-based mapping function) within an interactive Python session or notebook to visualize data distributions, model performance, or geographical route risk.
6) Customize parameters (titles, axis labels, etc.) as needed to better fit your analysis.

Model Training and Evaluation:

7) Run the training scripts (e.g., python train_models.py) to prepare the classification and regression datasets from the enriched CSV.
8) The code will split the data, train XGBoost models for delay prediction, and print evaluation metrics such as classification reports and Mean Absolute Error (MAE).
9) Review the output to assess model performance and make adjustments to feature engineering or hyperparameters if necessary.

Extending the Pipeline:

10) You can modify individual modules (e.g., feature engineering, weather integration, or GTFS enrichment) to accommodate new data sources or tweak the modeling approach.
11) Use the modular structure to experiment with different visualization or machine learning techniques.
By following these steps, users can seamlessly load and process transit data, generate enriched datasets, visualize key insights, and train predictive models to analyze transit delay events.

## Description:

# Initial Data visualization:

Creating the code ngetes.py was our Initial attempt to better understand and clean our data. The code begins by importing multiple CSV files containing data for different transit modes (bus, streetcar, subway) and additional external datasets (weather, events, and ridership).
Purpose: To bring together various sources of information and visualize it. 

Main result:

Bus Average Delay

![Image](https://github.com/user-attachments/assets/2090e58d-9419-4f4e-b115-19d9dc6d7490)

Streetcar Average Delay

![Image](https://github.com/user-attachments/assets/fc76569a-4b6b-40b9-b8b9-7579f638ea8d)

# Data processing:

Transit Data Loading: Reads separate CSV files for bus, streetcar, and subway datasets.
Subway Code Mapping: Loads a mapping of subway delay codes to descriptive incident texts, enabling easier interpretation of subway delay events.
Data Unification: Combines the different transit modes into a single, time-sorted DataFrame, for a consistent schema for further processing.
Data Cleaning: Drops records with missing timestamps or delay values and filters out any invalid or negative delay entries.

# Feature engineering:

1) Time based feature
Extracts critical temporal components from the timestamp—such as the hour, month, and weekday—and creates a binary flag to indicate weekends. These features help capture cyclical patterns in transit delays. The transit datasets (data/bus-data.csv, data/streetcar-data.csv, data/subway-data.csv) provide the raw timestamps.

2) Weather integration
Merges historical hourly weather data from the Open-Meteo Archive into the transit dataset. By rounding transit timestamps to the nearest hour, the weather conditions (such as temperature and precipitation) are aligned with the transit events, allowing the model to account for external weather influences. Weather data is fetched via an API (e.g., from Open-Meteo Archive). The transit data files (as above) supply the timestamps to match against.

3) Lag features
Calculates rolling statistics (such as the mean delay and the count of delay events) over a configurable time window (e.g., one hour). These lag features capture historical trends that may help predict future delays.The transit delay information is sourced from the combined dataset created from data/bus-data.csv, data/streetcar-data.csv, and data/subway-data.csv.

4) GTFS Data
Enhances the transit data with geographic and routing information from GTFS files. The project loads the GTFS stops file (located at data/external/gtfs/stops.txt), normalizes stop names, and uses fuzzy matching or direct string matching to merge latitude and longitude details into the transit dataset. This enrichment adds a spatial dimension to the analysis.GTFS stops file: data/external/gtfs/stops.txt. Transit data files (after unification) provide the location field used for matching.

# Model training:

The model training process is organized into two main pathways for classification and one for regression, each built upon key data preparation steps and model evaluation techniques using XGBoost. In the classification pipeline, the function prepare_classification_data (from model_training.py) converts the raw delay data into a binary target variable (delay_binary) by marking records with delays greater than or equal to a threshold 1, and 0 otherwise. It then selects key features such as hour, month, is_weekend, mode, and a unified route_or_line (which combines route or line information) and applies label encoding to transform categorical features into numerical values.

<img width="565" alt="Image" src="https://github.com/user-attachments/assets/21c742d4-7ae4-47e4-ba92-47c0e696ebed" />

After preparing the data, the classification function splits the dataset into training and testing sets, trains an XGBoost classifier (using parameters like 100 estimators and a maximum depth of 6), and then evaluates its performance with a classification report. Similarly, the regression pipeline filters out records without delays, prepares the same set of features, splits the data, trains an XGBoost regressor, and checks its performance by calculating the Mean Absolute Error (MAE). 

<img width="705" alt="Image" src="https://github.com/user-attachments/assets/f7ff38e9-925d-4afe-b322-eeab168520bd" />

The result are visually represented as shown below. As we can see our result achieved up to 80% accuracy.

![Image](https://github.com/user-attachments/assets/f69ee537-ab50-4ba1-8060-f2a3751cb7e2)

# EDA (Exploratory Data Analysis)

Subway heatmap:

This code creates an interactive subway heatmap for Toronto by loading a CSV file containing subway incident data with geographical coordinates, cleaning the data by removing entries without latitude or longitude, and then aggregating the data to count the number of incidents at each station. It uses a pivot table to gather each station's first recorded latitude and longitude along with the total incident count. A base map is then generated using Folium, centered on Toronto, and a heatmap layer is overlaid using the incident counts to visualize density—areas with higher incidents appear warmer. Additionally, CircleMarkers are added at each station, with the marker color set to red for stations above the 75th percentile in incident counts and blue otherwise, providing a clear visual distinction of high-incident areas. Finally, the map is saved as an HTML file, which can be viewed interactively in a web browser.

Geographical route risk:

This HTML file creates an interactive map visualization using the Leaflet library along with several plugins and dependencies such as jQuery, Bootstrap, Leaflet Awesome Markers, and the Leaflet MarkerCluster plugin. It sets up the map container and initializes a Leaflet map centered on Toronto with a specified zoom level and an OpenStreetMap tile layer as its base. The code then creates a marker cluster group to manage multiple markers efficiently, adding individual markers with customized icons (red, green, or orange) that denote different routes or risk levels. Each marker is equipped with a popup that displays detailed information (like "Location: Route X" and a corresponding risk value) when clicked. Overall, this file produces a dynamic, web-based map that visually aggregates transit data, making it easy to identify high-risk areas and view detailed information about specific locations through an intuitive, zoomable interface.

This is shown below where the risk factor is in the range from 0 to 1

![Image](https://github.com/user-attachments/assets/e844b507-cc3f-4480-b774-c7d1181ce43f)

![Image](https://github.com/user-attachments/assets/d903af93-4d99-4486-86b4-c7b23cc5ff8b)

# Final Data Visualization

Diamater - minute delayed
Colour ranges from blue to red - Blue represents lower delay frequency and Red represents higher delay frequency

Here we are able to remodel the vizualization to the figure below:

![Image](https://github.com/user-attachments/assets/f8b96282-91bd-47a0-933a-4273dc58c664)

This is the final combined visualization:

<img width="871" alt="Image" src="https://github.com/user-attachments/assets/3caf3d81-fe2d-4305-a5ab-02a2adb59ee4" />


# Sources

Open Data Toronto​ - tellingstorieswithdata.com

Weather features - [openreview.net](https://open-meteo.com)

streetcar delay - cs229.stanford.edu & ionnoant.github.io

Specific Weater condition - https://github.com/Patrickdg/TTC-Delay-Analysis#

Delay Duration Regression - yooksel.com

Geographic Hotspots of Delays - [github.com(https://github.com/JasonYao3/TTC_transit_delay_proj#)

Extreme weather delay - public.tableau.com





