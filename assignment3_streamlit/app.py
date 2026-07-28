# ---------------------------------------------------------
# import the libraries
# ---------------------------------------------------------


import json
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path


# ---------------------------------------------------------
# page settings


st.set_page_config(
    page_title="Vancouver Business Similarity Explorer",
    page_icon="🏙️",
    layout="wide",
)


# ---------------------------------------------------------
# simple app design


# i added my own colours to make the app look more complete.
# these colours only affect the presentation and not the analysis.
st.markdown(
    """
    <style>

    .stApp {
        background-color: #171421;
        color: #F5F1FF;
    }

    .block-container {
        max-width: 88rem;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2 {
        color: #D7B8FF;
    }

    h3 {
        color: #F4B8D7;
    }

    p, label {
        color: #F5F1FF;
    }

    div[data-testid="stMetric"] {
        background-color: #29233A;
        border-top: 0.25rem solid #D7B8FF;
        border-radius: 0.75rem;
        padding: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 0.08rem solid rgba(245, 241, 255, 0.20);
        border-radius: 0.75rem;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        background-color: #29233A;
        border-left: 0.3rem solid #F4B8D7;
        color: #F5F1FF;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# locate the original geojson file


# the original dataset should be in the same folder as app.py.
DATA_PATH = Path(__file__).parent / "business-licences.geojson"


# ---------------------------------------------------------
# business-group mapping from my part a notebook


# in part a, i reduced the many original business types into broader
# categories. i reused the same mapping here so both parts of the
# assignment remain consistent.
business_mapping = {

    # healthcare
    "Health Care Professionals and Services": "Healthcare",
    "Health Care Facility": "Healthcare",

    # food and hospitality
    "Restaurant": "Food & Hospitality",
    "Limited Service Food Establishment": "Food & Hospitality",

    # retail
    "Retail Dealer": "Retail",
    "Retail Dealer - Food": "Retail",
    "Wholesale Dealer - Non-Food": "Retail",

    # professional services
    "Legal Services": "Professional Services",
    "Financial Services": "Professional Services",
    "Business Support Services": "Professional Services",
    "Consulting and Management Services": "Professional Services",
    "Architectural and Engineering Services": "Professional Services",

    # real estate
    "Long-term Rental": "Real Estate",
    "Real Estate Services": "Real Estate",

    # technology
    "Information Communication Technology": "Technology",

    # construction
    "General Contractor": "Construction",

    # manufacturing
    "Non-Food Manufacturer Assembler and Processor": "Manufacturing",

    # personal services
    "Beauty Services": "Personal Services",
}


# ---------------------------------------------------------
# load and clean the original geojson


@st.cache_data(show_spinner=False)
def load_and_clean_data(file_path):
    """
    this function loads the original geojson and repeats the main cleaning
    decisions from my notebook that are needed for part b.
    """

    # open the original geojson file.
    with open(file_path, "r", encoding="utf-8") as file:
        geojson_data = json.load(file)

    rows = []

    # every geojson feature contains business information under properties
    # and its coordinates under geometry.
    for feature in geojson_data["features"]:

        properties = feature.get("properties", {})
        geometry = feature.get("geometry")

        longitude = None
        latitude = None

        # i only extract coordinates when the geometry is a valid point.
        if (
            geometry
            and geometry.get("type") == "Point"
            and geometry.get("coordinates")
            and len(geometry["coordinates"]) >= 2
        ):
            longitude, latitude = geometry["coordinates"][:2]

        rows.append(
            {
                **properties,
                "longitude": longitude,
                "latitude": latitude,
            }
        )

    df = pd.DataFrame(rows)

    # only issued licences were used in my notebook because they represent
    # businesses with active licences.
    df = df[df["status"] == "Issued"].copy()

    # businesses without coordinates cannot be placed on the map.
    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).copy()

    # part b also requires the neighbourhood and business type.
    df = df.dropna(
        subset=[
            "localarea",
            "businesstype",
        ]
    ).copy()

    # clean extra spaces from the text columns.
    df["localarea"] = (
        df["localarea"]
        .astype("string")
        .str.strip()
    )

    df["businesstype"] = (
        df["businesstype"]
        .astype("string")
        .str.strip()
    )

    # remove any blank strings that remain after cleaning.
    df = df[
        (df["localarea"] != "")
        & (df["businesstype"] != "")
    ].copy()

    # convert the coordinates into numeric values just in case any invalid
    # text values entered the columns.
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    ).copy()

    # recreate the same broad industry groups used in part a.
    # business types not included in the mapping become other.
    df["business_group"] = (
        df["businesstype"]
        .map(business_mapping)
        .fillna("Other")
    )

    return df[
        [
            "localarea",
            "businesstype",
            "business_group",
            "latitude",
            "longitude",
        ]
    ].copy()


# ---------------------------------------------------------
# check that the file exists


if not DATA_PATH.exists():
    st.error(
        "i could not find `business-licences.geojson`. "
        "place it in the same folder as `app.py`."
    )
    st.stop()


# ---------------------------------------------------------
# load the data


try:

    with st.spinner(
        "loading and cleaning the original business licence dataset..."
    ):
        df = load_and_clean_data(DATA_PATH)

except Exception as error:

    st.error(
        f"the dataset could not be loaded: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# app introduction


st.title("Vancouver Business Similarity Explorer")

st.write(
    """
    this app explores which vancouver neighbourhoods have similar business
    compositions. unlike part a, where each row represented one individual
    business, part b changes the unit of analysis so that each row represents
    one neighbourhood.
    """
)

st.info(
    """
    use the controls below to change the minimum number of businesses and
    the value of k. the clustering results, pca plot, map, and cluster
    profiles will update automatically.
    """
)


# =========================================================
# B1: AREA-LEVEL FEATURE ENGINEERING
# =========================================================

st.header("B1: Area-Level Feature Engineering")


# ---------------------------------------------------------
# step 1: choose the area unit
# ---------------------------------------------------------

st.subheader("Step 1: Choose the Unit of Analysis")

st.write(
    """
    i selected `localarea` as the unit of analysis because it provides
    readable names for vancouver neighbourhoods and has much better coverage
    than the postal-code column. using neighbourhood names also makes the
    final clustering results easier for me to understand and explain.
    """
)


# ---------------------------------------------------------
# step 2: count businesses
# ---------------------------------------------------------

st.subheader("Step 2: Count Businesses in Each Neighbourhood")

st.write(
    """
    before creating the area-level feature matrix, i counted the number of
    businesses in every neighbourhood. this helps identify thin areas that
    may not contain enough businesses to produce reliable percentages.
    """
)

business_counts = (
    df["localarea"]
    .value_counts()
    .rename_axis("Neighbourhood")
    .reset_index(name="Business Count")
)

count_col1, count_col2, count_col3 = st.columns(3)

with count_col1:
    st.metric(
        "Businesses After Cleaning",
        f"{len(df):,}",
    )

with count_col2:
    st.metric(
        "Neighbourhoods Found",
        df["localarea"].nunique(),
    )

with count_col3:
    st.metric(
        "Business Groups",
        df["business_group"].nunique(),
    )

with st.expander(
    "View business counts before the threshold"
):
    st.dataframe(
        business_counts,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# step 3: remove thin areas
# ---------------------------------------------------------

st.subheader("Step 3: Remove Thin Neighbourhoods")

st.write(
    """
    neighbourhoods with very few businesses can produce unstable percentage
    values. for example, if an area only had five businesses, one business
    would represent 20% of the entire neighbourhood.

    i used 50 businesses as the default threshold because it reduces the
    influence of individual records while still keeping the main vancouver
    neighbourhoods. i also made it interactive so the effect of the threshold
    can be explored.
    """
)

minimum_businesses = st.slider(
    "Minimum businesses required per neighbourhood",
    min_value=25,
    max_value=500,
    value=50,
    step=25,
)

eligible_areas = business_counts.loc[
    business_counts["Business Count"] >= minimum_businesses,
    "Neighbourhood",
]

filtered_df = df[
    df["localarea"].isin(eligible_areas)
].copy()

threshold_table = business_counts.copy()

threshold_table["Status"] = threshold_table[
    "Business Count"
].apply(
    lambda count: (
        "Included"
        if count >= minimum_businesses
        else "Excluded"
    )
)

included_neighbourhoods = (
    filtered_df["localarea"].nunique()
)

excluded_neighbourhoods = (
    df["localarea"].nunique()
    - included_neighbourhoods
)

threshold_col1, threshold_col2, threshold_col3 = st.columns(3)

with threshold_col1:
    st.metric(
        "Minimum Business Count",
        minimum_businesses,
    )

with threshold_col2:
    st.metric(
        "Included Neighbourhoods",
        included_neighbourhoods,
    )

with threshold_col3:
    st.metric(
        "Excluded Neighbourhoods",
        excluded_neighbourhoods,
    )

with st.expander("View threshold results"):
    st.dataframe(
        threshold_table,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# step 4: create the composition matrix
# ---------------------------------------------------------

st.subheader("Step 4: Create the Area-Level Feature Matrix")

st.write(
    """
    the matrix below changes the dataset from one row per business to one row
    per neighbourhood. each column represents one of the broad business
    groups that i created during part a.

    each value shows the percentage of that neighbourhood's businesses
    belonging to the group. i used percentages instead of raw counts so
    neighbourhoods with very different total business counts can still be
    compared fairly.
    """
)

composition_matrix = (
    pd.crosstab(
        filtered_df["localarea"],
        filtered_df["business_group"],
        normalize="index",
    )
    * 100
)

composition_matrix.index.name = "Neighbourhood"
composition_matrix.columns.name = "Business Group"

if composition_matrix.shape[0] < 3:
    st.error(
        "too few neighbourhoods remain for clustering. "
        "reduce the minimum business threshold."
    )
    st.stop()

matrix_col1, matrix_col2 = st.columns(2)

with matrix_col1:
    st.metric(
        "Rows: Neighbourhoods",
        composition_matrix.shape[0],
    )

with matrix_col2:
    st.metric(
        "Columns: Business Groups",
        composition_matrix.shape[1],
    )

st.dataframe(
    composition_matrix.round(2),
    use_container_width=True,
    height=450,
)


# ---------------------------------------------------------
# step 5: validate the matrix
# ---------------------------------------------------------

st.subheader("Step 5: Validate the Feature Matrix")

st.write(
    """
    because the feature matrix is normalized by row, all business-group
    percentages in each neighbourhood should total approximately 100%.
    """
)

row_totals = composition_matrix.sum(axis=1)

validation_table = pd.DataFrame(
    {
        "Neighbourhood": row_totals.index,
        "Total Percentage": row_totals.values,
    }
)

rows_are_valid = (
    (row_totals - 100).abs() < 0.01
).all()

if rows_are_valid:

    st.success(
        "validation passed. every neighbourhood totals approximately 100%."
    )

else:

    st.warning(
        "some neighbourhood rows do not total approximately 100%."
    )

with st.expander("View validation table"):
    st.dataframe(
        validation_table.round(2),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# b1 summary
# ---------------------------------------------------------

st.subheader("B1 Summary")

st.write(
    f"""
    i selected localarea as the area unit because it provides readable
    neighbourhood names and strong coverage. after applying the minimum
    threshold of {minimum_businesses} businesses,
    {composition_matrix.shape[0]} neighbourhoods remained.

    the final area-level matrix contains
    {composition_matrix.shape[1]} business-group features. each value
    represents the percentage of businesses belonging to that group within
    a neighbourhood. the validation confirmed that each neighbourhood row
    totals approximately 100%.
    """
)


# =========================================================
# B2: INTERACTIVE K-MEANS
# =========================================================

st.header("B2: Interactive K-Means Clustering")

st.write(
    """
    in this section, k-means compares neighbourhoods using the business-group
    percentage matrix created in b1. neighbourhoods with similar business
    compositions are assigned to the same cluster.
    """
)


# ---------------------------------------------------------
# select the value of k
# ---------------------------------------------------------

st.subheader("Step 1: Select the Number of Clusters")

st.write(
    """
    k represents the number of neighbourhood groups created by k-means.
    a smaller k creates broader groups, while a larger k creates more
    detailed groups.
    """
)

maximum_k = min(
    8,
    composition_matrix.shape[0] - 1,
)

selected_k = st.slider(
    "Select K",
    min_value=2,
    max_value=maximum_k,
    value=min(4, maximum_k),
    step=1,
)

st.write(
    f"the current model is grouping the neighbourhoods into "
    f"**{selected_k} clusters**."
)


# ---------------------------------------------------------
# run k-means
# ---------------------------------------------------------

st.subheader("Step 2: Run K-Means")

st.write(
    """
    i used a random state of 42 so the results are reproducible. i also used
    `n_init = 10`, which runs k-means with multiple starting centroid
    positions and reduces the chance of keeping a poor result.
    """
)

kmeans_model = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10,
)

cluster_labels = kmeans_model.fit_predict(
    composition_matrix
)

cluster_results = pd.DataFrame(
    {
        "Neighbourhood": composition_matrix.index,
        "Cluster": cluster_labels + 1,
    }
)

# i added 1 to the labels because clusters 1, 2, 3 are easier to read
# than clusters 0, 1, 2. this does not change the model.
st.dataframe(
    cluster_results.sort_values(
        by=[
            "Cluster",
            "Neighbourhood",
        ]
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# pca visualization
# ---------------------------------------------------------

st.subheader("Step 3: PCA Scatter Plot")

st.write(
    """
    the business composition matrix contains several features, so it cannot
    be directly displayed on a two-dimensional graph. pca reduces these
    features into two principal components while keeping as much of the
    original variation as possible.

    each point below represents one neighbourhood. neighbourhoods positioned
    closer together have more similar business-group percentages.
    """
)

pca_model = PCA(
    n_components=2
)

pca_coordinates = pca_model.fit_transform(
    composition_matrix
)

explained_variance = (
    pca_model.explained_variance_ratio_
    * 100
)

pca_results = pd.DataFrame(
    {
        "Neighbourhood": composition_matrix.index,
        "PC1": pca_coordinates[:, 0],
        "PC2": pca_coordinates[:, 1],
        "Cluster": (cluster_labels + 1).astype(str),
    }
)

app_colours = [
    "#D7B8FF",
    "#F4B8D7",
    "#91D7E3",
    "#F6C177",
    "#9CCFD8",
    "#EB6F92",
    "#C4A7E7",
    "#A8D8B9",
]

pca_figure = px.scatter(
    pca_results,
    x="PC1",
    y="PC2",
    color="Cluster",
    hover_name="Neighbourhood",
    title=(
        f"Neighbourhood Business Similarity with K = {selected_k}"
    ),
    labels={
        "PC1": (
            f"PC1 ({explained_variance[0]:.1f}% explained variance)"
        ),
        "PC2": (
            f"PC2 ({explained_variance[1]:.1f}% explained variance)"
        ),
    },
    color_discrete_sequence=app_colours,
)

pca_figure.update_traces(
    marker={
        "size": 14,
        "line": {
            "width": 1,
            "color": "#171421",
        },
    }
)

pca_figure.update_layout(
    paper_bgcolor="#171421",
    plot_bgcolor="#171421",
    font_color="#F5F1FF",
    title_font_color="#D7B8FF",
)

pca_figure.update_xaxes(
    gridcolor="rgba(245, 241, 255, 0.15)",
    zerolinecolor="rgba(245, 241, 255, 0.30)",
)

pca_figure.update_yaxes(
    gridcolor="rgba(245, 241, 255, 0.15)",
    zerolinecolor="rgba(245, 241, 255, 0.30)",
)

st.plotly_chart(
    pca_figure,
    use_container_width=True,
)

variance_col1, variance_col2, variance_col3 = st.columns(3)

with variance_col1:
    st.metric(
        "Selected K",
        selected_k,
    )

with variance_col2:
    st.metric(
        "PC1 Explained Variance",
        f"{explained_variance[0]:.2f}%",
    )

with variance_col3:
    st.metric(
        "PC2 Explained Variance",
        f"{explained_variance[1]:.2f}%",
    )


# ---------------------------------------------------------
# b2 summary
# ---------------------------------------------------------

st.subheader("B2 Summary")

st.write(
    f"""
    the current k-means model groups
    {composition_matrix.shape[0]} neighbourhoods into
    {selected_k} clusters using their business-group percentage profiles.

    the pca visualization displays
    {explained_variance[0] + explained_variance[1]:.2f}% of the total
    variation using the first two principal components. changing the k
    slider automatically recalculates the cluster assignments and updates
    the visualization.
    """
)


# =========================================================
# B3: GEOGRAPHIC VISUALIZATION
# =========================================================

st.header("B3: Geographic Visualization")

st.write(
    """
    this section places every included neighbourhood on a map. each point is
    located at the average latitude and longitude of its businesses. point
    colour shows the k-means cluster, while point size shows the number of
    businesses in that neighbourhood.
    """
)


# ---------------------------------------------------------
# calculate neighbourhood centre points
# ---------------------------------------------------------

st.subheader("Step 1: Calculate Neighbourhood Centre Points")

st.write(
    """
    to represent each neighbourhood with one point, i calculated the average
    latitude and longitude of all businesses located in the neighbourhood.
    this provides a simple representative location for the map.
    """
)

area_centres = (
    filtered_df
    .groupby("localarea")
    .agg(
        Latitude=("latitude", "mean"),
        Longitude=("longitude", "mean"),
        Business_Count=("localarea", "size"),
    )
    .reset_index()
)


# ---------------------------------------------------------
# combine map information
# ---------------------------------------------------------

st.subheader("Step 2: Prepare the Map Data")

map_data = area_centres.merge(
    cluster_results,
    left_on="localarea",
    right_on="Neighbourhood",
    how="inner",
)

map_data["Cluster"] = (
    map_data["Cluster"]
    .astype(str)
)

map_centre = {
    "lat": map_data["Latitude"].mean(),
    "lon": map_data["Longitude"].mean(),
}

with st.expander("View the map data"):
    st.dataframe(
        map_data[
            [
                "localarea",
                "Latitude",
                "Longitude",
                "Business_Count",
                "Cluster",
            ]
        ]
        .rename(
            columns={
                "localarea": "Neighbourhood",
                "Business_Count": "Business Count",
            }
        )
        .sort_values(
            by=[
                "Cluster",
                "Neighbourhood",
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# create the map
# ---------------------------------------------------------

st.subheader("Step 3: Interactive Cluster Map")

st.write(
    """
    larger points represent neighbourhoods containing more businesses.
    neighbourhoods with the same colour belong to the same k-means cluster.
    the map updates whenever the selected value of k changes.
    """
)

cluster_map = px.scatter_map(
    map_data,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    size="Business_Count",
    hover_name="localarea",
    hover_data={
        "Latitude": ":.4f",
        "Longitude": ":.4f",
        "Business_Count": True,
        "Cluster": True,
    },
    center=map_centre,
    zoom=10,
    size_max=40,
    title=(
        f"Vancouver Neighbourhood Clusters with K = {selected_k}"
    ),
    color_discrete_sequence=app_colours,
    map_style="carto-darkmatter",
)

cluster_map.update_layout(
    paper_bgcolor="#171421",
    font_color="#F5F1FF",
    title_font_color="#D7B8FF",
    margin={
        "l": 0,
        "r": 0,
        "t": 60,
        "b": 0,
    },
    height=650,
)

st.plotly_chart(
    cluster_map,
    use_container_width=True,
)


# ---------------------------------------------------------
# b3 summary
# ---------------------------------------------------------

st.subheader("B3 Summary")

largest_area_row = (
    map_data
    .sort_values(
        "Business_Count",
        ascending=False,
    )
    .iloc[0]
)

st.write(
    f"""
    the map displays {len(map_data)} neighbourhood centre points.
    **{largest_area_row["localarea"]}** has the largest business count
    among the included neighbourhoods, with
    **{int(largest_area_row["Business_Count"]):,} businesses**.

    the map shows whether neighbourhoods with similar business compositions
    are also geographically close. some cluster members may appear near one
    another, while others may be spread across different parts of vancouver.
    """
)


# =========================================================
# B4: CLUSTER MEMBERSHIP AND INTERPRETATION
# =========================================================

st.header("B4: Cluster Membership and Interpretation")

st.write(
    """
    this section identifies the exact neighbourhoods assigned to every
    cluster. it also profiles the clusters using their strongest average
    business-group percentages.

    these results update automatically whenever k changes.
    """
)

clustered_composition = composition_matrix.copy()

clustered_composition["Cluster"] = (
    cluster_labels + 1
)


# ---------------------------------------------------------
# neighbourhood membership
# ---------------------------------------------------------

st.subheader("Step 1: Neighbourhoods in Each Cluster")

for cluster_number in sorted(
    clustered_composition["Cluster"].unique()
):

    cluster_neighbourhoods = (
        clustered_composition[
            clustered_composition["Cluster"]
            == cluster_number
        ]
        .index
        .tolist()
    )

    st.markdown(
        f"### Cluster {cluster_number} "
        f"({len(cluster_neighbourhoods)} neighbourhoods)"
    )

    st.write(
        ", ".join(cluster_neighbourhoods)
    )


# ---------------------------------------------------------
# membership table
# ---------------------------------------------------------

st.subheader("Cluster Membership Table")

membership_table = (
    cluster_results
    .sort_values(
        by=[
            "Cluster",
            "Neighbourhood",
        ]
    )
    .reset_index(drop=True)
)

st.dataframe(
    membership_table,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# profile the clusters
# ---------------------------------------------------------

st.subheader("Step 2: Profile Each Cluster")

st.write(
    """
    to help interpret the clusters, i calculated the average percentage of
    each business group across all neighbourhoods belonging to the same
    cluster.

    the business groups with the highest average percentages help explain
    what the neighbourhoods in each cluster have in common.
    """
)

cluster_profiles = (
    clustered_composition
    .groupby("Cluster")
    .mean()
)

for cluster_number in cluster_profiles.index:

    cluster_neighbourhoods = (
        clustered_composition[
            clustered_composition["Cluster"]
            == cluster_number
        ]
        .index
        .tolist()
    )

    top_groups = (
        cluster_profiles
        .loc[cluster_number]
        .sort_values(ascending=False)
        .head(5)
    )

    top_groups_table = pd.DataFrame(
        {
            "Business Group": top_groups.index,
            "Average Percentage": top_groups.values,
        }
    )

    with st.expander(
        f"Cluster {cluster_number}: "
        f"{len(cluster_neighbourhoods)} neighbourhoods"
    ):

        st.markdown("**Neighbourhoods:**")

        st.write(
            ", ".join(cluster_neighbourhoods)
        )

        st.markdown(
            "**Five strongest average business groups:**"
        )

        st.dataframe(
            top_groups_table.round(2),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# automatically create a current summary
# ---------------------------------------------------------

st.subheader("Step 3: Current Cluster Summary")

st.write(
    """
    the summary below is automatically generated from the current k value.
    it lists the actual neighbourhood membership and the three strongest
    average business groups in every cluster.
    """
)

for cluster_number in cluster_profiles.index:

    cluster_neighbourhoods = (
        clustered_composition[
            clustered_composition["Cluster"]
            == cluster_number
        ]
        .index
        .tolist()
    )

    top_three_groups = (
        cluster_profiles
        .loc[cluster_number]
        .sort_values(ascending=False)
        .head(3)
    )

    group_descriptions = [
        f"{group} ({percentage:.2f}%)"
        for group, percentage in top_three_groups.items()
    ]

    neighbourhood_text = ", ".join(
        cluster_neighbourhoods
    )

    group_text = ", ".join(
        group_descriptions
    )

    st.write(
        f"""
        **Cluster {cluster_number}** contains
        **{len(cluster_neighbourhoods)} neighbourhoods**:
        {neighbourhood_text}.

        its three strongest average business-group features are:
        {group_text}.
        """
    )


# ---------------------------------------------------------
# interpretation
# ---------------------------------------------------------

st.subheader("Step 4: Interpretation")

st.write(
    """
    the clusters represent neighbourhoods with similar business compositions,
    rather than neighbourhoods that are necessarily beside one another.

    the pca graph helps show which neighbourhoods have similar percentage
    profiles, while the geographic map shows their real locations. when
    neighbourhoods from different parts of vancouver appear in the same
    cluster, this suggests that business similarity can cross geographic
    boundaries.

    the cluster profiles also show which broad industries contribute most
    strongly to each grouping. however, the category called `Other` may be
    large because it includes all original business types that were not part
    of the main mapping created in part a.
    """
)


# =========================================================
# B5: OPTIONAL REFLECTION
# =========================================================

st.header("B5: Reflection Questions")

st.subheader(
    "1. Do business-similar areas always appear geographically close?"
)

st.write(
    """
    the results suggest that business similarity does not always match
    geographic proximity. some neighbourhoods in the same cluster may be
    close together, but others can be located in different parts of
    vancouver. this happens because k-means uses business-group percentages
    rather than latitude and longitude when creating these clusters.
    """
)

st.subheader(
    '2. How does Part B compare with the Part A "Industry" clustering?'
)

st.write(
    """
    part a and part b use related information but answer different questions.
    in part a, every row represented one business, and businesses were
    compared using employee count, licence fee, lifecycle features, and
    industry.

    in part b, every row represents one neighbourhood. neighbourhoods are
    compared using the percentage distribution of business groups found in
    each area. part a therefore explains similarities between individual
    businesses, while part b explains similarities between entire
    neighbourhood business profiles.
    """
)

st.subheader(
    "3. What should be checked before writing the final reflection?"
)

st.write(
    """
    after selecting the final value of k, i would compare the actual cluster
    membership, dominant business groups, pca positions, and geographic map.
    i would then identify one neighbourhood pairing that surprised me and
    use the business-group percentages to explain why k-means may have placed
    them together.
    """
)


# =========================================================
# PROCESS ACKNOWLEDGEMENT
# =========================================================

st.header("Process Acknowledgement")

st.write(
    """
    throughout this assignment, i regularly referred back to the lab
    tutorials, example colab notebooks, and the guidance provided by the
    professor and tas, especially during the feature engineering process.
    i also frequently used ai in between steps to check my understanding,
    clarify concepts, help with syntax and debugging, and make sure i was
    moving in the right direction.
    """
)

st.caption(
    "data source: city of vancouver business licences open data."
)