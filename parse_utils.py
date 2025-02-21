import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

mainstream = [1, 3, 5, 8, 10, 12, 14]    # Create from Data Dictionary main stream
avantgarde = [2, 4, 6, 7, 9, 11, 13, 15]  # Create from Data Dictionary avantgarde
decade = pd.Series([40, 40, 50, 50, 60, 60, 60, 70, 70, 80, 80, 80, 80, 90, 90],
                   index = [ n + 1 for n in range(15)])   #create decade, and assign labels

music_styles = {
    "mainstream": mainstream,
    "avantgarde": avantgarde
}

def search_music_styles(number):
    if number in music_styles["mainstream"]:
        return 1
    elif number in music_styles["avantgarde"]:
        return 0
    else:
        return number


def find_categories(df1, df2):
    categorical_dtypes_df = df1[df1['type']=='categorical']['attribute'].values
    categorical_dtypes_df = [x for x in categorical_dtypes_df if x in df2.columns]
    binary_dtypes= [x for x in categorical_dtypes_df if len(df2[x].unique()) == 2]
    multi_level_dtypes = [x for x in categorical_dtypes_df if len(df2[x].unique()) > 2]

    return categorical_dtypes_df, binary_dtypes,  multi_level_dtypes



def replace_missing_data(df1, df2):
    for attribute, miss_index_value in zip(df1['attribute'], df1['missing_or_unknown']):
        missing_values = miss_index_value.strip('[]').split(',')
        missing_values = [int(value) if (value!='X' and value!='XX' and value!='')
                          else value for value in missing_values]
        if missing_values != ['']:
            df2[attribute] = df2[attribute].replace(missing_values, np.nan)

    return df2


def find_columns_with_missing_data(df):
    column_count_of_missing_df = pd.DataFrame({
        'column_name': df.isna().sum().index,
        'null_count': df.isna().sum().values,
        'percentage': df.isna().sum() / df.shape[0] * 100
    })

    return column_count_of_missing_df.sort_values(by='percentage', ascending=False)


def find_row_with_missing_data(df):
    """
    :param df: summary dataframe
    :return: rows that have missing data in summary dataframe
    """
    return  df.isna().sum(axis=1)


def split_dataset(df, df_missing_rows, threshold=20):
    """
    :param df_missing_rows:
    :param threshold:
    :param df: summary dataframe
    :return:
    """
    upper_threshold = df[df_missing_rows > threshold]
    lower_threshold = df[df_missing_rows <= threshold]

    return upper_threshold, lower_threshold


def encoded_rows_with_dummies(df, categorical_columns):
    """
    :param categorical_columns:
    :param df: - rows lt 20 subset threshold
    :param categorical_columns: as a dict
    :return: categorical_columns
    """
    for column in categorical_columns:
        found_row = df[column][df[column].notnull()]
        dummies = pd.get_dummies(found_row, prefix=column, drop_first=False)
        df = df.join(dummies)
        df.drop([column], axis=1, inplace=True)

    return df


def create_feature_rows(df):
    """
    :param df: data set with low threshold rows < 20
    :return: df with two engineer rows
    """
    filtered = df['CAMEO_INTL_2015'].notnull()  # Filter rows where 'CAMEO_INTL_2015' is not null

    # Use list comprehension to extract the wealth and life stage components
    # from 'CAMEO_INTL_2015' and assign them
    # Extract first digit, convert to int and assigin to Weatlh, Extract 2 digit,
    # convert to int and assign ti to LeftStage
    df.loc[filtered, ['WEALTH', 'LIFESTAGE']] = [
        (int(str(x)[0]), int(str(x)[1])) for x in df.loc[filtered, 'CAMEO_INTL_2015']
    ]

    return df


def find_how_much_data_missing(df):
    return [df.index[i] for i in range(df.shape[0]) if df.iloc[i] == 0]


def find_columns_to_drop_over_threshold(df, threshold=20):
    """
    Combines filtering of columns with missing values over the given threshold
    and the creation of a list of column names to drop into a single function.

    Parameters:
    column_count_of_missing_df (pd.DataFrame): A DataFrame containing columns 'column_name' and 'percentage'.
    threshold (float): The minimum percentage of missing values to filter columns. Default is 20.

    Returns:
    tuple: (filtered DataFrame, list of column names to drop)
    """
    # Filter the DataFrame and generate the list in one operation
    columns_with_over_threshold = df[df['percentage'] > threshold]
    columns_to_drop_list = columns_with_over_threshold['column_name'].tolist()

    # Return both in a tuple
    return columns_with_over_threshold, columns_to_drop_list

def find_rows_with_null_data_in_summary(df):
    """

    :param df: summary df
    :return: rows missing from summary that are missing in each column
    """
    df_rows_missing_from_summary = (df.isna().sum()/df.shape[0]).sort_values(ascending=False) * 100

    rows_missing = [df_rows_missing_from_summary.index[i]
                        for i in range(df_rows_missing_from_summary.shape[0])
                            if df_rows_missing_from_summary.iloc[i] == 0
                    ]

    return rows_missing


def find_mixed_data_type_rows(df, df_categorical_columns):
    """

    :param df: feature info
    :param df_categorical_columns:
    :return:
    """
    mixed_data_type_rows = df[df['type'] == 'mixed']['attribute'].values

    mixed_data_type_rows = [n for n in mixed_data_type_rows if n in df_categorical_columns]
    return mixed_data_type_rows


def first_pc_analysis(df, pca, ncomp=0):
    """
    Maps the weights of the first principal component to the corresponding feature names,
    sorts the weights in descending order, and prints the linked values.

    :param df: DataFrame, feature matrix from which PCA is computed
    :param pca: PCA, fitted PCA object containing components
    :return: DataFrame, sorted feature names and their corresponding weights
    """
    # Step 1: Map feature indices to feature names
    feature_map = pd.Series(df.columns, index=range(len(df.columns)))

    # Step 2: Extract weights for the first principal component
    first_pc_weights = pca.components_[ncomp]

    # Step 3: Map weights to corresponding feature names
    feature_weights = [(feature_map[ix], weight) for ix, weight in enumerate(first_pc_weights)]

    # Step 4: Sort the feature weights by absolute value, in descending order
    sorted_feature_weights = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)

    # Step 5: Convert sorted weights to a DataFrame for easier analysis
    sorted_df = pd.DataFrame(sorted_feature_weights, columns=["Feature", "Weight"])

    return sorted_df


def check_null_types(df):
    """
     Count how many null values are associated with dtypes in a dataframe.
    :param df:
    :return: df
    """
    null_counts = df.isna().sum()
    # create dataframe to hold the counts by dtypes
    null_summary = pd.DataFrame({
        'null_count': null_counts,
        'dtype': df.dtypes
    })

    # group by dtype
    null_by_dtype = null_summary.groupby('dtype')['null_count'].sum()

    return null_by_dtype


if __name__ == '__main__':
    semi = ';'

    azdias_demogrh_df = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=semi, na_values=['NaN', '[]'] )

    # Load in the feature summary file.
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=semi)

    # summary_df =  replace_missing_data(feat_info, summary df)
    summary_df  = replace_missing_data(feat_info, azdias_demogrh_df)

    #Count the Columns create a panda dataframe containing, column, name, null count, percentages
    column_count_of_missing_df = find_columns_with_missing_data(summary_df)

    #Find columns over percent threshold and columns to drop from list, threshold is set at 20
    columns_over_threshold_percent, columns_to_drop =  find_columns_to_drop_over_threshold(column_count_of_missing_df, 30)

    #Find The missing rows and split summary returning upper and lower threshold
    rows_missing_from_summary = find_row_with_missing_data(summary_df)
    rows_threshold_upper, rows_threshold_lower = split_dataset(summary_df, rows_missing_from_summary)

    # Find the rows that are missing in summary df
    columns_missing_from_summary =  find_rows_with_null_data_in_summary(summary_df)

    #  Find the categories, return as tuple for processing
    cat, binary, multi = find_categories(feat_info, summary_df)

    #Lower threshold drop the rows where its CAMEO_DEU_215
    rows_threshold_lower.drop(index=rows_threshold_lower.loc[
                    rows_threshold_lower['CAMEO_DEU_2015'] == True].index, inplace=True, axis=1)


    # Encode with  dummy variables call encoded_rows_with_dummies
    rows_threshold_lower = encoded_rows_with_dummies(rows_threshold_lower, multi)

    # Feature Engineering on PRAEGENDE_JUGENDJAHRE creating Movement and Decade Columns
    rows_threshold_lower['MOVEMENT'] = rows_threshold_lower['PRAEGENDE_JUGENDJAHRE'].apply(search_music_styles)
    rows_threshold_lower['DECADE'] =  rows_threshold_lower['PRAEGENDE_JUGENDJAHRE'].map(decade)


    # Drop the columns from summary based off of columns_to_drop and reset index on summary
    summary_df.drop(columns=columns_to_drop, inplace=True, axis=1)
    summary_df.reset_index(drop=True, inplace=True)

    # FeatureEngineering on CAMEO_INTL_2015'
    rows_threshold_lower = create_feature_rows(rows_threshold_lower)

    # Find the mix data type rows
    mixed_data_type_rows = find_mixed_data_type_rows(feat_info, rows_threshold_lower)

    # Drop the mix data type rows  from  rows_threshold_lower
    rows_threshold_lower.drop(mixed_data_type_rows, inplace=True, axis=1)
    rows_threshold_lower.reset_index(drop=True, inplace=True)

    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    customer_data = pd.DataFrame(simple_imputer.fit_transform(rows_threshold_lower), columns=rows_threshold_lower.columns)

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(customer_data), columns=customer_data.columns)

    pca = PCA(n_components=30)
    # pca.fit(scaled_data)
    principal_components = pca.transform(scaled_data)
    print( principal_components)

    # print(find_columns_with_missing_data(azdias_demogrh_df))