import pandas as pd
import numpy as np

mainstream = [1, 3, 5, 8, 10, 12, 14]     # Create from Data Dictionary main stream
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
        missing_values = [int(value) if (value!='X' and value!='XX' and value!='') else value for value in missing_values]
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
    return  df.isna().sum(axis=1)

def split_dataset(df):
    upper_threshold = df[df.isna().sum(axis=1) > 20]
    lower_threshold = df[df.isna().sum(axis=1) <= 20]

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

    # Use list comprehension to extract the wealth and life stage components from 'CAMEO_INTL_2015' and assign them
    # Extract first digit, convert to int and assigin to Weatlh, Extract 2 digit, convert to int and assign ti to LeftAte
    df.loc[filtered, ['WEALTH', 'LIFESTAGE']] = [
        (int(str(x)[0]), int(str(x)[1])) for x in df.loc[filtered, 'CAMEO_INTL_2015']
    ]

    return df

if __name__ == '__main__':
    semi = ';'

    azdias_demogrh_df = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=semi, na_values=['NaN', '[]'] )
    copy_of_azdias_demogrh_df = azdias_demogrh_df.copy()

    # Load in the feature summary file.
    feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=semi)

    my_df2 = replace_missing_data(feat_info, azdias_demogrh_df)
    # print(my_df2)
    # cat, bin, multi = find_categories(feat_info, azdias_demogrh_df)
    # print(multi)
    # upper, lower = split_dataset(azdias_demogrh_df)
    # print(upper.shape)
    # print(lower.shape)
    # row_missing_data = find_row_with_missing_data(azdias_demogrh_df)
    # print(len(row_missing_data.isna()))
    print(my_df2.head())

    # print(find_columns_with_missing_data(azdias_demogrh_df))