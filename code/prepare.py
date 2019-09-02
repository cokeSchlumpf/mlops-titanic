import pandas as pd
import numpy as np

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

def extract_title(df):
    """
    Extract the title from the name field. We assume that a word ending with period is a title.
    Additionaly we combine some title to reduce the number titles.
    """
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Jonkheer', 'Dona'], 'Lady')
    df['Title'] = df['Title'].replace(['Capt', 'Don', 'Major', 'Sir'], 'Sir')
    
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df

def calculate_family_size(df):
    """
    Calculate the count of family members traveling with the person and set a flag whether a person is traveling alone or not.
    """
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

def fillna_most_frequent(df, column, column_new = None):
    """
    Gets the most frequent value of the column and replaces null-values with this value.
    """

    if column_new is None:
        column_new = column

    frequent = df[column].dropna().mode()[0]
    df[column_new] = df[column].fillna(frequent)
    return df

def fillna_median(df, column, column_new = None):
    """
    Fill null-values of a column with the median of the column.
    """

    if column_new is None:
        column_new = column

    median = df[column].dropna().median()
    df[column_new] = df[column].fillna(median)
    return df

def continuous_to_ordinal(df, groups, column, column_new = None):
    """
    Sorts continuous numerical value into groups.
    """

    if column_new is None:
        column_new = column

    df[column_new] = pd.qcut(df[column], groups, duplicates = 'drop')
    df = ordinal_to_numbers(df, column, column_new)
    return df

def ordinal_to_numbers(df, column, column_new = None):
    """
    Helper function to transform a column with ordinal values to a new column including a numeric index
    for the column.
    """

    if column_new is None:
        column_new = column

    values = df[column].unique()
    df[column_new] = df[column].map(lambda value: np.where(values == value)[0][0])
    return df

def prepare(df):
    df = extract_title(df)
    df = ordinal_to_numbers(df, 'Title')
    df = ordinal_to_numbers(df, 'Sex')
    df = fillna_most_frequent(df, 'Embarked')
    df = ordinal_to_numbers(df, 'Embarked')
    df = fillna_median(df, 'Fare')
    df = fillna_median(df, 'Age')
    df = continuous_to_ordinal(df, 4, 'Fare')
    df = continuous_to_ordinal(df, 5, 'Age')
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    df = calculate_family_size(df)

    return df

train = prepare(train)
test = prepare(test)

train.to_pickle('../data/train_prepared.pkl')
test.to_pickle('../data/test_prepared.pkl')