import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pulp


def get_data(fileID):
    """
    This function gets the data which we are trying to encode.
    :param fileID: Name of file which we are reading from as a string
    :return df: Dataframe which contains the data
    """
    df = pd.read_csv(fileID)
    return df


def clean_data(df, labels):
    """
    This function cleans the data which we are trying to encode.
    :param df: the data which we are trying to clean
    :param labels: the labels which we are trying use
    :return clean_df: Dataframe which contains the cleaned data
    """
    clean_df = df.loc[:, labels]
    clean_df = clean_df.dropna(axis = 0)

    return clean_df


def encode_names(df, labels, skill, gender, club):
    """
    This function encodes the labels with values.
    :param df: the data which we are trying to encode
    :param labels: the labels which we are trying use
    :param skill: the skill which we are trying to encode
    :param gender: the gender which we are trying to encode
    :param club: the club which we are trying to encode
    :return: encoded_df: Dataframe which contains the encoded data
    """
    # Create encoders
    skillencode = LabelEncoder()
    clubencode = LabelEncoder()
    genderencode = LabelEncoder()

    # Fit data to encoder
    skillencode.fit(skill)
    clubencode.fit(club)
    genderencode.fit(gender)
    genderencode.classes_ = np.array(gender)
    clubencode.classes_ = np.array(club)
    skillencode.classes_ = np.array(skill)

    # create copy of original data
    encoded_data = df.copy()
    for label in labels:
        if "Skill" in label:
            encoded_data[label] = skillencode.transform(encoded_data[label])
        if "Gender" in label:
            encoded_data[label] = genderencode.transform(encoded_data[label])
        if "Club" in label:
            encoded_data[label] = clubencode.transform(encoded_data[label])

    return encoded_data

def simply_gender(df,gender):
    """
    This function simplifies the gender label into a binary values.
    :param df: the dataframe we are editing
    :param gender: the gender label
    :return:
    """
    for index in df.loc[gender, :].index:
        if df.loc[gender, index] == 2:
            df.loc[gender, index] = 1
    return df

def simply_club(df,club):
    """
    This function simplifies the gender label into a binary values.
    :param df: the dataframe we are editing
    :param gender: the gender label
    :return:
    """
    for index in df.loc[club, :].index:
        if df.loc[club, index] == 2:
            df.loc[club, index] = 0
    return df


# These function are chatGPT Generated because it'll take too long for me to read the documentation
def remove_substring_from_df(df: pd.DataFrame, substring: str) -> pd.DataFrame:
    """
    Remove a specific substring from all string values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        substring (str): The substring to remove.

    Returns:
        pd.DataFrame: A new DataFrame with the substring removed from string values.
    """
    return df.applymap(lambda x: x.replace(substring, '') if isinstance(x, str) else x)

import pandas as pd

def replace_substring_in_df(df: pd.DataFrame, target: str, replacement: str) -> pd.DataFrame:
    """
    Replace a specific substring in all string values of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The substring to replace.
        replacement (str): The string to replace it with.

    Returns:
        pd.DataFrame: A new DataFrame with the substring replaced in string values.
    """
    return df.applymap(lambda x: x.replace(target, replacement) if isinstance(x, str) else x)

def count_exact_matches(df: pd.DataFrame, target) -> int:
    """
    Count how many times the target value appears exactly in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target: The value to count (can be string, int, etc.).

    Returns:
        int: Number of exact matches.
    """
    return (df == target).sum().sum()
