from sklearn.preprocessing import LabelEncoder


class Preprocessor():

    @staticmethod
    def breast_cancer_classes(df):
        dataframe = df
        for index, row in enumerate(dataframe['class']):
            if row == 2:
                dataframe.at[index, 'class'] = 0
            if row == 4:
                dataframe.at[index, 'class'] = 1
        dataframe = dataframe.dropna()
        return dataframe

    @staticmethod
    def glass_type_encode(type_column):
        le = LabelEncoder()
        le.fit(type_column)
        glass_type = le.transform(type_column)

        return glass_type