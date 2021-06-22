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

    @staticmethod
    def clean_people_interest(dataframe):
        df = dataframe
        a=[]
        for col in df:
            if df[col].nunique()==1:
                a.append(col)
            else:
                pass
        df[a] = df[a].fillna(0)

        threshold=0.1*6340

        b=[]
        for col in df:
            if df[col].isnull().sum()>threshold:
                b.append(col)
            else:
                pass
        df=df.drop(b,axis=1)
        c=[]

        for col in df:
            if (df[col].isnull().sum()<threshold) & (df[col].isnull().sum()>0):
                c.append(col)
            else:
                pass
        df = df.dropna(subset=c)

        df['group'] = df['group'].map({'C':0,'P':1,'R':2,'I':3})

        return df