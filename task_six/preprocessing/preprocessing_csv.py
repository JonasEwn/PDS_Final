import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Preprocessing_CSV_Seniority():
    def __init__(self, file_path):
        self.file_path = file_path
        self.df: pd.DataFrame = None
        self.X = None
        self.y = None
        self.y_str = None
        self.label_encoder = LabelEncoder()

        self.read_csv()

    def clean_text(self, text: str):
        """
        Removes - and / and replaces with <space>
        """
        text = text.lower().strip().replace("-", " ").replace("/", " ")
        return text

    def read_csv(self):
        """
        Reads CSV file and saves them in class properties
        """
        self.df = pd.read_csv(self.file_path)

        # Check if correct file is given
        requiered_cols = {"text", "label"}
        if not requiered_cols.issubset(self.df.columns):
            raise ValueError(
                f"Wrong file mate :("
            )

        self.df["text"] = self.df["text"].astype(str).apply(self.clean_text)
        self.y_str = self.df["label"].astype(str)
        self.y = self.label_encoder.fit_transform(self.y_str)
        self.X = self.df["text"]

    def labels(self):
        """
        Just quick check can be removed
        """
        return {
            (i, label) for i, label in enumerate(self.label_encoder.classes_)
        }

