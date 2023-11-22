# MIT License
#
# Copyright (c) 2023 Panagiotis Anagnostou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
    This script reads the AI-GA dataset and splits it into train and test sets.
    The default split is 80% train and 20% test. The script also has the option
    to clean the text of the abstracts. The cleaning process is described in the
    clean_text function and is based on the cleaning process of the AI-GA
    dataset. The data are separated by the title of the paper, so that the same
    paper is original and generated abstracts to be in the same set. Finally,
    the train and test sets are saved as csv files in the same directory, with
    the name "train.csv" and "test.csv" respectively.
"""

from tqdm import tqdm

import pandas as pd
import re
import stopwordsiso as stopwords
import sys


def clean_text(abstract):
    """
    Function for basic text cleaning of the AI-GA dataset. The cleaning we
    followed steps are:

    1. Remove HTML tags
    2. Remove -
    3. Remove punctuation
    4. Convert to lowercase
    5. Remove numbers
    6. Remove stop words
    7. Remove extra whitespace
    8. Remove study and paper keywords. These keywords are not useful for the
    classification task, because we classify scientific papers.

    Parameters
    ----------
    abstract : str
        The text to be cleaned.

    Returns
    -------
    text : str
        The cleaned text.

    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", abstract)

    # Remove -
    text = re.sub(r"-", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d", "", text)

    # Remove stop words
    text = " ".join(
        [word for word in text.split() if word not in stopwords.stopwords("en")]
    )

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"study", " ", text)
    text = re.sub(r"paper", " ", text)

    return text


if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv("/Users/dustinhayes/Desktop/DEEP LEARNING FINAL PROJECT/gpt-2-output-dataset/Deep Learning Final Project Data/ai-ga-dataset.csv", engine="c")

    # Initialize the command line arguments
    clean = False
    random_state = 42
    # Acceptable command line arguments
    command_line_args = ["clean", "random_state"]

    # Check if the user provided any command line arguments and if they have valid names
    if len(sys.argv) > 3:
        raise ValueError("Too many arguments")
    elif len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.split("=")[0] not in command_line_args:
                raise ValueError(f"Invalid argument: {arg.split('=')[0]}")
            else:
                if arg.split("=")[0] == "clean":
                    clean = arg.split("=")[1]
                elif arg.split("=")[0] == "random_state":
                    random_state = arg.split("=")[1]

    # Validate the clean and random_state values
    # ... [rest of your existing code for validating 'clean' and 'random_state'] ...

    # Clean the text if the user provided the clean command line argument
    if clean:
        print("Cleaning text...")
        tqdm.pandas()
        df["abstract"] = df["abstract"].progress_apply(clean_text)

    original = df[df.label == 0]
    ai = df[df.label == 1]

    print("Splitting data into train, validation, and test sets...")
    print("The train set contains 60%, the validation set contains 20%, and the test set contains 20% of the data")

    # Splitting logic
    train_percent = 0.6
    validation_percent = 0.2

    # Split AI and original data separately to maintain the same titles in each set
    ai_train = ai.sample(frac=train_percent, random_state=random_state)
    ai_remain = ai.drop(ai_train.index)
    ai_validation = ai_remain.sample(frac=validation_percent / (1 - train_percent), random_state=random_state)
    ai_test = ai_remain.drop(ai_validation.index)

    original_train = original[original.title.isin(ai_train.title)]
    original_remain = original.drop(original_train.index)
    original_validation = original_remain[original_remain.title.isin(ai_validation.title)]
    original_test = original_remain.drop(original_validation.index)

    # Combine the sets
    train = pd.concat([original_train, ai_train])
    validation = pd.concat([original_validation, ai_validation])
    test = pd.concat([original_test, ai_test])

    # Shuffle the datasets and reset index
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"The train set shape is {train.shape}")
    print(f"The validation set shape is {validation.shape}")
    print(f"The test set shape is {test.shape}")

    # Save the datasets as CSV files
    print("Saving train, validation, and test sets...")
    train.to_csv("train.csv", index=False)
    validation.to_csv("validation.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("Done!!!")
