# transform_image_to_text

The purpose of this program is to extract text from a photo into a text file for exemple When you have a course in handout and you quickly want to be able to modify it on your computer.

At first the program "detection_word.py" detects the presence of words thanks to a mask and extract them.
Then the program "detection_letter.py" detects the letters from the words detected and extract them.

The program "dataset_font.py" create a dataset with all lowercase and uppercase letters in multiple fonts.
The program "CNN_letter.ipynb" train a cnn to recognize a photo of a letter.

