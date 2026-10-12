# IMDB-Genre-Classification
Multi-label genre classifier for IMDb titles. A SimpleRNN model predicts up to 25 genres from a plot description, and a Streamlit app lets you try it with a confidence threshold.
Note: the notebook’s test results are weak (micro F1 ≈ 0.01 for SimpleRNN), so the model mostly predicts Drama. You may want to say that in the README, or retrain before presenting it as working.
Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. 
Download all (.py, .ipynb, .pkl, .h5) files in a folder In imdb_genre_classification.py file, go to terminal and run streamlit To run streamlit, go to terminal of the .py file and enter       py -m streamlit run "file_location\imdb_genre_classification.py" 
For example, in my device I enter: 
py -m streamlit run "L:\Softwares\VSCode\Files\RNN\imdb_genre_classification\imdb_genre_classification.py" 
A localhost appears in your default web browser and finally you are about to predict the movie genre on the basis of your own description (story/ abstract) of a movie/ web-series.
