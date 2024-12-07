Drive Link: https://drive.google.com/drive/folders/1TwWVFfHZ6RUHhiW_g_rtyZo2k7PTZZF-?usp=sharing

From the above drive link download myenv.zip, unzip and place it in the main folder structure.

Next, download the classifier_dataset.zip, unzip it and rename it as dataset place it under comment_classifier folder.

Next, download the trained_models.zip, unzip it and place it under comment_classifier folder.

Next, download the generator_dataset.zip, unzip it and rename it as dataset place it under comment_generator folder.

Next, download the saved_model.zip, unzip it and place it under comment_generator folder.

Training and executing comment_classifier:

Change Directory: "cd src/comment_classifier"
For training, run "python train.py"
For prediction, run "python prediction.py"

Training and executing comment_generator:

Change Directory: "cd src/comment_generator"
For training, run "python training.py"
For prediction, run "python testing.py"
