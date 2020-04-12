# LumberjackNet

This network uses the audio collected during the harvesting of pine trees to estimate the diameter of trunk. This is specially useful to model the volume of the tree which is important in the yield evaluation.

To reproduce this results, clone this repository, enter the project folder,create a Python 3.x virtual environment and run:

`pip install -r requirements.txt`

To download the dataset run the file `download_dataset.sh`. You may have to give execution permission on the file `chmod +x download_dataset.sh`. Run the file:

`./download_dataset.sh`

There is some housekeeping necessary to make the data processing easier in training time. To do that, execute the following line:

`python prep_dataset.py`

You should now see the `train_split.json` and `test_split.json` files in the main folder.

To start training your network you can must open the `lumberjack_net.py` and modify the parameters in the main() function.

When you are ready to run, you can execute:

`python lumberjack_net.py`

The results of your training and validation will be found in the folder `expXX` under the folder `logs`.


I would like to thank the authors of the reference paper for providing a comprehensive explanation of the dataset and for making the dataset open so that this work could exist.

Pan, Pengmin & Mcdonald, Timothy. (2019). Computers and Electronics in Agriculture Tree size estimation from a feller-buncher's cutting sound. Computers and Electronics in Agriculture. 159. 50-58. 10.1016/j.compag.2019.02.021. 
