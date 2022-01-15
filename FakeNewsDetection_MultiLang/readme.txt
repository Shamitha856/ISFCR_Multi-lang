Guide for a stepwise execution of the entire project- FakeNewsDetection_MultiLang

Overview: This project basically consists of 3 different parts(models) for categorizing news as fake or real. One model for English, one for Kannada and similarly one for Hindi.

Steps to execute:

1)Download the entire project folder named " FakeNewsDetection_MultiLang" from the github repo
	repo link:https://github.com/Shamitha856/ISFCR_Multi-lang

2)Create an environment or a virtual environment to execute the project downloaded.

3).py files having implementation code of the web interface are under the "WebInterface" folder inside the main project folder.
	There are 3 folders within WebInterface folder "English"," Hindi", and "Kannada" which contain all the files required for implementation of the 	frontend.
	Inturn under each of these folder there are 2 other folders static (containing style file for UI) and templates (containing an html file for web 	interface), retain this directory structure as it is(as they are uploaded onto the repo) to avoid issues related to file paths.
	app_language.py are the API files for the frontend.(there are 3 of them one for each language)

4)Inside main project directory there exists "Models" folder which contain 4 pickle files (2 for english[1 for prediction through text and the other for prediction 	via URL], 1 for kannada and 1 for hindi) store these dumped model files in some directory(better to be under parent directory).

5)Install all the necessary requirements from "requirements_webinterface.txt" under the main project folder into the execution environment.

6)Files required to run the web interface: pickle files and 3 folders("English","kannada", and "Hindi") mentioned in point 3 + requirements
	once all 3 are ready inplace run app_english.py app_kannada.py and app_hindi.py(inside WebInterface 	folder) either using Spyder(recommended) 	or as running any normal .py script files.
	Each of the 3 .py files are to be run seperately.
	This step completes rendering web interface on port 5000(mentioned in .py files) one at a time.



Steps for Generating pickle models(not required for the execution of the project):

1)Create an environment to execute.

2)Install all requirements in "requirements.txt" under main folder using the command.

3)All the datasets required for building the models are provided as a google drive link in "dataset.txt", download all folders and store them in the project 		directory.

4)To generate the models run the notebooks(3) in the "Notebooks" folder under the project in the environment kernel created(Jupyter can be used).

5)Other way to generate models:call the .py files present in the "Python Files" directory, from flask code .

6)By the end of step 5 all pickle models would be dumped, and same steps mentioned previously are to be followed to further create a web interface.