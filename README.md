# Application for price prediction of an apartment in Odessa

     ● I’ve developed an application that will  be useful for real estate organizations involved in the sale of real estate objects
     
     ● Based on the ensemble of trained models the application identifies the main features (such as area, number of rooms, location area, etc.) and its influence on the value of the property
     
     ● Based on 8 basic indicators, the User receives a forecast for the cost of housing, the accuracy of which lies close to 80%
     
     ● The application allows the User to identify those real estate objects that are undervalued or overvalued in terms of its value. That in the future can serve as a starting point for making a decision to buy or sell an object.
     
     ● When designing the application, I designed a parser of real estate objects, used ensembles of mathematical models (stacking) to improve the stability of forecasts of the value of real estate objects, which had a positive effect on the experience of the Users.
     
     ● The application offline on a weekly basis makes self-study on new real estate objects, which allows it to track the latest trends in housing prices and provide up-to-date forecasts
     
     ● This application was developed on the base of my experience with the House Prices: Advanced Regression Techniques script, which was included in the top 2% rating on the Kaggle resource https://www.kaggle.com/dmkravtsov/3-2-house-prices 
     
     ● A test version of the application is available at http://odessapricepredictor.herokuapp.com/ 

# Create a local folder for your project: $mkdir project && cd project

#Upload the project with the command: $git clone https://github.com/dmkravtsov/odessapricepredictor.git

#Create virtual environment:  $virtualenv venv

#Initialize the repository: $ git init

#Add your environment to .gitignore:  $echo 'venv' > .gitignore

#Activate virtual environment: $source venv/bin/activate

#Install packages from requirements.txt file: $pip install -r requirements.txt

#Test predictor on web: http://odessapricepredictor.herokuapp.com/
