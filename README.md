# Application for price prediction of an apartment

The main aim of the project  -  to identify the best model to make the appartments price prediction with given features and use a supervised learning technique.

This dataset is available on Kaggle: https://www.kaggle.com/dmkravtsov/real-estate-odessa-prices-2020

## Total description:

● This application can be useful for organizations involved in the sale of real estate objects

● Based on the ensemble of trained models the application identifies the main features (such as area, number of rooms, location area, etc.) and its influence on the odjects value.

● Based on 8 basic indicators, the User receives a forecast for the cost of an appartments, the accuracy of which lies close to 80%

● The application allows the User to identify those real estate objects that are undervalued or overvalued in terms of its value. That in the future can serve as a starting point for making a decision to buy or sell an object.

● When designing the application, I designed a parser of real estate objects, used ensembles of mathematical models (stacking) to improve the stability of forecasts of the value of real estate objects, which had a positive effect on the experience of the Users.

● The application offline on a weekly basis makes self-study on new real estate objects, which allows it to track the latest trends in housing prices and provide up-to-date forecasts.

● This application was developed on the base of my experience with the House Prices: Advanced Regression Techniques script, which was included in the top 2% rating on the Kaggle resource https://www.kaggle.com/dmkravtsov/3-2-house-prices 

## Technologies used:

1. Python -  for backend
2. Beautifulsoup - for data parsing
3. PostgreSQL, SQLAlchemy - for database organize
4. Docker and Docker-compose - to create, configure and run application services
5. HTML and CSS - for frontend
6. Flask framework -  for integration of frontend and backend
7. Numpy, Sklearn, Pandas, RandomForestRegressor

## Steps to run the code:

1. Create a local folder for your project: $mkdir project && cd project
2. Upload the project with the command: $git clone https://github.com/dmkravtsov/odessapricepredictor.git
3. Create virtual environment:  $virtualenv venv
4. Initialize the repository: $ git init
5. Add your environment to .gitignore:  $echo 'venv' > .gitignore
6. Activate virtual environment: $source venv/bin/activate
7. Run docker compose file from project directory: docker-compose up
8. Real estate data will be parced once per week

### Please just fill in all the lines and get a forecast for the cost of your appartment

## Test predictor available on web: http://odessapricepredictor.herokuapp.com/  
## or on your localhost: http://0.0.0.0:5000/

![Alt text](api/static/css/predictor.jpg?raw=true) 

Please just fill in all the lines and get a forecast for the cost of your appartment.

Actual prices vs predicted:

![Alt text](api/static/css/diagram.png?raw=true) 

Best model will be considered as:

![Alt text](api/static/css/best_model.png?raw=true) 

## Summary: 

After getting the result  I still have to put much more efforts to improve my model. The result reflects that some of the valuable data might not yet to be discovered from the dataset. Probably need to review all data missed, outliers, and spend more time for data analysis and multicollinearity issue.


Thank you for your time!!