## Application for price prediction of an apartment

● This application can be useful for organizations involved in the sale of real estate objects

● Based on the ensemble of trained models the application identifies the main features (such as area, number of rooms, location area, etc.) and its influence on the odjects value.

● Based on 8 basic indicators, the User receives a forecast for the cost of an appartments, the accuracy of which lies close to 80%

● The application allows the User to identify those real estate objects that are undervalued or overvalued in terms of its value. That in the future can serve as a starting point for making a decision to buy or sell an object.

● When designing the application, I designed a parser of real estate objects, used ensembles of mathematical models (stacking) to improve the stability of forecasts of the value of real estate objects, which had a positive effect on the experience of the Users.

● The application offline on a weekly basis makes self-study on new real estate objects, which allows it to track the latest trends in housing prices and provide up-to-date forecasts.

● This application was developed on the base of my experience with the House Prices: Advanced Regression Techniques script, which was included in the top 2% rating on the Kaggle resource https://www.kaggle.com/dmkravtsov/3-2-house-prices 

## Technologies used:

Python -  for backend
Beautifulsoup - for data parsing
PostgreSQL, SQLAlchemy - for database organize
Docker and Docker-compose - to create, configure and run application services
HTML and CSS - for frontend
Flask framework -  for integration of frontend and backend
Numpy, Sklearn, Pandas, RandomForestRegressor

# Steps to run the code:

1. Create a local folder for your project: $mkdir project && cd project

2. Upload the project with the command: $git clone https://github.com/dmkravtsov/odessapricepredictor.git

3. Create virtual environment:  $virtualenv venv

4. Initialize the repository: $ git init

5. Add your environment to .gitignore:  $echo 'venv' > .gitignore

6. Activate virtual environment: $source venv/bin/activate

7. Run docker compose file from project directory: docker-compose up

8. Real estate data will be parced once per week

# Test predictor available on web: http://odessapricepredictor.herokuapp.com/  or on your localhost: http://0.0.0.0:5000/

![Alt text](api/static/css/predictor.jpg?raw=true) 