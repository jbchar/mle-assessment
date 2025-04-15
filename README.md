# Home Price Prediction Api
Welcome! documented in this repo is the mock api that I created using docker and fastapi as part of the take home assessment for the MLE position at PhData. I hope you enjoy reading through it, and Im happy to answer any questions you have about the code.

 ![PhData logo](https://github.com/jbchar/mle-assessment/blob/main/src/mle_project_challenge_2/phData.png)

## Overview:
for this project I chose to use fastapi as the framework to build the api, here are some of the benefits of choosing fastapi:

* it is an opensource framework with a large community of users, so finding resources and documentation for development is easy
* Fast setup (Im only committing a few hours, so getting extra bang for my buck is valuable)
* good async support which is helpful for IO bound operations. This doesnt apply to our current usecase but large queries are common in ML workflows
* easy to implement things like things like OAuth2 via built in functions. important for securing sensitive apis
* easy testing through built in test clients (I didnt do this due to time constraints)
* automatic documentation of endpoints (very helpful when other engineers will be integrating api in their code, in this case they can be found at http://localhost:8000/docs)

I also used pydantic in order to map api input to feature inputs. The benefit of this is:
* built in input validation
* flexible inputs (allows for optional inputs so that we can accomodate different user inputs)
* easy to cast to and from other data structures (dict, json, dataframe, etc.)

And In order to deploy the api I used Docker as was recommended in the assignment documentation. I used Docker files with the python:3.9-slim image

## Endpoints:
* http://localhost:8000/houses/predict [post]:
    * Endpoint to generate inference from the model. Returns the prediction along with the features that were supplied to the model. Returns the actual price if it was included in the input
* http://localhost:8000/refresh_model [post]:
    * Endpoint that refreshes the model without restarting the server running the api, as per the supplied instructions
* http://localhost:8000/model_performance [get]:
    * Endpoint that returns the performance metrics of the trained models. The model with the lowest Mean Squared Error is the model that will be used in the API



## Installation:

This api is deployed using docker with a fairly simple instalation. simply run the following commands:
* deploy the api:
    * ```docker build -t ml-api . ```
    * ```docker run -p 8000:8000 ml-api ```

* you can then run the script from the top level directory with
    * ```python test_endpoint.py```

 you can also hit the api with adhoc curl commands. Here is an example curl commmand with some extra inputs handled by pydantic (local environment dependent, this is the curl from postman):
```
curl --location 'http://localhost:8000/houses/predict' \
--header 'Content-Type: application/json' \
--data '{
    "id": "2008000270",
    "date": "20150115T000000",
    "zipcode": "98198",
    "bedrooms": 3,
    "bathrooms": 1.5,
    "sqft_living": 1060,
    "sqft_lot": 9711,
    "floors": 1,
    "waterfront": 0,
    "sqft_above": 1500,
    "sqft_basement": 500
    }'
 ```

once the api is up and running, it can be reached via curl commands directed to localhost on port


## Modeling
* the base model was using a train test split, with a default 25% test size. however the test set was not used. I modified the code so that multiple models can be trained, and the test set would be used to evaluate the models and deploy the one with the better performance (due to stochastic nature of training this wouldnt necessarily be a great idea to do in production. i.e you might wind up with different models deployed depending on random chance). The evaluation function that I've written provides MAE, MSE, and R-square but for evaluating the models I chose to use MSE.

* There are some outliers in the data that require investigation. properties with 33 bedrooms, Price of 77 million, and some properties with 0 beds and baths (studio apt / comercial property?), year build 1900 (this is posible but could also be an error), . I would need to do additional research as to how to handle these outliers based on whether they are truly errors, or simply unique properties in the dataset. If they are truly errors, I would simply remove them. There are also some columns that could be categorical treated as numerical, i.e condition and grade, usually that's fine but would want to clarify and potentially use onehot encoding. Additionally would want to clarify what some of the columns represent in detail to ensure they are being processed appropriately

* I added an GBM model to compare with the KNN model. I felt this was a good fit for the problem while being easy to implement as it is similar in difficulty to KNN but can accomadate many complex relationships in input data and with many input features I felt this might provide better results

* The model performance is as follows
    * KNN model: While the KNN has a decent R-squared value (how much of the variance is explained by the model) at ~0.73, it has a high mean absolute error of ~102000, meaning on average the prediction will differ from the true price by 100k. This is quite significant and might unermine the usefulness of the api. The MSE is ~407000
    * GBM model: The GMB model fit the data a bit better with an R-squared of ~0.8 and an MSE of ~91000. This not very precise but perhaps with additional hyper parameter tuning it could improve. the MSE is ~299000

* there were a few small errors in the create model code that I corrected. For instance the create_model module incorectly listing both the housing data and demografic data location as "data/kc_house_data.csv", but this is actually not an issue becuase the demographic data location is actually hardcoded later in the file. I corrected this anyways

* in the real solution the models would be trained separately and the artifacts would be stored separately via a service like S3 or artifactory. This would be pulled into memory when the app starts up. I also wrote a mock function to manually refresh the model in memory given a new model version was created (this woulnt be needed depending on how the model was deployed ,i.e in aws with a service that has red/blue deployments)

## To-do
This was obviously not enough time to fully flesh out the api. There are a few basic things that I skipped in order to move more quickly. This includes but isnt limited to:

* unit testing (I skipped tdd in order to save time)
* securing the api (whether via api key or preferably OAuth2)
* hyper paramerter tuning of the model via some CV technique like grid search
* better logging instead of print statements
* more error handling

## Presentation
Presentation and proposed infra diagrams [here](https://docs.google.com/presentation/d/14XSqYv7xkXGyTvZMFF_zoMGKr8YykScvFxvyDY6CMME/edit?slide=id.gc6f73a04f_0_0#slide=id.gc6f73a04f_0_0)
