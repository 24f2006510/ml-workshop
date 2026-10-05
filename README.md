# Movie Review Sentiment Analysis Model API
This model predicts the sentiment of a movie review as "Positive" or "Negetive"

## Example of RequestBody : {"review" : "this movie was great"}

## To run Locally :
* pip install -r requirements.txt
* uvicorn main:app --reload

## test locally :
* curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"review\": \"this movie is good\"}"
