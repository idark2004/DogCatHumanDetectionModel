import uvicorn
from fastapi import FastAPI, File
from processing import predict, preprocess, read_image


#fastapi instance
app = FastAPI()

#.../api/predict
@app.post("/api/predict")
async def predict_image(file : bytes = File(...)):
    #Read uploaded image
    image = read_image(file)
    #Preprocess
    image = preprocess(image)
    
    #Predict
    prediction = predict(image)
    return {"Predicted" : prediction[0]}

if __name__ == '__main__':
    uvicorn.run(app,host="126.0.0.0", port=6660)