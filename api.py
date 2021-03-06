import uvicorn
from fastapi import FastAPI, File
from processing import predict, preprocess, read_image


#fastapi instance
app = FastAPI()

#CORS config
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get('/')
async def index():
    return "Hello"

if __name__ == '__main__':
    uvicorn.run(app,host="126.0.0.0", port=6660)
