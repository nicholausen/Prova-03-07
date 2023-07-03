from fastapi import FastAPI,Depends,HTTPException,status
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI(title="API StartUp", version="1.0")


class Data(BaseModel):
    rd_spend: float =100000
    administration: float =100000
    marketing_spend: float =150000


@app.on_event("startup")
def startup_event():
    "modello *.pkl di ML"
    global model 

    model = joblib.load('/Users/nic/Desktop/Prova Grotti /Prova-03-07/modello.pkl')
    print("MODEL LOADED")

    return model


@app.get("/")
def home():
    return {" ---->          http://localhost:8000/docs     <----------"}

@app.get("/predict")
async def predictget(data:Data=Depends()):
    try:
        X = [[data.rd_spend, data.administration, data.marketing_spend]]
        y_pred = model.predict(X)[0]
        res = round(y_pred,2)
        return {'prediction':res}
    except:
        raise HTTPException(status_code=404, detail="error")


@app.post("/predict")
async def predictpost(data:Data):
    try:
        X = [[data.rd_spend, data.administration, data.marketing_spend]]
        y_pred = model.predict(X)[0]
        res = round(y_pred,2)
        return {'prediction':res}
    except:
        raise HTTPException(status_code=404, detail="error")



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
