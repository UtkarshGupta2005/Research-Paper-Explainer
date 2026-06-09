from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    # Save + process PDF
    return {"status": "processed"}

@app.post("/query")
async def query(q: str):
    response = q.run(q)
    return {"answer": response}