import joblib
from datetime import date
from pydantic import BaseModel

import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Imovel(BaseModel):
    regiao: str
    tipo_empreendimento: str
    agente: str
    construtor: str
    dist: str
    incorporado: str
    construtor: str
    ano_lancamento: date
    quartos: int
    banheiros: int
    garagens: int
    banheiros: int
    area_util: float
    area_total: float
    andares: int
    banheirosQuartos: int


@app.post("/imovel/previsao-preco/")
def previsao_preco_imovel(imovel: Imovel):

    preco = predict_price(imovel)
    
    return {"precoPrevisto": round(preco, 2)}

def predict_price(imovel):
    model = joblib.load('src/resources/modelo_preco_imoveis.pkl')
    
    dados = pd.DataFrame({
        'REGIAO': [imovel.regiao],
        'TIPO_EMP': [imovel.tipo_empreendimento],
        'AGENTE': [imovel.agente],
        'CONSTRUTOR': [imovel.construtor],
        'DIST': [imovel.dist],
        'INCORPORAD': [imovel.incorporado],
        'CONSTRUTOR': [imovel.construtor],
        'ANO_LAN': [imovel.ano_lancamento.year],
        'DORM_UNID': [imovel.quartos],
        'BANH_UNID': [imovel.banheiros],
        'GAR_UNID': [imovel.garagens],
        'BANH_DORM': [imovel.banheirosQuartos],
        'AR_UT_UNID': [imovel.area_util],
        'ANDARES': [imovel.andares],
        'AR_TT_UNID': [imovel.area_total]
    })

    price = model.predict(dados)
    return price[0]


