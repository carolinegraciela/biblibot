import os
import pandas as pd

from langchain_community.vectorstores import FAISS

class BibleRetriever:
    def __init__(self, embeddings):
        self.data_path = os.getenv("BIBLE_DATA_PATH")
        self.index_faiss = os.getenv("INDEX_NAME")

        self.df_alkitab = pd.read_excel(self.data_path, engine='openpyxl', dtype=str)
        self.vectorstore = FAISS.load_local(self.index_faiss, embeddings, allow_dangerous_deserialization=True)

    async def requestDaftarKitab(self, id:int):
        daftar_kitab = self.df_alkitab['Book Name'].unique().tolist()
        return daftar_kitab[:39] if id == 0 else daftar_kitab[39:]

    async def requestListPasal(self, kitab:str):
        df_filter_kitab = self.df_alkitab[self.df_alkitab['Book Name'] == kitab]
        daftar_pasal = df_filter_kitab['Chapter'].unique().tolist()

        return daftar_pasal

    async def requestListAyat(self, kitab:str, pasal:int):
        pasal = str(pasal)
        daftar_ayat = self.df_alkitab[(self.df_alkitab['Book Name'] == kitab) &
                                (self.df_alkitab['Chapter'] == pasal)]['Verse'].unique().tolist()
        return daftar_ayat

    async def requestTeksAyat(self, kitab: str, pasal: int, ayat:int):
        pasal = str(pasal)
        ayat = str(ayat)

        filter_isi_ayat = self.df_alkitab[(self.df_alkitab['Book Name'] == kitab) &
                                    (self.df_alkitab['Chapter'] == pasal) &
                                    (self.df_alkitab['Verse'] == ayat)]
        isi_ayat = filter_isi_ayat['Text'].iloc[0]
        
        return kitab, pasal, ayat, isi_ayat

    def retrieveAnswers(self, query:str):
        docs = self.vectorstore.similarity_search(query, k=15)
        
        return docs
