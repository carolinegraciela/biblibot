import os
import pandas as pd

from langchain_community.vectorstores import FAISS

class BibleRetriever:
    def __init__(self, embeddings):
        self.data_path = os.getenv("BIBLE_DATA_PATH")
        self.index_faiss = os.getenv("INDEX_NAME")

        self.df_alkitab = pd.read_excel(self.data_path, engine='openpyxl', dtype=str)
        self.vectorstore = FAISS.load_local(self.index_faiss, embeddings, allow_dangerous_deserialization=True)
        self.current_kitab = 0
        self.current_pasal = 0
        self.current_ayat = 0

    async def requestDaftarKitab(self, id:int):
        daftar_kitab = self.df_alkitab['Book Name'].unique().tolist()
        return daftar_kitab[:39] if id == 0 else daftar_kitab[39:]

    async def requestListPasal(self, kitab:str):
        self.current_kitab = kitab
        df_filter_kitab = self.df_alkitab[self.df_alkitab['Book Name'] == self.current_kitab]
        daftar_pasal = df_filter_kitab['Chapter'].unique().tolist()

        return daftar_pasal

    async def requestListAyat(self, pasal:int):
        self.current_pasal = str(pasal)
        daftar_ayat = self.df_alkitab[(self.df_alkitab['Book Name'] == self.current_kitab) &
                                (self.df_alkitab['Chapter'] == self.current_pasal)]['Verse'].unique().tolist()

        return daftar_ayat

    async def requestTeksAyat(self, ayat:int):
        self.current_ayat = str(ayat)
        filter_isi_ayat = self.df_alkitab[(self.df_alkitab['Book Name'] == self.current_kitab) &
                                    (self.df_alkitab['Chapter'] == self.current_pasal) &
                                    (self.df_alkitab['Verse'] == self.current_ayat)]
        isi_ayat = filter_isi_ayat['Text'].iloc[0]
        
        return self.current_kitab, self.current_pasal, self.current_ayat, isi_ayat

    def retrieveAnswers(self, query:str):
        docs = self.vectorstore.similarity_search(query, k=15)
        
        return docs
