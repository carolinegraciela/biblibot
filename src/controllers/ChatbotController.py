import re

#CLASS
from src.services.BibleRetriever import BibleRetriever 
from src.services.LLmService import LLmService

#COMMUNITY
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

#QUERY REWRITING
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

#CHAIN
from langchain_core.runnables import RunnablePassthrough

class ChatbotController:
    def __init__(self):
        self.__embedding_model = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-m3",
            model_kwargs = {'device': 'cpu'}
        )
        self._retriever = BibleRetriever(self.__embedding_model)
        self.llama_service = LLmService()
        self.llm = self.llama_service.generateResponse()
        self.reranker = self.llama_service.rerankerModel()

        self.system_instruction = """
            Anda adalah ahli teologi Kristen yang menjawab pertanyaan pengguna dengan menggunakan prinsip-prinsip teologis dari AYAT Alkitab referensi yang disediakan. 
            Tugas Anda adalah memberikan penjelasan teologis yang mendalam, kontekstual, dan sepenuhnya didasarkan pada prinsip-prinsip tersebut.

            Ketentuan Penting:
            1. HANYA buat SATU paragraf yang terdiri dari 4-5 kalimat. Jangan panjang-panjang.
            2. Jawablah menggunakan prinsip-prinsip dari AYAT ALKITAB yang diberikan.
            3. DILARANG KERAS MENGUTIP TEKS AYAT SECARA HARFIAH (literal). Tugas Anda adalah menjelaskan PRINSIP TEOLOGISnya dalam kata-kata Anda sendiri, bukan menyalin teks ayat tersebut.
            4. DILARANG KERAS mengarang fakta sejarah atau informasi fiktif.
            5. Jika pertanyaan membahas hal modern (misalnya rokok, internet) yang tidak disebutkan secara harfiah dalam Alkitab, jelaskan bagaimana PRINSIP MORAL/ETIKA dari ayat referensi dapat diterapkan dalam konteks modern tersebut.
            6. Dalam penjelasan Anda, jangan mengulangi kata-kata dalam ayat secara harfiah. Gunakan sinonim atau penjelasan konseptual.
            """
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction),
            ("human", "AYAT ALKITAB REFERENSI (untuk prinsip):\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nPENJELASAN PRINSIP TEOLOGIS (dalam 4-5 kalimat, dilarang mengutip teks):") 
        ])
        self.generation_chain = self.prompt | self.llm | StrOutputParser()


    # 2nd MENU: Question-Answering with Biblibot
    def query_rewriting(self, original_query):
        examples = [
            {
                "input": "boleh gak pacaran beda agama",
                "output": "- Cari ayat tentang pasangan yang tidak seimbang antara terang dan gelap.\n- Ayat tentang anak-anak terang tidak bisa bersatu dengan anak-anak gelap\n- Cari ayat tentang pasangan yang tidak sepadan"
            },
            {
                "input": "orang yang suka tatoan itu dosakah?",
                "output": "- Cari ayat yang menyatakan bahwa tubuh adalah Bait Suci Kristus.\n- Ayat tentang menjaga bait Kristus\n- Cari ayat yang menyatakan bahwa tubuh bukan milik diri sendiri, melainkan bait Allah"
            },
            {
                "input": "siapa Yohanes dalam Alkitab?",
                "output": "- Cari ayat yang membahas tentang kelahiran Yohanes\n- Cari ayat yang menyatakan pelayanan Yohanes\n- Cari ayat yang isinya \"Yohanes adalah anak dari...\""
            },
            {
                "input": "hsudhufdha",
                "output": "[GIBBERISH]"
            },
            {
                "input": "24$jshduie*dshg^",
                "output": "[GIBBERISH]"
            },
            {
                "input": "halo biblibot, selamat pagi!",
                "output": "[GREETING]"
            },
            {
                "input": "bagaimana cara membuat nasi goreng jawa?",
                "output": "[OUT_OF_DOMAIN]"
            }
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = examples,
        )

        system_instruction =  """### INSTRUKSI
        Anda adalah sistem klasifikasi dan penulis ulang kueri (Query Rewriter) untuk chatbot pencarian ayat Alkitab.
        Tugas Anda adalah menganalisis input pengguna dan memberikan output berdasarkan 4 kondisi berikut:

        1. GIBBERISH: Jika input adalah ketikan acak, tidak bermakna, atau hanya simbol (misal: "asdfg", "...", "husdbuwiefue"), balas HANYA dengan: [GIBBERISH]
        2. SAPAAN: Jika input murni sapaan, balas HANYA dengan: [GREETING]
        3. LUAR TOPIK: Jika input TIDAK ADA hubungannya sama sekali dengan Alkitab, Kekristenan, atau moralitas/pergumulan hidup, balas HANYA dengan: [OUT_OF_DOMAIN]
        4. KUERI VALID: Jika berkaitan dengan Alkitab atau pergumulan hidup, ubah kueri menjadi 3 KALIMAT PERNYATAAN untuk pencarian efektif.

        ### PANDUAN KONTEKS (Untuk Kueri Valid)
        - "benci, kesal, marah, dendam" -> "ayat tentang mengasihi musuh dan sesama"
        - "insecure, merasa tidak pantas" -> "ayat yang menyatakan bahwa Yesus mengasihi dan menghargai manusia"
        - "rokok, vape, vaping, tato" -> "ayat bahwa tubuh ialah Bait Suci Kristus"
        - "pacaran beda agama" -> "ayat tentang pasangan yang tidak seimbang terang dan gelap"
        - "hujat, gosip" -> "ayat yang menyatakan untuk kita berhati-hati menggunakan lidah"
        - "meso, ngomong kotor, anjir" -> "ayat yang menyatakan bahwa hati-hati menggunakan mulut (mulut yang sama yang digunakan untuk memuji Allah)"

        ### ATURAN MUTLAK
        - JAWAB HANYA DENGAN TAG KODE (seperti [GIBBERISH]) ATAU HASIL PARAFRASE.
        - DILARANG MEMBERIKAN PENJELASAN, BASA-BASI, ATAU MENYAPA BALIK DI SINI.
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            few_shot_prompt,                  
            ("human", "{original_query}")    
        ])

        model_with_stop = self.llm.bind(stop=["User:", "###", "Human:"])
        
        query_rewriter = final_prompt | model_with_stop | StrOutputParser()
        response = query_rewriter.invoke({"original_query": original_query})

        clean_response = response.strip()
        print(f"[Query Rewriting Output]:\n{clean_response}")

        return clean_response

    def hyde(self, kueri):
        system_template = """Anda adalah ahli teologi dan sejarah Alkitab. Tugas Anda adalah membuat "dokumen hipotetis" BERDASARKAN Alkitab 
        berupa SATU PARAGRAF (4-5 kalimat) yang mendalam untuk menjawab pertanyaan pengguna.
        Paragraf ini akan digunakan oleh mesin pencari untuk mencocokkan makna (semantic search) dengan ayat-ayat Alkitab.

        ATURAN MUTLAK:
        - Jawab HANYA dengan SATU paragraf utuh.
        - DILARANG menggunakan bullet points (-) atau angka.
        - DILARANG menambahkan awalan kata seperti "QUERY:" atau "Jawaban:".
        - JAWABAN HARUS Berdasarkan pada ALKITAB.
        - Jika kueri tentang teologi/moral, berikan penjelasan teologis. 
        - Jika kueri tentang TOKOH, TEMPAT, atau PERISTIWA, berikan ringkasan narasi sejarah, peran tokoh tersebut, dan fakta-fakta kunci yang tercatat dalam Alkitab.

        CONTOH 1 (Kueri Moral/Teologis):
        Pertanyaan: bagaimana mengatasi teman yang suka nusuk dari belakang?
        Dokumen Hipotetis: Menghadapi pengkhianatan dari orang terdekat memang menyakitkan, namun ajaran Kekristenan menekankan pentingnya pengampunan dan kasih karunia. Daripada membalas dendam atau menyimpan kepahitan, kita diajarkan untuk mendoakan mereka yang menyakiti kita dan menyerahkan keadilan seutuhnya kepada Tuhan. Menjaga hati agar tetap bersih dari kebencian adalah wujud dari ajaran kasih.

        CONTOH 2 (Kueri Tokoh/Fakta Alkitab):
        Pertanyaan: Ayat tentang Musa yang dipilih Allah untuk memimpin bangsa Israel.
        Dokumen Hipotetis: Musa adalah tokoh sentral dalam Perjanjian Lama yang dipanggil Allah secara langsung melalui peristiwa semak duri yang menyala di Gunung Horeb. Meskipun Musa awalnya merasa ragu dan tidak fasih berbicara, Allah mengutusnya untuk menghadap Firaun dan membebaskan bangsa Israel dari perbudakan di tanah Mesir. Melalui berbagai tanda mukjizat dan penyertaan ilahi, Musa diangkat menjadi pemimpin sekaligus perantara perjanjian antara Allah dan umat pilihan-Nya dalam perjalanan menuju Tanah Perjanjian.

        Pertanyaan: {query}
        Dokumen Hipotetis:"""


        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Pertanyaan: {query}\nDokumen Hipotetis:")    
        ])

        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        input_variables = {"query": kueri} 
        response = hyde_chain.invoke(input_variables)
        print(f"Dokumen hipotesis: {response.strip()}")

        return response.strip()

    def reranking(self, query: str, retrieved_docs: str, top_n: int = 3):
        sentence_pairs = [[query, doc.page_content] for doc in retrieved_docs]
        scores = self.reranker.score(
            sentence_pairs
        )
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        
        ranked = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
        best_docs = []

        for score, doc in ranked[:top_n]:
            print(f"Rerank Score: {score:.4f} | {doc.page_content[:80]}...")        
            if hasattr(doc, 'metadata'):
                doc.metadata["rerank_score"] = float(score)            
            best_docs.append(doc)

        return best_docs

    def generate_response(self, user_query: str):
        rewritten_query = self.query_rewriting(user_query)
        print(f"Query hasil rewriting: {rewritten_query}")

        if "[GIBBERISH]" in rewritten_query:
            return "Maaf, saya tidak mengerti pertanyaanmu. Ketik lebih jelas lagi, ya!😊"
        elif "[GREETING]" in rewritten_query:
            return "Halo, shalom! Saya Biblibot, ada yang bisa dibantu?"
        elif "[OUT_OF_DOMAIN]" in rewritten_query:
            return "Maaf, saya adalah chatbot yang didesain untuk membantu pencarian ayat Alkitab secara kontesktual. Apa yang ingin kamu tanyakan?"
        
        hasil_hyde = self.hyde(rewritten_query)   
        retrieval_docs = self._retriever.retrieveAnswers(hasil_hyde)
        final_retrieval = self.reranking(rewritten_query, retrieval_docs)

        hasil_retrieval = "\n".join([doc.page_content for doc in final_retrieval])

        system_instruction = """
            Anda adalah ahli teologi kristen yang menjawab segala pertanyaan berbasis AYAT Alkitab.
            Tugas anda adalah membuat sebuah jawaban teologis terkait pertanyaan pengguna yang BERBASIS ayat Alkitab,
            dengan beberapa ketentuan di bawah:
            1. Jawaban terdiri dari 4-5 kalimat. HANYA 1 paragraf, jangan panjang-panjang.
            2. Jawab HANYA menggunakan prinsip-prinsip dari AYAT ALKITAB.
            3. DILARANG mengutip AYAT ALKITAB (Tugasmu hanya memberi PENJELASAN bukan MENGUTIP AYAt)
            4. DILARANG KERAS mengarang fakta sejarah (misalnya: membahas penemuan senjata api, sejarah rokok, atau hal-hal fiktif lainnya).
            5. Jika pertanyaan membahas hal modern (seperti rokok, internet) yang tidak disebutkan secara harfiah, jelaskan bagaimana PRINSIP MORAL dari ayat referensi dapat diterapkan.
            6. Jangan mengutip ulang isi ayat secara harfiah.
            """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "AYAT ALKITAB:\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nKALIMAT PENGANTAR JAWABAN:")        
        ])

        chain = prompt | self.llm | StrOutputParser()
        ringkasan_llm = chain.invoke({"context": hasil_retrieval, "question": user_query}).strip()

        if not ringkasan_llm:
            ringkasan_llm = "Prinsip Alkitab yang sesuai pertanyaan anda dapat dilihat pada referensi ayat berikut:"
        jawaban_final = f"{ringkasan_llm}\n\n"
        jawaban_final += "Berikut adalah referensi ayat yang relevan:\n"

        for doc in final_retrieval:
            sumber = doc.metadata.get('sumber', '-')
            isi_ayat = doc.page_content
            jawaban_final += f"- **{sumber}**: \"{isi_ayat}\"\n"
            jawaban_final += "\n"

        return jawaban_final

    # 1st MENU: Digital Bible reading   
    def getDaftarKitab(self, id:int) -> list[str]:
        return self._retriever.requestDaftarKitab(id)
    
    def getDaftarPasal(self, kitab:str) -> list[int]:
        return self._retriever.requestListPasal(kitab)
    
    def getDaftarAyat(self, pasal:int) -> list[int]:
        return self._retriever.requestListAyat(pasal)
    
    def getTeksAyat(self, ayat:int) -> str:
        return self._retriever.requestTeksAyat(ayat)