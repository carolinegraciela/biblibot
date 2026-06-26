import asyncio

#CLASS
from src.services.BibleRetriever import BibleRetriever 
from src.services.LLmService import LLmService

#COMMUNITY
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

#QUERY REWRITING
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

#MEMORY
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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
        self.store = {}

        
    # 2nd MENU: Question-Answering with Biblibot
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory: 
        """Get or create a history"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def intent_classification(self, original_query, session_id):
        examples = [
            {"input": "ayat Alkitab yang menyatakan bahwa ...", "output": "[VALID_QUERY]"},
            {"input": "ayat Alkitab ttg Allah Tritunggal", "output": "[VALID_QUERY]"},
            {"input": "ayat yang menyatakan Yesus adalah Allah", "output": "[VALID_QUERY]"},
            
            # --- CONTRASTIVE PAIRING ---
            {"input": "ayat tentang keselamatan hanya ada dalam yesus", "output": "[VALID_QUERY]"},
            {"input": "apakah benar keselamatan hanya ada dalam yesus? tolong jelaskan", "output": "[EXEGESIS_REQUEST]"},            
            {"input": "ayat tentang perumpamaan anak yang hilang", "output": "[VALID_QUERY]"},
            {"input": "apa makna dari perumpamaan anak yang hilang?", "output": "[EXEGESIS_REQUEST]"},
            {"input": "arti dari ayat ...?", "output": "[EXEGESIS_REQUEST]"},
            {"input": "makna perumpamaan ...?", "output": "[EXEGESIS_REQUEST]"},

            # ---------------------------

            {"input": "gimana cara bikin nasi goreng jawa?", "output": "[OUT_OF_DOMAIN]"},
            {"input": "halo biblibot!", "output": "[GREETING]"},
            {"input": "asdfghjkl", "output": "[GIBBERISH]"}
        ]

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt = example_prompt,
            examples = examples,
        )

        system_instruction = """### INSTRUKSI
        Anda adalah sistem klasifikasi Niat Pengguna (Intent Classification) untuk chatbot pencarian ayat.
        Tugas Anda adalah menganalisis input pengguna dan WAJIB memberikan output SATU TAG SAJA berdasarkan hierarki 5 kondisi berikut:

        1. [EXEGESIS_REQUEST] : Jika pengguna secara eksplisit menanyakan makna, arti, maksud, tafsiran, atau penjelasan teologis (misal: "apa arti", "maksud dari", "tafsirkan"). JANGAN gunakan tag ini jika pengguna murni mencari ayat atau menceritakan peristiwa.
        2. [GIBBERISH] : Jika input adalah ketikan acak, tidak bermakna, atau hanya simbol (misal: "asdfg").
        3. [GREETING] : Jika input murni sapaan (halo, pagi, shalom).
        4. [OUT_OF_DOMAIN] : HANYA gunakan jika input MURNI tentang hal duniawi (tutorial, tanya resep, info pariwisata murni seperti "berapa harga tiket ke bali"). 
           *PANTANGAN KERAS*: Jika pengguna menanyakan ETIKA, BATASAN MORAL, atau BOLEH/TIDAKNYA suatu aktivitas secara rohani (misal: "boleh gak pakai baju terbuka saat liburan?", "nonton bioskop dosa gak?"), JANGAN gunakan tag ini! Itu adalah [VALID_QUERY].
        5. [VALID_QUERY] : Jika pengguna membagikan masalah hidup, mencari ayat, menyebutkan nama tokoh, atau meminta solusi rohani.

        ### ATURAN MUTLAK
        - OUTPUT HANYA BOLEH BERUPA SATU TAG KODE DI ATAS. 
        - DILARANG MEMBERIKAN PENJELASAN, BASA-BASI, ATAU TEKS TAMBAHAN APAPUN.
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            few_shot_prompt,  
            ("placeholder", "{chat_history}"),                
            ("human", "{original_query}")    
        ])

        model_with_stop = self.llm.bind(stop=["User:", "###", "Human:"])
        
        intent_classifier = final_prompt | model_with_stop | StrOutputParser()
        history_obj = self.get_session_history(session_id)
        pesan_riwayat_sebelumnya = history_obj.messages

        response = intent_classifier.invoke({
            "chat_history": pesan_riwayat_sebelumnya,
            "original_query": original_query})

        clean_response = response.strip()
        return clean_response

    def hyde(self, kueri):
        system_template = """### INSTRUKSI
        Anda adalah mesin pembuat Dokumen Hipotetis (HyDE) untuk mengoptimalkan pencarian semantik (Vector Search) pada Alkitab Terjemahan Baru (TB).
        Input yang Anda terima adalah kueri mentah dari pengguna (bisa berupa curhatan, bahasa gaul, pertanyaan historis, atau masalah hidup).

        Tugas Anda: Ekstrak inti teologis atau historis dari input tersebut, lalu tulis teks hipotetis yang GAYA BAHASA, KOSAKATA, dan STRUKTUR KALIMATNYA meniru persis ayat Alkitab TB. Teks ini harus seolah-olah disalin langsung dari Alkitab.

        ### ATURAN PERCABANGAN 5W1H (WAJIB DIPATUHI):
        1. SOLUSI / EKSPLANASI (Masalah Hidup/Curhat/Doktrin): Buat 3-4 kalimat padat bergaya Alkitab TB yang memuat prinsip teologis tegas untuk kueri tersebut.
        2. FAKTA (Who/Where/When - Siapa/Di mana/Kapan): Jika kueri mencari fakta, daftar nama, atau sejarah (contoh: siapa, sebutkan, di mana), DILARANG KERAS membuat dialog karangan. Buat 1-2 kalimat deklaratif yang LANGSUNG MENYEBUTKAN NAMA TOKOH, LOKASI, ATAU WAKTU secara spesifik, meniru gaya penulisan silsilah atau daftar nama dalam Alkitab.

        ### ATURAN MUTLAK & ANTI-HALUSINASI:
        1. Jawab HANYA dengan teks hipotetis. DILARANG KERAS memberikan pengantar (seperti "Ayat yang relevan:", "Mari kita baca").
        2. TRANSLASI BAHASA MODERN (KAMUS WAJIB): 
           - "komsel / ibadah / gereja" = "pertemuan ibadah / persekutuan"
           - "badminton / padel / hobi / nongkrong" = "kesenangan duniawi / perkara dunia" (PANTANGAN: Jangan hubungkan olahraga dengan ayat tentang suami-istri, percabulan, atau hawa nafsu!).
           - "pacaran beda agama / beda keyakinan" = "pasangan yang tidak seimbang / terang dan gelap" (PANTANGAN: Jangan gunakan ayat tentang kesetaraan manusia dalam Kristus untuk menoleransi hal ini!).
           - "bos / atasan" = "tuan".
           - "rokok / merokok / tato / vaping / miras" = "merajah tanda pada kulit / mencemari bait Roh Kudus / diperhamba oleh sesuatu" (PANTANGAN: Fokus pada prinsip kekudusan tubuh sebagai bait Allah dan penguasaan diri. Jangan hubungkan dengan ayat tentang makanan atau binatang najis!).

        ### CONTOH PEMROSESAN:
        Input: "saya sedang diajak teman saya untuk padel, tapi jamnya berkaitan dengan jam komsel. bimbang."
        Dokumen Hipotetis: Carilah dahulu Kerajaan Allah dan kebenarannya, maka semuanya itu akan ditambahkan kepadamu. Janganlah kita menjauhkan diri dari pertemuan-pertemuan ibadah kita, seperti dibiasakan oleh beberapa orang, tetapi marilah kita saling menasihati.

        Input: "Saya mau ke lombok sama temen2, ke pantainya. Apakah tidak apa-apa untuk pakai pakaian yang terbuka?"
        Dokumen Hipotetis: Muliakanlah Allah dengan tubuhmu, sebab tubuhmu adalah bait Roh Kudus yang diam di dalam kamu. Demikian juga hendaknya perempuan berdandan dengan pantas, dengan sopan dan sederhana.

        Input: "Ada orang yang deketin saya, dan dia tipe saya pol. Tapi dia dari agama berbeda, gimana menurutmu?"
        Dokumen Hipotetis: Janganlah kamu merupakan pasangan yang tidak seimbang dengan orang-orang yang tak percaya. Sebab persamaan apakah terdapat antara kebenaran dan kedurhakaan? Atau bagaimanakah terang dapat bersatu dengan gelap?

        Input: "Siapakah anak daud yg menjadi raja israel?"
        Kemudian seluruh jemaat mengakui Salomo, anak Daud, sebagai raja. Mereka mengurapi dia bagi TUHAN menjadi raja menggantikan Daud, ayahnya.

        Input: "siapa saja murid yesus?"
        Dokumen Hipotetis: Inilah nama kedua belas rasul itu: Pertama Simon yang disebut Petrus dan Andreas saudaranya, dan Yakobus anak Zebedeus dan Yohanes saudaranya, Filipus dan Bartolomeus, Tomas dan Matius pemungut cukai, Yakobus anak Alfeus, dan Tadeus, Simon orang Zelot dan Yudas Iskariot yang mengkhianati Dia.

        Input: "Ayat ttg kisah Abraham mau membunuh Ishak"
        Allah mencobai Abraham dan berfirman kepadanya agar mempersembahkan Ishak, anak tunggalnya yang dikasihinya, sebagai korban bakaran. Ketika Abraham mengulurkan tangannya dan mengambil pisau untuk menyembelih anaknya, berserulah Malaikat TUHAN dari langit melarangnya.

        Input: "Ayat tentang keselamatan"
        Sebab karena kasih karunia kamu diselamatkan oleh iman; itu bukan hasil usahamu, tetapi pemberian Allah. Keselamatan tidak ada di dalam siapapun juga selain di dalam Dia. Sebab di bawah kolong langit ini tidak ada nama lain yang diberikan kepada manusia yang olehnya kita dapat diselamatkan.        
        """

        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Input:\n{query}\n\nDokumen Hipotetis:")    
        ])

        try:
            hyde_chain = hyde_prompt | self.llm | StrOutputParser()
            response = hyde_chain.invoke({"query": kueri}).strip()
            
            if response.lower().startswith("berikut adalah") or response.lower().startswith("tentu"):
                response = '\n'.join(response.split('\n')[1:]).strip()

            print(f"Dokumen hipotesis: {response}")
            return response
            
        except Exception as e:
            print(f"[API ERROR] HyDE Generation Failed: {str(e)}")
            return kueri 

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

    async def generate_response(self, user_query: str, session_id: str):
        intent_tag = await asyncio.to_thread(self.intent_classification, user_query, session_id)
        print(f"Query Intention: {intent_tag}")

        if "[GIBBERISH]" in intent_tag:
            return "Maaf, saya tidak mengerti pesan yang kamu kirimkan. Bisa tolong ketik ulang dengan lebih jelas? 😊"
        elif "[GREETING]" in intent_tag:
            return "Halo, shalom! Saya Biblibot, asisten pencarian ayat Alkitab secara kontekstual. Ada topik pergumulan atau kisah Alkitab yang ingin kamu cari hari ini? 🙏"
        elif "[OUT_OF_DOMAIN]" in intent_tag:
            return (
                "Maaf, saya hanya didesain untuk membantu mencari ayat Alkitab berdasarkan topik atau pergumulan rohani. "
                "Saya tidak bisa menjawab pertanyaan umum di luar Alkitab seperti resep masakan, tips teknologi, atau pariwisata. "
                "Silakan ketikkan topik rohani atau masalah yang sedang kamu hadapi ya! ✨"
            )
        elif "[EXEGESIS_REQUEST]" in intent_tag:
            return (
                "Maaf, saya dirancang khusus sebagai asisten mesin pencari ayat secara kontekstual, "
                "bukan untuk menafsirkan, menjelaskan makna teologis, atau memberikan arti pada ayat tertentu.\n\n"
                "Untuk mendapatkan penjelasan yang mendalam dan akurat mengenai tafsiran tersebut, "
                "saya sangat menyarankan untuk berdiskusi dengan pendeta, pembimbing rohani, atau membaca buku tafsir Alkitab. 🙏\n\n"
                "Namun, jika kamu ingin mencari ayat-ayat lain dengan *tema serupa*, silakan ketikkan topik pergumulan atau kata kuncinya!"
            )
        elif "[KUERI TIDAK VALID" in intent_tag or not intent_tag.strip():
            return (
                "Maaf, kueri yang kamu masukkan tidak dapat saya proses. "
                "Pastikan kamu memasukkan topik kehidupan, cerita tokoh Alkitab, atau situasi yang sedang kamu alami "
                "agar saya bisa mencarikan ayat Alkitab yang tepat untukmu. Silakan coba lagi! 😊"
            )
        
        hyde = await asyncio.to_thread(self.hyde, user_query)
        retrieval_docs = await asyncio.to_thread(self._retriever.retrieveAnswers, query=hyde)        
        final_retrieval = await asyncio.to_thread(self.reranking, query=hyde, retrieved_docs=retrieval_docs)

        system_instruction = """
        Anda adalah seorang mentor rohani dan sahabat seiman yang hangat, bijak, namun SANGAT TEGAS dalam memegang prinsip Firman Tuhan. Tugas Anda adalah menjawab masalah pengguna berdasarkan ayat Alkitab yang disajikan dalam konteks.

        ### ATURAN MUTLAK PENYARINGAN KONTEKS & ANTI-HERESY (WAJIB DIPATUHI):
        1. DILARANG MEMUTARBALIKKAN KONTEKS (CONTEXT TWISTING): Jangan pernah menggunakan ayat tentang keselamatan, kasih karunia, atau kesetaraan rohani (seperti Galatia 3:28) untuk membenarkan kompromi duniawi seperti pacaran beda agama, percabulan, atau dosa lainnya.
        2. KETAHUI BATASAN AYAT: Jika ayat yang disajikan dalam konteks TIDAK MENDUKUNG perbuatan pengguna secara alkitabiah, Anda WAJIB menegur dengan tegas. Jangan memaksakan ayat agar terdengar "mendukung" atau "menyetujui" keinginan pengguna.
        3. ANALISIS SITUASI PENGGUNA: Deduksi secara logis fase hidup pengguna.
        4. BUANG KONTEKS SAMPAH: Anda WAJIB MENGABAIKAN ayat dalam konteks yang secara kata mirip, tetapi salah sasaran secara situasi.

        ### ATURAN GAYA BAHASA & TONE (WAJIB DIPATUHI):
        1. BERBICARA SEPERTI MANUSIA: Gunakan bahasa sehari-hari yang mengalir, santai, dan penuh empati layaknya sahabat yang sedang *ngobrol* di kedai kopi. 
        2. DILARANG KERAS MELAKUKAN META-TALK: Jangan pernah membuka kalimat dengan menjelaskan apa yang Anda lakukan. HAPUS SEMUA frasa kaku seperti: "Dalam konteks ini", "Berdasarkan ayat di atas", "Saya menemukan ayat", "Dalam kasus ini". Langsung saja masuk ke inti pembicaraan.
        3. TEGAS TAPI TIDAK MENGGURU: Sampaikan kebenaran secara lugas (to the point) tanpa terkesan menghakimi, tetapi DILARANG berkompromi dengan pandangan sekuler atau toleransi modern.
        
        ### KERANGKA PENJAWABAN 5W1H (TARGET SASARAN MUTLAK):
        Analisis niat di balik kueri pengguna dan sesuaikan fokus jawaban Anda berdasarkan salah satu dari 3 kategori berikut agar tidak bertele-tele:
        - FAKTA (Who/Where/When - Siapa/Di mana/Kapan): Jika kueri mencari fakta historis, LANGSUNG sebutkan identitas tokoh, lokasi, atau waktu peristiwa tanpa menceritakan ulang seluruh plotnya.
        - EKSPLANASI (What/Why - Apa/Mengapa): Jika kueri mencari alasan atau doktrin, FOKUS pada latar belakang teologis, tujuan Allah, atau makna di balik sebuah perintah.
        - SOLUSI (How - Bagaimana/Curhat): Jika kueri adalah curhatan atau meminta panduan praktis, FOKUS pada langkah nyata, perubahan pola pikir, atau sikap hati yang harus diambil pengguna sesuai prinsip ayat.

        ### ATURAN MUTLAK FORMAT & PANJANG JAWABAN (HARD CONSTRAINT):
        1. BATASAN PANJANG: Anda WAJIB menjawab TEPAT dalam 3 SAMPAI 4 KALIMAT saja. DILARANG KERAS menulis kalimat ke-5. Jika lebih dari 4 kalimat, Anda gagal menjalankan sistem.
        2. Jawab langsung ke akar masalah berdasarkan pemetaan 5W1H di atas. DILARANG membuat struktur esai, pembukaan basa-basi, atau penutup bertele-tele.
        3. REFERENSI NATURAL & TANPA COPY-PASTE: Anda WAJIB mendasarkan setiap solusi pada konteks ayat yang disajikan sistem. Anda diizinkan menyebutkan nama kitab/pasal referensinya dengan gaya bahasa santai (contoh: "Kalau kita ingat teguran Paulus di Galatia...", "Amsal mengingatkan kita bahwa..."), namun DILARANG KERAS menyalin ulang teks ayat tersebut kata per kata. Ekstrak langsung intisari teologisnya.
        4. GUNAKAN KALIMAT PADAT (STRUKTUR S-P): Setiap kalimat WAJIB langsung menembak ke inti masalah. DILARANG KERAS menggunakan frasa pengisi, kata pengantar, atau kata hubung yang membuang kuota kalimat (HAPUS kata seperti: "Oleh karena itu", "Maka dari itu", "Sebagai sahabat", "Perlu diingat bahwa", "Ketahuilah bahwa"). Pastikan setiap kata bernilai dan memiliki bobot solusi.

        ### ATURAN MUTLAK PENYARINGAN KONTEKS (DYNAMIC STATE-AWARENESS):
        1. ANALISIS SITUASI PENGGUNA: Deduksi secara logis fase hidup atau situasi spesifik pengguna berdasarkan pertanyaannya (lajang, berduka, bekerja, dsb).
        2. BUANG KONTEKS SAMPAH: Anda WAJIB MENGABAIKAN ayat dalam konteks yang secara kata mirip, tetapi salah sasaran secara situasi (Contoh: masalah bos di kantor jangan dijawab pakai ayat konflik orang tua-anak). Fokus HANYA pada ayat yang disajikan yang relevan dengan 5W1H pengguna.
        """        
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("placeholder", "{chat_history}"),
            ("human", "Ayat Alkitab:\n{context}\n\nPERTANYAAN PENGGUNA: {question}\n\nKALIMAT PENGANTAR JAWABAN:")        
        ])

        chain = prompt | self.llm | StrOutputParser()
        memory_chain = RunnableWithMessageHistory(
            runnable = chain,
            get_session_history = self.get_session_history,
            input_messages_key = "question",
            history_messages_key = "chat_history"
        )

        ringkasan_llm = await memory_chain.ainvoke({"context": final_retrieval, 
                                      "question": user_query},
                                      config = {"configurable": {"session_id": session_id}}
                                      )
        ringkasan_llm = ringkasan_llm.strip()

        jawaban_final = ""
        for doc in final_retrieval:
            sumber = doc.metadata.get('sumber', '-')
            isi_ayat = doc.page_content
            jawaban_final += f"- **{sumber}**: \"{isi_ayat}\"\n"
            jawaban_final += "\n"

        if not ringkasan_llm:
            ringkasan_llm = "Prinsip Alkitab yang sesuai pertanyaan anda dapat dilihat pada referensi ayat berikut:"
        jawaban_final += f"{ringkasan_llm}\n\n"

        return jawaban_final

    # 1st MENU: Digital Bible reading   
    def getDaftarKitab(self, id:int) -> list[str]:
        return self._retriever.requestDaftarKitab(id)
    
    def getDaftarPasal(self, kitab:str) -> list[int]:
        return self._retriever.requestListPasal(kitab)
    
    def getDaftarAyat(self, kitab: str, pasal:int) -> list[int]:
        return self._retriever.requestListAyat(kitab, pasal)
    
    def getTeksAyat(self, kitab: str, pasal: int, ayat:int) -> str:
        return self._retriever.requestTeksAyat(kitab, pasal, ayat)