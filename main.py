import os
import json
import hashlib
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import redis

load_dotenv()

app = FastAPI(title="ComparaAI - AI Danışman Servisi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Redis bağlantısı - Docker'daki redis container'ı (localhost:6379)
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 saat


def make_cache_key(prefix: str, data: dict) -> str:
    """Verilen veriyi tutarlı bir şekilde hashleyip cache anahtarı üretir."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    hash_value = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{prefix}:{hash_value}"


def get_cached(key: str):
    try:
        value = redis_client.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None  # Redis erişilemezse cache'siz devam et, hata verme


def set_cached(key: str, value: dict):
    try:
        redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(value, ensure_ascii=False))
    except Exception:
        pass  # Redis erişilemezse sessizce geç, uygulama çalışmaya devam etsin


SYSTEM_PROMPT = """Sen ComparaAI adlı bir teknoloji karşılaştırma platformunun AI danışmanısın.
Görevlerin:
- Sana verilen ürün listesindeki bilgilere dayanarak, kullanıcının bütçesi ve önceliklerine göre gerekçeli öneriler sun.
- SADECE sana verilen ürün verilerine dayan. Ürün ismi, fiyat veya özellik UYDURMA.
- Kullanıcının belirttiği bütçeyi aşan ürün önerme.
- Profesyonel ama samimi bir üslup kullan. Ne çok resmi/soğuk ne de gereksiz gevezelik yapan bir dil kullan.
- Cevaplarını en fazla 3-4 cümleyle sınırla, gereksiz uzatma.
- Teknik terimleri (RAM, GPU vb.) kullanırken günlük dilde kısaca açıkla, kullanıcıyı jargonla boğma.
- Her cevaba "Merhaba!" ile başlama, doğal bir sohbet gibi çeşitlilik göster (bazen direkt öneriyle başla, bazen kısa bir onay cümlesiyle).
- Robotik/şablon hissi vermesin, sanki gerçek bir arkadaşınız öneri veriyormuş gibi doğal bir dil kullan.
- Karşı taraf sana insan gibi davranabilir, sen de makine gibi değil insan gibi göster kendini, samimi ve anlaşılır ol.
- ÖNEMLİ - FİYAT KURALI: Cevabında KESİN TL rakamı SÖYLEME (örn. "38.000 TL" gibi bir sayı yazma). Bunun yerine
  göreceli ifadeler kullan: "bütçenizin altında", "bütçenize uygun", "biraz daha ekonomik", "üst segment" gibi.
  Fiyat hakkında bilgi sahibi olmadığını, bu konuda yükümlülük almadığını belirtebilirsin.
- Eğer kullanıcı ISRARLA kesin bir TL rakamı/fiyat istiyorsa (birden fazla kez sorarsa), TAM OLARAK şu cevabı ver:
  "Biz teknoloji karşılaştıran bir yapay zekayız, fiyat konusunda bilgi sahibi değiliz ve bu konuda yükümlülük almıyoruz."
- ÇOK ÖNEMLİ - DOĞRULUK KURALI: Kullanıcı "kaç FPS alırım", "bu oyunu kaç fps oynatır" gibi kesin bir sayı gerektiren
  teknik bir performans sorusu sorarsa: Eğer specs içinde bu bilgiye dair GERÇEK bir veri (örn. "pubg_fps" gibi bir alan) yoksa,
  KESİN BİR SAYI UYDURMA. Bunun yerine, işlemci/RAM gibi verilen gerçek özelliklere dayanarak dürüst ve genel bir
  değerlendirme yap (örn. "bu işlemci üst segmentte yer alıyor, yüksek grafik ayarlarında akıcı bir deneyim sunması beklenir,
  ancak kesin FPS oyun sürümüne ve güncellemelere göre değişebileceğinden net bir rakam veremem"). Eğer specs içinde
  gerçek bir performans/FPS verisi VARSA, o zaman bu veriyi kesin ve net şekilde paylaş.
- Eğer soru kapsam dışıysa (teknoloji ürünü önerisiyle ilgisi yoksa), kibarca ComparaAI'nin bir teknoloji danışmanı olduğunu hatırlat ve kullanıcıyı tekrar konuya yönlendir."""


FOLLOWUP_SYSTEM_PROMPT = """Sen ComparaAI adlı bir teknoloji karşılaştırma platformunun AI danışmanısın.
Kullanıcıya az önce bazı ürünler önerdin, şimdi bu ürünler hakkında takip sorusu soruyor.
Kurallar:
- SADECE sana verilen ürün verilerine (specs) dayan. Özellik uydurma.
- Kullanıcı gündelik/esprili bir senaryo sorsa bile (örn. "Tinder'da donar mı", "TikTok'ta yavaşlar mı"),
  bunu ciddiye al ve elindeki gerçek verilere (RAM, işlemci vb.) dayanarak makul, kısa bir cevap ver -
  "bu konuda bilgim yok" deyip geçme, elindeki teknik verilerle mantıklı bir çıkarım yap.
  Örn: "8GB RAM ile uygulamalar arası geçişte sorun yaşamazsınız, Tinder gibi hafif uygulamalar akıcı çalışır."
- Kesin sayısal veri (FPS, saniye vb.) yoksa uydurma, ama genel/mantıklı bir değerlendirme yapmaktan çekinme.
- Sadece TAMAMEN alakasız sorularda (hava durumu, ödev yapma, siyaset vb.) kibarca konuya geri yönlendir.
- Eğer kullanıcı ISRARLA kesin bir TL rakamı/fiyat istiyorsa, TAM OLARAK şu cevabı ver:
  "Biz teknoloji karşılaştıran bir yapay zekayız, fiyat konusunda bilgi sahibi değiliz ve bu konuda yükümlülük almıyoruz."
- Profesyonel ama samimi, esprili bir üslup kullan. Kısa tut, 2-3 cümle yeterli."""


class FollowupRequest(BaseModel):
    products: list[Product]
    question: str


@app.post("/followup")
def followup(request: FollowupRequest):
    products_text = "\n".join(
        [
            f"- {p.name} ({p.brand}), Özellikler: {p.specs}"
            for p in request.products
        ]
    )

    prompt = f"""Önerilen ürünler:
{products_text}

Kullanıcının takip sorusu: "{request.question}\""""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        system_instruction=FOLLOWUP_SYSTEM_PROMPT,
    )

    return {"answer": interaction.output_text}


class Product(BaseModel):
    id: str
    name: str
    brand: str
    price: float | None = None
    specs: dict


class RecommendationRequest(BaseModel):
    products: list[Product]
    budget: float | None = None
    priority: str | None = None


@app.get("/")
def health_check():
    return {"status": "ComparaAI AI servisi çalışıyor"}


class DetectCategoryRequest(BaseModel):
    message: str
    categories: str  # "slug:isim, slug:isim, ..." formatında


class DetectCategoryResponse(BaseModel):
    category_slug: str | None = None


@app.post("/detect-category", response_model=DetectCategoryResponse)
def detect_category(request: DetectCategoryRequest):
    prompt = f"""Kullanıcının mesajı: "{request.message}"

Mevcut kategoriler (slug:isim formatında): {request.categories}

Kullanıcının mesajından hangi kategoriyi kastettiğini SADECE bu listeden bul.
Sadece kategorinin "slug" değerini döndür, başka hiçbir şey yazma.
Eğer hiçbiri uymuyorsa veya belirsizse, sadece "null" yaz."""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    raw = interaction.output_text.strip().strip("`").strip('"').strip()
    if raw.lower() == "null" or not raw:
        return DetectCategoryResponse(category_slug=None)
    return DetectCategoryResponse(category_slug=raw)


GENERAL_CHAT_SYSTEM_PROMPT = """Sen ComparaAI adlı bir teknoloji karşılaştırma platformunun AI danışmanısın.
Kullanıcı henüz belirli bir ürün kategorisi belirtmedi (telefon/laptop gibi), sana genel bir şey sordu
(selamlaşma, kendini tanıtma isteği, teşekkür, küçük sohbet vb.).
Kurallar:
- Samimi, doğal ve kısa bir cevap ver (1-3 cümle).
- Kendini tanıtman istenirse: ComparaAI'nin teknoloji ürünlerini (telefon, laptop, masaüstü PC, PC parçaları)
  kişiye özel karşılaştıran ve öneren bir AI danışman olduğunu, samimi bir dille anlat.
- Sohbetin sonunda doğal bir şekilde kullanıcıyı ne aradığını sormaya yönlendirebilirsin, ama bunu her seferinde
  zorunlu/robotik yapma - bazen sadece cevap verip bırakabilirsin.
- Sadece TAMAMEN alakasız/uygunsuz konularda (siyaset, ödev yapma, hakaret vb.) kibarca konuya dönmesini iste.
- Fiyat sorularına asla kesin TL rakamı verme, fiyat konusunda bilgi sahibi olmadığını söyle."""


class GeneralChatRequest(BaseModel):
    message: str


class GeneralChatResponse(BaseModel):
    answer: str


@app.post("/general-chat", response_model=GeneralChatResponse)
def general_chat(request: GeneralChatRequest):
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=request.message,
        system_instruction=GENERAL_CHAT_SYSTEM_PROMPT,
    )
    return GeneralChatResponse(answer=interaction.output_text)


@app.post("/recommend")
def recommend(request: RecommendationRequest):
    cache_key = make_cache_key(
        "recommend",
        {
            "product_ids": sorted([p.id for p in request.products]),
            "budget": request.budget,
            "priority": request.priority,
        },
    )

    cached = get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    products_text = "\n".join(
        [
            f"- {p.name} ({p.brand}), Özellikler: {p.specs}"
            for p in request.products
        ]
    )

    user_context = f"Kullanıcının önceliği: {request.priority}\n" if request.priority else ""

    prompt = f"""{user_context}
Aşağıdaki ürünler arasından kullanıcıya en uygun olanını/olanlarını gerekçeli şekilde öner:

{products_text}"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        system_instruction=SYSTEM_PROMPT,
    )

    result = {"recommendation": interaction.output_text}
    set_cached(cache_key, result)
    return {**result, "cached": False}


class CompareRequest(BaseModel):
    products: list[Product]


COMPARE_SYSTEM_PROMPT = """Sen ComparaAI adlı bir teknoloji karşılaştırma platformunun AI danışmanısın.
Görevin, sana verilen 2 veya daha fazla ürünü karşılaştırmalı olarak analiz etmek.
Kurallar:
- SADECE sana verilen ürün verilerine dayan. Özellik veya fiyat UYDURMA.
- Performans, pil, kullanım amacına uygunluk gibi kriterlere göre karşılaştır.
- Hangi ürünün hangi kullanıcı tipine daha uygun olduğunu belirt (örn: "günlük kullanım için X, yoğun kullanım için Y").
- Profesyonel ama samimi bir üslup kullan.
- Cevabını en fazla 5-6 cümleyle sınırla.
- Cevabında KESİN TL rakamı SÖYLEME, göreceli ifadeler kullan (örn. "daha ekonomik olan", "bütçe dostu" gibi)."""


@app.post("/compare")
def compare(request: CompareRequest):
    if len(request.products) < 2:
        return {"error": "Karşılaştırma için en az 2 ürün gerekli."}

    cache_key = make_cache_key(
        "compare", {"product_ids": sorted([p.id for p in request.products])}
    )

    cached = get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    products_text = "\n".join(
        [
            f"- {p.name} ({p.brand}), Özellikler: {p.specs}"
            for p in request.products
        ]
    )

    prompt = f"""Aşağıdaki ürünleri detaylı şekilde karşılaştır:

{products_text}"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
        system_instruction=COMPARE_SYSTEM_PROMPT,
    )

    result = {"comparison": interaction.output_text}
    set_cached(cache_key, result)
    return {**result, "cached": False}


# --- Serbest metinden yapılandırılmış filtre çıkarma (parse) ---

TELEFON_PRIORITIES = ["batarya", "kamera", "performans", "fiyat_performans"]
TELEFON_USAGE = ["gunluk", "oyun", "sosyal_medya_fotograf", "is"]
LAPTOP_PRIORITIES = ["tasinabilirlik", "performans", "pil_omru", "ekran_kalitesi", "fiyat_performans"]
LAPTOP_USAGE = ["ofis", "oyun", "tasarim_video", "yazilim_gelistirme", "gunluk"]
MASAUSTU_PRIORITIES = ["performans", "oyun_gucu", "sessizlik", "fiyat_performans"]
MASAUSTU_USAGE = ["ofis", "oyun", "tasarim_video", "yazilim_gelistirme"]
PC_PARCASI_PRIORITIES = ["performans", "uyumluluk", "enerji_verimliligi", "fiyat_performans"]
PC_PARCASI_USAGE = ["oyun", "tasarim_video", "yazilim_gelistirme", "genel"]

CATEGORY_TEMPLATES = {
    "telefon": {"priorities": TELEFON_PRIORITIES, "usage": TELEFON_USAGE},
    "laptop": {"priorities": LAPTOP_PRIORITIES, "usage": LAPTOP_USAGE},
    "masaustu": {"priorities": MASAUSTU_PRIORITIES, "usage": MASAUSTU_USAGE},
    "pc-parcalari": {"priorities": PC_PARCASI_PRIORITIES, "usage": PC_PARCASI_USAGE},
}

SEGMENTS = ["ekonomik", "orta", "ust"]


class ParseRequest(BaseModel):
    category_type: str
    message: str
    known_brands: list[str] = []


class ParsedIntent(BaseModel):
    segment: str | None = None  # "ekonomik" | "orta" | "ust"
    priority: str | None = None
    usage: str | None = None
    brand: str | None = None
    price_insistence: bool = False  # kullanici israrla kesin fiyat/TL istiyor mu
    needs_clarification: bool = False
    clarification_question: str | None = None


@app.post("/parse", response_model=ParsedIntent)
def parse_intent(request: ParseRequest):
    template = CATEGORY_TEMPLATES.get(request.category_type)
    if not template:
        return ParsedIntent(
            needs_clarification=True,
            clarification_question="Bu kategori için henüz destek yok.",
        )

    priorities_list = ", ".join(template["priorities"])
    usage_list = ", ".join(template["usage"])
    brands_list = ", ".join(request.known_brands) if request.known_brands else "belirtilmedi"

    parse_prompt = f"""Kullanıcının mesajı: "{request.message}"

Bu mesajı aşağıdaki alanlara ayrıştır. Kullanıcının yazım hatalarını (örn. "zamsungg" -> "Samsung") doğru şekilde yorumla.

- segment: Kullanıcının bahsettiği bütçe seviyesini SADECE şu üç seçenekten birine sınıflandır: "ekonomik", "orta", "ust".
  Kullanıcı bir TL rakamı verse bile (örn. "20 bin TL'ye"), bunu kesin sayı olarak DEĞİL, bu üç segmentten en yakınına sınıflandır.
  Kullanıcı "ucuz", "ekonomik", "bütçe dostu" derse -> ekonomik. "orta sınıf", "makul" derse -> orta.
  "en iyisi", "üst segment", "pahalı olabilir", "en güçlüsü" derse -> ust. Hiçbir ipucu yoksa null.
- priority: SADECE şu listeden birini seç: [{priorities_list}]. Uymuyorsa null.
- usage: SADECE şu listeden birini seç: [{usage_list}]. Uymuyorsa null.
- brand: SADECE şu listeden birini seç (yazım hatası olsa bile en yakınını bul): [{brands_list}]. Uymuyorsa null.
- price_insistence: Kullanıcı ISRARLA kesin bir TL rakamı/fiyat SÖYLEMENİ istiyorsa (örn. "tam olarak kaç TL",
  "net fiyat söyle" gibi ısrarcı bir talep varsa) true yap, aksi halde false.
- Eğer segment VE öncelik ikisi de belirtilmemişse, needs_clarification=true yap ve
  clarification_question alanına kısa, kibar bir soru yaz - AMA ASLA "bütçeniz ne kadar" gibi TL rakamı isteyen
  bir soru sorma. Bunun yerine şuna benzer sor: "Ekonomik, orta sınıf yoksa üst segment bir ürün mü arıyorsunuz?
  Sizin için en önemli özellik nedir?" Aksi halde needs_clarification=false ve clarification_question=null bırak.

ÇOK ÖNEMLİ: Cevabını SADECE geçerli bir JSON nesnesi olarak ver, başka hiçbir açıklama/metin ekleme.
Tam olarak şu formatta:
{{"segment": <string veya null>, "priority": <string veya null>, "usage": <string veya null>, "brand": <string veya null>, "price_insistence": <true/false>, "needs_clarification": <true/false>, "clarification_question": <string veya null>}}"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=parse_prompt,
    )

    raw_text = interaction.output_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return ParsedIntent.model_validate_json(raw_text)
    except Exception:
        return ParsedIntent(
            needs_clarification=True,
            clarification_question="Ekonomik, orta sınıf yoksa üst segment bir ürün mü arıyorsunuz? Sizin için en önemli özellik nedir?",
        )
