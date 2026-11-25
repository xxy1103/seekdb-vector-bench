"""Real 384-D neural embedding (all-MiniLM-L6-v2) product_catalog corpus generator."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

CATALOG_PROFILES = [
    (
        "running_shoes",
        "RS",
        [
            "marathon road training shoe with nitrogen-infused midsole foam and heel impact absorption",
            "lightweight carbon-plated racing trainer built for tempo runs on asphalt tracks",
            "breathable engineered mesh daily jogger offering arch stability for flat-footed runners",
            "high-cushion recovery road runner designed to reduce knee joint stress on concrete",
        ],
        ["carbon plate", "nitrogen foam", "8mm drop", "engineered mesh"],
        [
            "soft sole sneakers for long pavement runs and joint pain relief",
            "fast competition footwear built for tarmac speed workouts",
            "airy everyday jogging trainers stabilizing inward foot roll",
            "plush distance running kicks absorbing hard street impact",
        ],
    ),
    (
        "trail_boots",
        "TB",
        [
            "waterproof alpine hiking boot featuring deep Vibram mud-shedding lugs and ankle brace",
            "rock-plate protected mountain trail shoe engineered for wet scree and technical descents",
            "insulated winter trekking boot with thermal lining for sub-zero backcountry trails",
            "ultralight fast-packing mid-cut boot combining Gore-Tex membrane and toe bumper",
        ],
        ["Vibram Megagrip", "Gore-Tex membrane", "rock plate", "ankle cuff"],
        [
            "rainproof alpine footwear with deep outsole cleats for muddy slopes",
            "rugged mountain path sneakers shielding feet on slippery boulders",
            "cold weather expedition footwear keeping toes warm in snow",
            "mid-height breathable rainproof trekker for fast ridge scrambles",
        ],
    ),
    (
        "cycling_gear",
        "CG",
        [
            "aerodynamic MIPS road bike helmet with internal airflow channels and magnetic buckle",
            "carbon-sole clipless cycling shoe compatible with three-bolt road pedal cleats",
            "rechargeable 1200-lumen handlebar bike headlight with cutoff beam for night commuting",
            "insulated thermal bib tights with multi-density chamois pad for winter gravel rides",
        ],
        ["MIPS liner", "carbon sole", "1200lm USB-C", "thermal fleece"],
        [
            "breathable streamlined bicycle headgear preventing rotational brain injury",
            "rigid sole road biking footwear locking onto three-bolt cleats",
            "high output rechargeable bicycle front lamp for evening street commute",
            "warm cushioned winter bicycling leggings for cold dirt road rides",
        ],
    ),
    (
        "smart_watch",
        "SW",
        [
            "dual-frequency GPS multisport watch tracking heart-rate variability and lactate threshold",
            "solar-charging titanium outdoor watch featuring offline topographic contour maps",
            "compact AMOLED fitness tracker monitoring sleep stages and blood oxygen saturation",
            "triathlon swim-proof sports watch with open-water stroke detection and barometric altimeter",
        ],
        ["dual-band GPS", "titanium bezel", "sapphire glass", "14-day battery"],
        [
            "athlete wrist wearable logging pulse recovery and anaerobic effort",
            "sunlit powered rugged expedition timepiece with built-in terrain charts",
            "slim bright screen health band monitoring nightly rest and oxygen",
            "watertight swim-bike-run wrist computer with altitude pressure sensor",
        ],
    ),
    (
        "fitness_apparel",
        "FA",
        [
            "seamless moisture-wicking compression baselayer top for high-intensity gym intervals",
            "packable windproof ripstop running shell jacket with reflective night safety trim",
            "four-way stretch quick-dry training shorts with zippered phone pocket",
            "merino-wool blend thermal quarter-zip pullover regulating warmth during cold warmups",
        ],
        ["four-way stretch", "ripstop nylon", "merino blend", "moisture-wicking"],
        [
            "sweat-wicking tight workout shirt for intense strength circuits",
            "compact windbreaker outer layer for evening runs with safety stripes",
            "stretchy fast drying gym trunks with zippered smartphone pocket",
            "warm wool midlayer half-zip sweater for chilly dawn stretches",
        ],
    ),
    (
        "camping_tent",
        "CT",
        [
            "freestanding two-person backpacking tent with DAC aluminum poles and 3000mm rainfly",
            "four-season geodesic mountaineering shelter built to withstand 90 km/h ridge winds",
            "ultralight single-wall dyneema trekking-pole shelter weighing under 600 grams",
            "family four-person car-camping cabin tent with standing headroom and mesh vestibule",
        ],
        ["DAC aluminum", "3000mm ripstop", "geodesic frame", "mesh vestibule"],
        [
            "two sleeper portable camp shelter with metal poles and rain cover",
            "heavy duty winter alpine expedition bivouac withstanding severe gales",
            "ultralight minimalist tarp shelter pitched using hiking poles",
            "tall four sleeper family campsite lodge with bug mesh porch",
        ],
    ),
    (
        "climbing_rope",
        "CR",
        [
            "9.5mm UIAA dry-treated single dynamic climbing rope with middle mark for sport lead",
            "twin-certified 7.9mm ice and alpine half rope resisting sharp granite edge abrasion",
            "static 10.5mm kernmantle rescue and canyoneering line with low elongation",
            "ergonomic assisted-braking belay device paired with triple-lock HMS carabiner",
        ],
        ["UIAA dry-treat", "9.5mm dynamic", "kernmantle", "HMS triple-lock"],
        [
            "water-repellent lead cord with center indicator for outdoor rock routes",
            "slender twin glacier cord resisting frozen moisture and sharp granite",
            "low-elongation static cord for rappelling and rescue hauling",
            "auto-braking belay hardware bundled with twist-lock safety D-ring",
        ],
    ),
]


@dataclass
class ProductCorpus:
    """Real 384-D neural embedding product_catalog corpus with attributes and SKU codes."""

    ids: np.ndarray
    unit_vectors: np.ndarray         # float32 (N, 384) real L2-normalized all-MiniLM-L6-v2
    weighted_vectors: np.ndarray     # float32 (N, 384) unit_vectors * popularity_weight (for metric test)
    popularity_weights: np.ndarray   # float32 (N,)
    categories: list[str]
    prices: np.ndarray               # float32 (N,)
    in_stock: np.ndarray             # bool (N,)
    sku_codes: list[str]
    profile_ids: np.ndarray          # (N,) 0..27 exact sub-profile ground truth
    texts: list[str]

    query_unit_vectors: np.ndarray   # float32 (Q, 384) real embeddings of query_texts
    query_weighted_vectors: np.ndarray
    query_texts: list[str]
    query_types: list[str]           # "semantic", "exact_sku", "compound"
    query_target_cat: list[str]
    query_target_profile: np.ndarray
    query_target_sku: list[str]
    query_price_cap: np.ndarray


def generate_product_corpus(
    n_docs: int = 2000,
    seed: int = 42,
    cache_path: Path | None = None,
) -> ProductCorpus:
    """
    Generate (or load cached) N=2,000 product_catalog corpus with real 384-D all-MiniLM-L6-v2
    neural embeddings and 120 realistic queries (40 semantic, 40 exact_sku, 40 compound).
    """
    if cache_path is None:
        cache_path = Path(__file__).resolve().parents[2] / "results" / "corpus_cache_384d.npz"

    rng = np.random.default_rng(seed)

    categories: list[str] = []
    prices_list: list[float] = []
    in_stock_list: list[bool] = []
    sku_codes: list[str] = []
    profile_ids_list: list[int] = []
    texts: list[str] = []

    cat_probs = [0.28, 0.22, 0.18, 0.14, 0.10, 0.05, 0.03]
    for i in range(n_docs):
        c_idx = int(rng.choice(len(CATALOG_PROFILES), p=cat_probs))
        p_idx = int(rng.integers(0, 4))
        cat_name, prefix, phrases, specs, _ = CATALOG_PROFILES[c_idx]

        # Unique SKU per product (or shared pair every 20 items) so exact SKU queries target 1-2 docs
        sku_num = 1000 + (i // 2 if i % 20 < 2 else i)
        sku = f"SKU-{prefix}-{sku_num}"
        price = round(float(rng.uniform(89.0, 1899.0)), 2)
        stock = bool(rng.random() < 0.80)
        spec_tag = specs[p_idx]
        variant = ["Pro", "Lite", "Max", "GTX", "SE"][i % 5]
        year = 2024 + (i % 3)

        categories.append(cat_name)
        prices_list.append(price)
        in_stock_list.append(stock)
        sku_codes.append(sku)
        profile_ids_list.append(c_idx * 4 + p_idx)
        texts.append(
            f"{cat_name.replace('_', ' ')} {variant} {year} {sku}: "
            f"{phrases[p_idx]} with {spec_tag}, price ${price:.0f}"
        )

    q_texts: list[str] = []
    q_types: list[str] = []
    q_target_cat: list[str] = []
    q_target_profile: list[int] = []
    q_target_sku: list[str] = []
    q_price_cap: list[float] = []

    for qi in range(40):
        c_idx = qi % len(CATALOG_PROFILES)
        p_idx = (qi // len(CATALOG_PROFILES)) % 4
        cat_name, _, _, _, sem_queries = CATALOG_PROFILES[c_idx]
        q_texts.append(sem_queries[p_idx])
        q_types.append("semantic")
        q_target_cat.append(cat_name)
        q_target_profile.append(c_idx * 4 + p_idx)
        q_target_sku.append("NONE")
        q_price_cap.append(9999.0)

    chosen_sku_docs = rng.choice(n_docs, size=40, replace=False)
    for doc_i in chosen_sku_docs:
        sku = sku_codes[int(doc_i)]
        q_texts.append(f"compatibility datasheet and warranty replacement for {sku}")
        q_types.append("exact_sku")
        q_target_cat.append(categories[int(doc_i)])
        q_target_profile.append(profile_ids_list[int(doc_i)])
        q_target_sku.append(sku)
        q_price_cap.append(9999.0)

    for qi in range(40):
        c_idx = qi % 5
        p_idx = (qi // 5) % 4
        cat_name, _, _, specs, sem_queries = CATALOG_PROFILES[c_idx]
        matching = [
            idx for idx in range(n_docs)
            if categories[idx] == cat_name and profile_ids_list[idx] == c_idx * 4 + p_idx and prices_list[idx] < 950.0
        ]
        target_doc = matching[0] if matching else int(rng.choice([idx for idx in range(n_docs) if categories[idx] == cat_name]))
        sku = sku_codes[target_doc]
        q_texts.append(f"{sem_queries[p_idx]} {specs[p_idx]} {sku}")
        q_types.append("compound")
        q_target_cat.append(cat_name)
        q_target_profile.append(c_idx * 4 + p_idx)
        q_target_sku.append(sku)
        q_price_cap.append(950.0)

    sig_hash = hashlib.sha256(("\n".join(texts[:20] + q_texts[:20])).encode("utf-8")).hexdigest()[:16]

    doc_vecs: np.ndarray | None = None
    q_vecs: np.ndarray | None = None
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        if str(data["sig_hash"]) == sig_hash and data["doc_vecs"].shape == (n_docs, 384) and data["q_vecs"].shape == (len(q_texts), 384):
            doc_vecs = data["doc_vecs"].astype(np.float32)
            q_vecs = data["q_vecs"].astype(np.float32)

    if doc_vecs is None or q_vecs is None:
        import pyseekdb
        ef = pyseekdb.DefaultEmbeddingFunction()
        doc_chunks = []
        for start in range(0, len(texts), 128):
            doc_chunks.append(np.array(ef(texts[start : start + 128]), dtype=np.float32))
        doc_vecs = np.vstack(doc_chunks)
        q_vecs = np.array(ef(q_texts), dtype=np.float32)

        doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, sig_hash=np.array(sig_hash), doc_vecs=doc_vecs, q_vecs=q_vecs)

    pop_weights = rng.lognormal(mean=0.3, sigma=0.55, size=n_docs).astype(np.float32)
    weighted_vecs = (doc_vecs * pop_weights[:, None]).astype(np.float32)
    q_scales = rng.uniform(0.7, 2.3, size=len(q_texts)).astype(np.float32)
    q_weighted = (q_vecs * q_scales[:, None]).astype(np.float32)

    return ProductCorpus(
        ids=np.arange(1, n_docs + 1, dtype=np.int64),
        unit_vectors=doc_vecs,
        weighted_vectors=weighted_vecs,
        popularity_weights=pop_weights,
        categories=categories,
        prices=np.array(prices_list, dtype=np.float32),
        in_stock=np.array(in_stock_list, dtype=bool),
        sku_codes=sku_codes,
        profile_ids=np.array(profile_ids_list, dtype=np.int64),
        texts=texts,
        query_unit_vectors=q_vecs,
        query_weighted_vectors=q_weighted,
        query_texts=q_texts,
        query_types=q_types,
        query_target_cat=q_target_cat,
        query_target_profile=np.array(q_target_profile, dtype=np.int64),
        query_target_sku=q_target_sku,
        query_price_cap=np.array(q_price_cap, dtype=np.float32),
    )
