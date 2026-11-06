
import streamlit as st
import torch
import requests
import re
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from ultralytics import YOLO
from collections import Counter
from urllib.parse import urlencode

# =====================================================
# 1. PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="VisionChat",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ VisionChat")
st.caption(
    "AI Image Understanding, Object Detection "
    "and Smart Product Shopping Assistant"
)

st.info(
    "Local AI image analysis with online shopping search. "
    "No separate AI API key is needed."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    torch.set_num_threads(4)
except Exception:
    pass


# =====================================================
# 2. LOAD IMAGE DESCRIPTION MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading image model...")
def load_caption():

    name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(name)

    model = BlipForConditionalGeneration.from_pretrained(name)

    model.to(DEVICE)
    model.eval()

    return processor, model


# =====================================================
# 3. LOAD IMAGE QUESTION MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading question model...")
def load_vqa():

    name = "Salesforce/blip-vqa-base"

    processor = BlipProcessor.from_pretrained(name)

    model = BlipForQuestionAnswering.from_pretrained(name)

    model.to(DEVICE)
    model.eval()

    return processor, model


# =====================================================
# 4. LOAD OBJECT DETECTION MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading object detector...")
def load_yolo():

    return YOLO("yolo11n.pt")


# =====================================================
# 5. IMAGE PREPROCESSING
# =====================================================

def prepare_image(image, max_side=320):

    image = image.convert("RGB")

    image.thumbnail(
        (max_side, max_side),
        Image.Resampling.BILINEAR
    )

    return image


# =====================================================
# 6. IMAGE DESCRIPTION
# =====================================================

def caption_image(image):

    processor, model = load_caption()

    small_image = prepare_image(image)

    inputs = processor(
        images=small_image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=25,
            num_beams=1
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


# =====================================================
# 7. IMAGE QUESTION ANSWERING
# =====================================================

def ask_image(image, question):

    processor, model = load_vqa()

    small_image = prepare_image(image)

    inputs = processor(
        small_image,
        question,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=15,
            num_beams=1
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


# =====================================================
# 8. GENERATE AUTOMATIC PRODUCT SEARCH QUERY
# =====================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "on", "in", "at", "with", "of", "and", "or",
    "this", "that", "it", "there", "here",
    "image", "picture", "photo", "photograph",
    "showing", "shown", "looks", "like",
    "sitting", "standing", "placed", "lying",
    "holding", "background", "table", "surface",
    "yellow", "white", "black", "blue", "red",
    "green", "pink", "brown", "wooden",
    "color", "colour", "visible"
}

QUESTION_WORDS = {
    "can", "could", "would", "will", "please",
    "give", "find", "show", "tell", "where",
    "what", "which", "how", "me", "buy",
    "link", "price", "purchase", "get"
}


def generate_product_query(description):

    if not description:
        return ""

    # Remove punctuation and convert to lowercase.
    text = re.sub(
        r"[^a-zA-Z0-9\s-]",
        " ",
        description.lower()
    )

    words = text.split()

    # Remove words describing the background
    # and unnecessary sentence words.
    keywords = [
        word for word in words
        if word not in STOPWORDS
        and word not in QUESTION_WORDS
        and len(word) > 1
    ]

    # Remove duplicate words while retaining order.
    keywords = list(dict.fromkeys(keywords))

    # Keep the search focused.
    keywords = keywords[:5]

    return " ".join(keywords)


def clean_user_query(query, automatic_query):

    if not query or not query.strip():
        return automatic_query

    text = query.strip().lower()

    words = set(re.findall(r"[a-zA-Z]+", text))

    # Reject conversational requests instead of
    # accidentally searching the entire question.
    if words.intersection(QUESTION_WORDS):
        return automatic_query

    if len(text.split()) > 8:
        return automatic_query

    return query.strip()


# =====================================================
# 9. OBJECT DETECTION
# =====================================================

def detect_objects(image):

    model = load_yolo()

    result = model.predict(
        source=image,
        conf=0.25,
        imgsz=416,
        verbose=False
    )[0]

    objects = []

    for box in result.boxes:

        class_id = int(box.cls.item())

        objects.append({
            "Object": result.names[class_id],
            "Confidence": f"{float(box.conf.item()):.1%}",
            "Coordinates": [
                round(float(v), 1)
                for v in box.xyxy[0].tolist()
            ]
        })

    marked_image = Image.fromarray(
        result.plot()[:, :, ::-1]
    )

    return objects, marked_image


# =====================================================
# 10. GOOGLE SHOPPING SEARCH LINK
# =====================================================

def google_shopping_link(query):

    params = urlencode({
        "tbm": "shop",
        "q": query,
        "gl": "in",
        "hl": "en"
    })

    return "https://www.google.com/search?" + params


# =====================================================
# 11. LIVE SHOPPING SEARCH
# =====================================================

def search_products(query, api_key):

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": "in",
        "hl": "en",
        "location": "Pune, Maharashtra, India",
        "api_key": api_key
    }

    response = requests.get(
        url,
        params=params,
        timeout=25
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise ValueError(data["error"])

    return data.get("shopping_results", [])


# =====================================================
# 12. FILTER IRRELEVANT PRODUCTS
# =====================================================

def filter_products(products, query):

    query_words = set(
        re.findall(r"[a-zA-Z0-9]+", query.lower())
    )

    query_words = {
        word for word in query_words
        if len(word) > 2
    }

    if not query_words:
        return []

    matched = []

    for product in products:

        title = product.get("title", "").lower()

        title_words = set(
            re.findall(r"[a-zA-Z0-9]+", title)
        )

        matching_words = query_words.intersection(title_words)

        # Calculate relevance.
        score = len(matching_words) / len(query_words)

        if score >= 0.25:

            product["_relevance"] = score

            matched.append(product)

    matched.sort(
        key=lambda item: item["_relevance"],
        reverse=True
    )

    return matched[:12]


# =====================================================
# 13. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("Shopping Configuration")

api_key = st.sidebar.text_input(
    "SerpAPI Key",
    type="password",
    help="Required for live product listings and prices."
)

st.sidebar.caption(
    "Keep your API key private."
)

st.sidebar.divider()

st.sidebar.write("Processing Device:", DEVICE.upper())

st.sidebar.write("Shopping Region: India")


# =====================================================
# 14. UPLOAD IMAGE
# =====================================================

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.subheader("Image Information")

        st.write(
            f"**Dimensions:** {image.width} × {image.height}"
        )

        st.write(
            f"**File Size:** {uploaded.size / 1024:.1f} KB"
        )

        st.write(
            f"**Format:** {uploaded.type}"
        )

    # -------------------------------------------------
    # FOUR APPLICATION FEATURES
    # -------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Image Analysis",
        "🔎 Object Detection",
        "💬 Ask About Image",
        "🛍️ Product Shopping"
    ])

    # =================================================
    # TAB 1: IMAGE ANALYSIS
    # =================================================

    with tab1:

        st.subheader("AI Image Description")

        if st.button(
            "Analyze Image",
            type="primary"
        ):

            try:

                with st.spinner("Analyzing image..."):

                    description = caption_image(image)

                st.session_state["description"] = description

            except Exception as error:

                st.error(str(error))

        if "description" in st.session_state:

            st.success(
                st.session_state["description"]
            )

            st.caption(
                "The description is generated by a local AI model "
                "and may not identify every detail."
            )

    # =================================================
    # TAB 2: OBJECT DETECTION
    # =================================================

    with tab2:

        st.subheader("Object Detection")

        if st.button("Detect Objects"):

            try:

                with st.spinner("Detecting objects..."):

                    objects, marked = detect_objects(image)

                if objects:

                    st.image(
                        marked,
                        caption="Detected Objects",
                        use_container_width=True
                    )

                    counts = Counter(
                        item["Object"]
                        for item in objects
                    )

                    st.subheader("Object Summary")

                    for name, count in counts.items():

                        st.write(f"- {name}: {count}")

                    st.subheader("Detection Details")

                    st.dataframe(
                        objects,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.warning("No supported objects detected.")

            except Exception as error:

                st.error(str(error))

    # =================================================
    # TAB 3: ASK ABOUT IMAGE
    # =================================================

    with tab3:

        st.subheader("Ask Questions About Your Image")

        question = st.text_input(
            "Enter your question",
            placeholder="What is the main subject?"
        )

        if st.button("Get Answer"):

            if not question.strip():

                st.warning("Please enter a question.")

            else:

                try:

                    with st.spinner("Finding answer..."):

                        answer = ask_image(
                            image,
                            question.strip()
                        )

                    st.success(answer)

                except Exception as error:

                    st.error(str(error))

    # =================================================
    # TAB 4: PRODUCT SHOPPING
    # =================================================

    with tab4:

        st.subheader("🛍️ Smart Product Shopping Assistant")

        st.write(
            "Upload a product image and automatically "
            "find matching products online."
        )

        # ---------------------------------------------
        # AUTOMATIC PRODUCT IDENTIFICATION
        # ---------------------------------------------

        st.markdown("### 1. Identify Product")

        if st.button(
            "Identify Product",
            type="primary"
        ):

            try:

                with st.spinner("Identifying product..."):

                    description = caption_image(image)

                    automatic_query = generate_product_query(
                        description
                    )

                st.session_state["product_description"] = description

                st.session_state["automatic_query"] = automatic_query

            except Exception as error:

                st.error(f"Product identification error: {error}")

        if "product_description" in st.session_state:

            st.markdown("**AI Product Description**")

            st.success(
                st.session_state["product_description"]
            )

            st.markdown("**Automatically Generated Search Query**")

            st.code(
                st.session_state["automatic_query"]
            )

        # ---------------------------------------------
        # SEARCH PRODUCTS
        # ---------------------------------------------

        st.divider()

        st.markdown("### 2. Find Matching Products")

        automatic_query = st.session_state.get(
            "automatic_query",
            ""
        )

        if automatic_query:

            st.write(
                f"VisionChat identified: **{automatic_query}**"
            )

        else:

            st.info(
                "Click Identify Product first to generate "
                "a search query from your image."
            )

        # Optional manual correction.
        with st.expander("Edit search query (optional)"):

            edited_query = st.text_input(
                "Product name or keywords",
                value=automatic_query,
                key="edited_product_query"
            )

        search_query = clean_user_query(
            st.session_state.get(
                "edited_product_query",
                automatic_query
            ),
            automatic_query
        )

        if automatic_query:

            if not api_key:

                st.warning(
                    "Enter your SerpAPI key in the sidebar "
                    "to display live product listings."
                )

            else:

                if st.button(
                    "🛒 Find Matching Products",
                    type="primary"
                ):

                    try:

                        with st.spinner(
                            "Searching Indian shopping listings..."
                        ):

                            raw_products = search_products(
                                search_query,
                                api_key
                            )

                            matched_products = filter_products(
                                raw_products,
                                search_query
                            )

                        st.session_state["matched_products"] = matched_products

                        st.session_state["searched_query"] = search_query

                    except requests.exceptions.Timeout:

                        st.error(
                            "Shopping search timed out. Please try again."
                        )

                    except requests.exceptions.RequestException as error:

                        st.error(
                            f"Shopping connection error: {error}"
                        )

                    except Exception as error:

                        st.error(
                            f"Shopping search error: {error}"
                        )

            # -----------------------------------------
            # DISPLAY SHOPPING RESULTS
            # -----------------------------------------

            if (
                "matched_products" in st.session_state
                and st.session_state.get("searched_query")
                == search_query
            ):

                products = st.session_state["matched_products"]

                st.divider()

                st.subheader("🛍️ Matching Products")

                if products:

                    st.success(
                        f"Found {len(products)} relevant product listings."
                    )

                    for product in products:

                        with st.container(border=True):

                            left, right = st.columns([1, 2])

                            with left:

                                thumbnail = product.get("thumbnail")

                                if thumbnail:

                                    st.image(
                                        thumbnail,
                                        use_container_width=True
                                    )

                                else:

                                    st.write("Product image unavailable")

                            with right:

                                st.markdown(
                                    f"**{product.get('title', 'Product')}**"
                                )

                                st.write(
                                    "**Price:**",
                                    product.get(
                                        "price",
                                        "Price unavailable"
                                    )
                                )

                                st.write(
                                    "**Store:**",
                                    product.get(
                                        "source",
                                        "Store unavailable"
                                    )
                                )

                                if product.get("rating") is not None:

                                    st.write(
                                        "**Rating:**",
                                        product["rating"]
                                    )

                                if product.get("reviews") is not None:

                                    st.write(
                                        "**Reviews:**",
                                        product["reviews"]
                                    )

                                purchase_link = (
                                    product.get("product_link")
                                    or product.get("link")
                                )

                                if purchase_link:

                                    st.link_button(
                                        "🛒 View Product / Buy Now",
                                        purchase_link
                                    )

                else:

                    st.warning(
                        "No sufficiently relevant products were found."
                    )

                    st.write(
                        "Try a different product image or use the "
                        "optional search editor to correct the product name."
                    )

                    st.link_button(
                        "Search Similar Products on Google Shopping",
                        google_shopping_link(search_query)
                    )

        # ---------------------------------------------
        # GOOGLE SHOPPING FALLBACK
        # ---------------------------------------------

        if automatic_query:

            st.divider()

            st.markdown("### Google Shopping")

            st.link_button(
                "🔎 Open Similar Products on Google Shopping",
                google_shopping_link(search_query),
                use_container_width=True
            )

            st.caption(
                "This search link works without a separate shopping API."
            )

else:

    st.markdown("""
    ## Welcome to VisionChat

    ### Features

    - AI image descriptions
    - Object detection with YOLO
    - Ask questions about images
    - Automatic product identification
    - Automatic shopping search queries
    - Relevant product filtering
    - Product images and prices
    - Retailer names and purchase links
    - Indian shopping search

    Upload an image to get started.
    """)


# =====================================================
# 15. TECHNOLOGY INFORMATION
# =====================================================

with st.expander("Models and Technologies Used"):

    st.write("""
    - BLIP: Image captioning
    - BLIP VQA: Image question answering
    - YOLO11 Nano: Object detection
    - PyTorch: Deep learning
    - Transformers: AI model loading
    - Streamlit: Web application
    - SerpAPI: Live shopping listings
    """)

st.divider()

st.caption(
    "VisionChat | AI Image Understanding "
    "and Smart Product Shopping Assistant"
)
