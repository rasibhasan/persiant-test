import streamlit as st
import os
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from pinecone_text.sparse import BM25Encoder
import json
import nltk

# Set a persistent directory for NLTK data
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

# Check and download necessary NLTK components
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)

# # Get the API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def debug_print_context(inputs):
    """Debug function to print context details."""
    con = inputs.get("context", [])
    context = []
    for doc in con:
        context.append(doc.metadata)

    return inputs

def enhance_query(query, model_name):
    """Enhance the query using LLM before retrieval."""
    # Create a prompt for query enhancement
    query_enhancement_prompt = """
    شما یک متخصص در بهبود جستجو هستید. لطفاً پرسش کاربر را غنی‌سازی کنید تا برای جستجوی مبتنی بر بازیابی بهینه شود.
    هدف شما اضافه کردن کلمات کلیدی مرتبط و مفاهیم مرتبط با پرسش اصلی است.
    
    پرسش اولیه:
    {query}
    
    پرسش غنی‌سازی شده (با کلمات کلیدی و مفاهیم مرتبط اضافی):
    """
    
    # Create LLM instance for query enhancement
    enhancement_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3, api_key=OPENAI_API_KEY)
    
    # Create a simple prompt template
    enhancement_prompt = ChatPromptTemplate.from_template(query_enhancement_prompt)
    
    # Create a simple chain for query enhancement
    enhancement_chain = (
        enhancement_prompt 
        | enhancement_llm 
        | StrOutputParser()
    )
    
    # Invoke the chain with the query
    enhanced_query = enhancement_chain.invoke({"query": query})
    
    return enhanced_query

def create_chatbot_retrieval_qa(main_query, additional_note, vs, categories, sub_categories, model_name, use_query_enhancement):
    """Modified to handle query enhancement toggle and model selection."""
    prompt_template = """
    شما یک دستیار هوشمند و مفید هستید. با استفاده از متن زیر به پرسش مطرح‌شده با دقت، شفافیت، و به صورت کامل پاسخ دهید:
    1. پاسخ را **به زبان فارسی** ارائه دهید.
    2. **جزئیات کامل** را پوشش دهید و اطمینان حاصل کنید که تمام جنبه‌های سؤال به دقت بررسی شده‌اند.
    3. تاریخ‌ها و اطلاعات ارائه‌شده باید **مطابق با متن** باشند. از درج تاریخ‌های نادرست خودداری کنید.
    4. در صورت نیاز به ارجاع به تاریخ، از **نام فایل برای تاریخ دقیق** استفاده کنید.
    5. نام فایل را در مرجع پاسخ بدهید

    **متن:**
    {context}

    **سؤال اصلی:**
    {main_question}

    **یادداشت اضافی:**
    {additional_note}s
    """
    after_rag_prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    def filtered_retriever(query):
        filter_dict = {}
        if categories != ['ALL'] and categories != []:
            if categories:
                filter_dict["category"] = {"$in": categories}
        if sub_categories != ['ALL'] and sub_categories != []:
            if sub_categories:
                filter_dict["year"] = {"$in": sub_categories}
        
        # Apply query enhancement only if enabled
        if use_query_enhancement:
            enhanced_query = enhance_query(query, model_name)
            retrieval_query = enhanced_query
        else:
            retrieval_query = query
        
        return vs.get_relevant_documents(
            retrieval_query,
            filter=filter_dict
        )

    chain = (
        {
            "context": lambda x: filtered_retriever(x["main_question"]),
            "main_question": lambda x: x["main_question"],
            "additional_note": lambda x: x["additional_note"]
        }
        | RunnablePassthrough(lambda inputs: debug_print_context(inputs))
        | after_rag_prompt
        | llm
        | StrOutputParser()
    )

    return chain

def initialize_chatbot(alpha=0.3, top_k=60):
    """Initialize the chatbot with Pinecone index and embeddings."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "persian-test"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    index = pc.Index(INDEX_NAME)

    bm25_encoder = BM25Encoder().load("full_bm25_values.json")

    vectorstore = PineconeHybridSearchRetriever(
        alpha=alpha, 
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=top_k
    )

    return vectorstore

# Streamlit page configuration
st.set_page_config(
    page_title="Persian Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS with added loading animation styles
st.markdown("""
    <style>
        body { direction: rtl; text-align: right;}
        h1, h2, h3, h4, h5, h6 { text-align: right; }
        .st-emotion-cache-12fmjuu { display: none;}
        p { font-size:25px !important; }
        .loading-message {
            text-align: center;
            font-size: 20px;
            margin: 20px;
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
        }
        .stTextInput input, .stTextArea textarea {
            font-size: 25px !important;
        }
        .st-af {
            font-size: 1.1rem !important;
        }
        .search-params {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        /* Fix for RTL slider issues */
        .stSlider [data-baseweb="slider"] {
            direction: ltr;
        }
        .stSlider [data-testid="stMarkdownContainer"] {
            text-align: right;
            direction: rtl;
        }
        /* Custom styling for toggle container */
        .custom-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 15px;
        }
        /* Ensure checkbox label is properly aligned in RTL */
        .stCheckbox [data-testid="stMarkdownContainer"] {
            text-align: right !important;
            direction: rtl !important;
        }
        /* For better alignment in radio buttons */
        .stRadio [data-testid="stMarkdownContainer"] {
            text-align: right !important;
            direction: rtl !important;
        }
        /* Bold text for toggle label */
        .toggle-label {
            font-weight: bold;
            font-size: 18px;
        }
        .toggle-description {
            margin-top: 5px;
            color: #4a4a4a;
        }
        .custom-container:empty {
            display: none;
        }

    </style>
""", unsafe_allow_html=True)

def get_selected_subfolders(selected_folders):
    with open("folder_structure.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    if selected_folders==[]:
        return ['ALL']
    folder_dict = data[0]
    subfolder_list = ['ALL']
    for folder in selected_folders:
        if folder in folder_dict:
            subfolder_list.extend(folder_dict[folder])
    return subfolder_list

def main():
    st.markdown("<h1 class='persian-text'>چت‌بات فارسی</h1>", unsafe_allow_html=True)

    # Initialize session state for loading and search parameters
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'alpha' not in st.session_state:
        st.session_state.alpha = 0.3
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 60
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "gpt-4o-mini"
    if 'use_query_enhancement' not in st.session_state:
        st.session_state.use_query_enhancement = True

    # Predefined categories
    with open("folder_structure.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Extract main folder names
    cat = list(data[0].keys()) 

    # Search Parameters Section (collapsible)
    with st.expander("تنظیمات جستجو (پیشرفته)", expanded=False):
        st.markdown("<div class='search-params'>", unsafe_allow_html=True)
        
        # Define callback for alpha slider
        def on_alpha_change():
            st.session_state.vectorstore = None
            
        # Define callback for top_k slider
        def on_top_k_change():
            st.session_state.vectorstore = None
        
        # Use two columns with class for better RTL support
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stSlider">', unsafe_allow_html=True)
            st.session_state.alpha = st.slider(
                "نسبت جستجوی هیبریدی (alpha):",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.alpha,
                step=0.1,
                help="مقدار بالاتر به معنای وزن بیشتر برای جستجوی معنایی است. مقدار کمتر وزن بیشتری به جستجوی کلیدواژه می‌دهد.",
                key="alpha_slider",
                on_change=on_alpha_change
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stSlider">', unsafe_allow_html=True)
            st.session_state.top_k = st.slider(
                "تعداد نتایج (top_k):",
                min_value=10,
                max_value=200,
                value=st.session_state.top_k,
                step=10,
                help="تعداد نتایج مرتبطی که از پایگاه داده بازیابی می‌شود.",
                key="top_k_slider",
                on_change=on_top_k_change
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # Model selection
    model_options = {
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4o": "GPT-4o",
        "o3-mini": "O3 Mini",
        "o1": "O1",

    }
    
    col1, col2 = st.columns([2, 1])  # Adjusted column width ratio
    
    with col1:
        selected_model = st.selectbox(
            "انتخاب مدل هوش مصنوعی:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.model_name)
        )
        # Update session state with selected model
        st.session_state.model_name = selected_model
    
    # Improved Query Enhancement Toggle with Radio Buttons
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    
    st.markdown('<p class="toggle-label">بهبود پرسش‌ها با هوش مصنوعی:</p>', unsafe_allow_html=True)
    
    enhancement_options = {
        True: "فعال - استفاده از هوش مصنوعی برای غنی‌سازی پرسش‌ها",
        False: "غیرفعال - استفاده از پرسش‌های اصلی بدون تغییر"
    }
    
    selected_option = st.radio(
        "",  # Empty label since we use custom HTML label above
        options=list(enhancement_options.keys()),
        format_func=lambda x: enhancement_options[x],
        index=1 if not st.session_state.use_query_enhancement else 0,
        horizontal=True,
        key="query_enhancement_radio"
    )
    
    # Update session state with selected enhancement option
    st.session_state.use_query_enhancement = selected_option
    
    # Description based on selected option
    if st.session_state.use_query_enhancement:
        st.markdown("""
            <p class="toggle-description">
            با فعال بودن این گزینه، هوش مصنوعی پرسش‌های شما را با افزودن کلمات کلیدی و مفاهیم مرتبط غنی‌سازی می‌کند 
            تا نتایج جستجوی دقیق‌تری ارائه دهد.
            </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <p class="toggle-description">
            با غیرفعال بودن این گزینه، پرسش‌های شما بدون هیچ تغییری برای جستجو استفاده می‌شوند.
            </p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize chatbot if needed
    if st.session_state.vectorstore is None:
        with st.spinner('در حال راه‌اندازی چت‌بات...'):
            try:
                st.session_state.vectorstore = initialize_chatbot(
                    alpha=st.session_state.alpha,
                    top_k=st.session_state.top_k
                )
                st.success(f"پارامترهای جستجو: alpha={st.session_state.alpha}, top_k={st.session_state.top_k}")
            except Exception as e:
                st.error(f"خطا در راه‌اندازی chatbot: {e}")
                return
            
    # Category selections
    categories = st.multiselect(
        "دسته‌بندی را انتخاب کنید:",
        cat,
        default=[]
    )
    
    sub_cat = get_selected_subfolders(categories)

    sub_categories = st.multiselect(
        "زیر دسته‌بندی را انتخاب کنید:",
        sub_cat,
        default=[]
    )

    # Two separate input fields
    main_query = st.text_area(
        "سؤال اصلی خود را اینجا وارد کنید:",
        height=100
    )

    additional_note = st.text_area(
        "یادداشت اضافی (اختیاری):",
        height=100
    )

    # Display current model and settings
    enhancement_status = "فعال" if st.session_state.use_query_enhancement else "غیرفعال"
    settings_info = f"مدل انتخاب شده: {model_options[st.session_state.model_name]} | بهبود پرسش با هوش مصنوعی: {enhancement_status}"
    st.info(settings_info)

    # Submit button
    if st.button("ارسال"):
        response_placeholder = st.empty()

        if not main_query:
            st.warning("لطفاً سؤال اصلی خود را وارد کنید.")
            return

        if not categories and not sub_categories:
            st.warning("لطفاً حداقل یک دسته‌بندی یا زیر دسته‌بندی را انتخاب کنید.")
            return
        
        response_placeholder = st.empty()

        try:
            # Show loading spinner
            with st.spinner('لطفاً صبر کنید...'):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Update progress for vector search
                status_text.text("در حال جستجوی اطلاعات مرتبط...")
                progress_bar.progress(33)
                
                # Create chatbot
                chatbot = create_chatbot_retrieval_qa(
                    main_query,
                    additional_note,
                    st.session_state.vectorstore,
                    categories,
                    sub_categories,
                    st.session_state.model_name,
                    st.session_state.use_query_enhancement  # Pass the enhancement toggle state
                )
                
                # Update progress for processing
                status_text.text("در حال پردازش اطلاعات...")
                progress_bar.progress(66)
                
                # Get response
                response = chatbot.invoke({
                    "main_question": main_query,
                    "additional_note": additional_note if additional_note else ""
                })
                
                # Update progress for completion
                status_text.text("در حال آماده‌سازی پاسخ...")
                progress_bar.progress(100)
                
                # Clear progress indicators
                time.sleep(0.5)  # Short delay for smooth transition
                progress_bar.empty()
                status_text.empty()

                # Display response
                response_placeholder.markdown("**پاسخ:**")
                response_placeholder.write(response)

        except Exception as e:
            st.error(f"خطا در پردازش سوال: {e}")
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()