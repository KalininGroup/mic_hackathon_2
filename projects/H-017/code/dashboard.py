import streamlit as st
import json
import base64
import pandas as pd
import os
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="MARVL Figure Viewer",
    page_icon="🔬",
    layout="wide"
)

# Load data
@st.cache_data
def load_jsonl_data(jsonl_path):
    """Load data from JSONL file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

@st.cache_data
def load_journal_mapping(csv_path):
    """Load journal to PII mapping from CSV"""
    df = pd.read_csv(csv_path)
    # Get unique journal-PII pairs
    journal_pii = df[['journal', 'pii']].drop_duplicates()
    return journal_pii

def decode_base64_image(image_base64):
    """Decode base64 image string to PIL Image"""
    if not image_base64:
        return None
    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        st.error(f"Error decoding image: {e}")
        return None

# Load data - handle paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
jsonl_path = os.path.join(script_dir, 'pii_fig_ref_para_dict.jsonl')
csv_path = os.path.join(script_dir, '../data/corpus_acta_mini_tagged.csv')
csv_path = os.path.normpath(csv_path)

try:
    data = load_jsonl_data(jsonl_path)
    journal_pii_df = load_journal_mapping(csv_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# Create a mapping from PII to journal
pii_to_journal = dict(zip(journal_pii_df['pii'], journal_pii_df['journal']))

# Add journal to each data entry
for entry in data:
    entry['journal'] = pii_to_journal.get(entry['pii'], 'Unknown')

# Title
st.title("🔬 MARVL Figure Viewer")
st.markdown("Visualize figures from scientific papers with captions and descriptions")

# Check if we should switch to browse tab from search
if 'switch_to_browse' in st.session_state and st.session_state['switch_to_browse']:
    default_view = "📋 Browse"
    del st.session_state['switch_to_browse']
else:
    default_view = "🔍 Search"

# View selector - styled as tabs
view_mode = st.radio(
    "View Mode",
    ["🔍 Search", "📋 Browse"],
    index=0 if default_view == "🔍 Search" else 1,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# Search functionality
def search_figures(data, search_query, search_in_caption=True, search_in_descriptions=True):
    """Search figures by keyword in captions and/or descriptions"""
    if not search_query:
        return []
    
    search_query_lower = search_query.lower()
    results = []
    
    for entry in data:
        matches = False
        match_text = ""
        
        # Search in caption
        if search_in_caption and entry.get('caption'):
            caption_lower = entry['caption'].lower()
            if search_query_lower in caption_lower:
                matches = True
                # Find context around the match
                idx = caption_lower.find(search_query_lower)
                start = max(0, idx - 50)
                end = min(len(entry['caption']), idx + len(search_query) + 50)
                match_text = f"Caption: ...{entry['caption'][start:end]}..."
        
        # Search in descriptions
        if search_in_descriptions and entry.get('descriptions'):
            for i, desc in enumerate(entry['descriptions']):
                desc_lower = desc.lower()
                if search_query_lower in desc_lower:
                    matches = True
                    # Find context around the match
                    idx = desc_lower.find(search_query_lower)
                    start = max(0, idx - 50)
                    end = min(len(desc), idx + len(search_query) + 50)
                    match_text = f"Description {i+1}: ...{desc[start:end]}..."
                    break
        
        if matches:
            results.append({
                'entry': entry,
                'match_text': match_text
            })
    
    return results

if view_mode == "🔍 Search":
    st.header("Search Figures by Keywords")
    
    # Search input
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_query = st.text_input(
            "Enter keywords to search",
            placeholder="e.g., TEM, microstructure, dislocation",
            key="search_input"
        )
    
    with col_search2:
        st.write("")  # Spacing
        search_button = st.button("Search", type="primary")
    
    # Search options
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        search_in_caption = st.checkbox("Search in captions", value=True)
    with col_opt2:
        search_in_descriptions = st.checkbox("Search in descriptions", value=True)
    
    # Perform search
    if search_query or search_button:
        if search_query:
            search_results = search_figures(data, search_query, search_in_caption, search_in_descriptions)
            
            if search_results:
                st.success(f"Found {len(search_results)} matching figure(s)")
                
                # Display results
                for idx, result in enumerate(search_results):
                    entry = result['entry']
                    
                    with st.expander(f"**{entry['figure_id']}** - {entry['pii']} ({entry['journal']})", expanded=False):
                        # Create columns for image and info
                        col_img, col_info = st.columns([1, 2])
                        
                        with col_img:
                            if entry['image']:
                                image = decode_base64_image(entry['image'])
                                if image:
                                    st.image(image, caption=entry['figure_id'])
                        
                        with col_info:
                            st.markdown(f"**Figure ID:** {entry['figure_id']}")
                            st.markdown(f"**Paper ID:** {entry['pii']}")
                            st.markdown(f"**Journal:** {entry['journal']}")
                            
                            if entry['caption']:
                                st.markdown(f"**Caption:** {entry['caption'][:200]}...")
                            
                            st.markdown(f"**Match:** {result['match_text']}")
                            
                            # Button to view full details
                            if st.button(f"View Full Details", key=f"view_{idx}"):
                                st.session_state['selected_pii'] = entry['pii']
                                st.session_state['selected_figure'] = entry['figure_id']
                                st.session_state['selected_journal'] = entry['journal']
                                st.session_state['switch_to_browse'] = True
                                st.rerun()
                
                # Summary statistics
                st.markdown("---")
                st.markdown("### Search Summary")
                result_piis = set(r['entry']['pii'] for r in search_results)
                result_journals = set(r['entry']['journal'] for r in search_results)
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Matching Figures", len(search_results))
                with col_sum2:
                    st.metric("Papers", len(result_piis))
                with col_sum3:
                    st.metric("Journals", len(result_journals))
            else:
                st.warning("No figures found matching your search query. Try different keywords.")
        else:
            st.info("Enter keywords in the search box above to find figures.")
    else:
        st.info("👆 Enter keywords to search through figure captions and descriptions")

elif view_mode == "📋 Browse":
    st.header("Browse Figures")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Get unique journals
    journals = sorted([j for j in journal_pii_df['journal'].unique() if pd.notna(j)])
    selected_journal = st.sidebar.selectbox(
        "Select Journal",
        options=journals,
        index=0 if journals else None
    )
    
    # Filter PIIs by selected journal
    if selected_journal:
        available_piis = sorted(journal_pii_df[journal_pii_df['journal'] == selected_journal]['pii'].unique())
        selected_pii = st.sidebar.selectbox(
            "Select Paper (PII)",
            options=available_piis,
            index=0 if available_piis else None
        )
    else:
        selected_pii = None
    
    # Filter figures by selected PII
    if selected_pii:
        available_figures = sorted([d['figure_id'] for d in data if d['pii'] == selected_pii])
        selected_figure = st.sidebar.selectbox(
            "Select Figure",
            options=available_figures,
            index=0 if available_figures else None
        )
    else:
        selected_figure = None
    
    # Check if we should navigate from search results
    if 'selected_pii' in st.session_state and 'selected_figure' in st.session_state:
        selected_pii = st.session_state['selected_pii']
        selected_figure = st.session_state['selected_figure']
        selected_journal = st.session_state.get('selected_journal', None)
        # Clear session state after using
        del st.session_state['selected_pii']
        del st.session_state['selected_figure']
        if 'selected_journal' in st.session_state:
            del st.session_state['selected_journal']
    
    # Display selected data
    if selected_figure:
        # Find the selected entry
        selected_entry = next((d for d in data if d['pii'] == selected_pii and d['figure_id'] == selected_figure), None)
        
        if selected_entry:
            # Display metadata
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.markdown(f"**Figure ID:** {selected_entry['figure_id']}")
            with col_meta2:
                st.markdown(f"**Paper ID (PII):** {selected_entry['pii']}")
            with col_meta3:
                st.markdown(f"**Journal:** {selected_entry['journal']}")
            
            st.markdown("---")
            
            # Display image with caption
            if selected_entry['image']:
                image = decode_base64_image(selected_entry['image'])
                if image:
                    # Display image centered
                    st.subheader(f"Figure: {selected_entry['figure_id']}")
                    st.image(image, caption=selected_entry['caption'] if selected_entry['caption'] else selected_entry['figure_id'])
                else:
                    st.warning("Could not decode image")
            else:
                st.warning("No image available for this figure")
            
            st.markdown("---")
            
            # Display caption
            if selected_entry['caption']:
                st.subheader("Caption")
                st.markdown(selected_entry['caption'])
                st.markdown("---")
            
            # Display descriptions
            if selected_entry['descriptions']:
                st.subheader("Descriptions from Paper")
                for i, desc in enumerate(selected_entry['descriptions'], 1):
                    st.markdown(f"**Description {i}:**")
                    st.markdown(desc)
                    if i < len(selected_entry['descriptions']):
                        st.markdown("---")
            else:
                st.info("No descriptions available for this figure")
            
            # Statistics
            st.sidebar.markdown("---")
            st.sidebar.markdown("### Statistics")
            st.sidebar.metric("Total Papers", len(available_piis))
            st.sidebar.metric("Total Figures", len(available_figures))
            st.sidebar.metric("Total Descriptions", len(selected_entry['descriptions']))
            
        else:
            st.warning("Selected figure not found in data")
    else:
        # Show overview when nothing is selected
        st.info("👈 Please select a journal, paper, and figure from the sidebar to view details")
        
        # Show summary statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        total_papers = len(journal_pii_df['pii'].unique())
        total_figures = len(data)
        total_journals = len(journals)
        
        with col1:
            st.metric("Total Journals", total_journals)
        with col2:
            st.metric("Total Papers", total_papers)
        with col3:
            st.metric("Total Figures", total_figures)
        
        # Show journal breakdown
        if journals:
            st.subheader("Figures by Journal")
            journal_counts = {}
            for entry in data:
                journal = entry.get('journal', 'Unknown')
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
            
            journal_df = pd.DataFrame(list(journal_counts.items()), columns=['Journal', 'Number of Figures'])
            journal_df = journal_df.sort_values('Number of Figures', ascending=False)
            st.dataframe(journal_df, use_container_width=True)

