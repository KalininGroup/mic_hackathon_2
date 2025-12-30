from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import json
import base64

def clean_math(math_tag):
    """
    Convert <math> tag into clean text like 'T=0 K'
    """
    # Get main text (ignoring nested <ce:hsp>)
    main_text = "".join([t for t in math_tag.find_all(text=True, recursive=False)]).strip()
    # Get unit text from <rm> if exists
    unit_text = "".join([t.get_text(strip=True) for t in math_tag.find_all("rm")])
    if unit_text:
        return f"{main_text} {unit_text}"
    return main_text

def extract_readable_text(para):
    # Replace all <math> tags with cleaned text
    for math_tag in para.find_all("math"):
        math_tag.replace_with(clean_math(math_tag))
    
    # Remove <ce:float-anchor> tags
    for fa_tag in para.find_all("ce:float-anchor"):
        fa_tag.decompose()
    
    # Replace <ce:cross-ref> with their text
    for cref in para.find_all("ce:cross-ref"):
        cref.replace_with(cref.get_text(strip=True))
    
    # Get final cleaned text
    cleaned_text = para.get_text(" ", strip=True)
    # Collapse multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

tagged_df = pd.read_csv('../data/corpus_acta_mini_tagged.csv')
paths = '../data/' + tagged_df['journal'] + '/' + tagged_df['pii'] + '/' + tagged_df['pii'] + '.xml'


pii_fig_ref_para_dict = {}
for path in paths:
    with open(path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read())
        
    pii = path.split('/')[-1].split('.')[0]
    # Get the directory containing the XML file (where images are stored)
    xml_dir = os.path.dirname(path)

    fig_ref_para_dict = {}
    if pii not in pii_fig_ref_para_dict:
        pii_fig_ref_para_dict[pii] = {}

    # find all figures elements
    figures = soup.find_all('ce:figure')
    paragraphs = soup.find_all('ce:para')
    
    for i in range(len(figures)):
        figure_id = figures[i].get('id')
        
        # Extract caption from figure
        caption_tag = figures[i].find('ce:caption')
        caption_text = ""
        if caption_tag:
            caption_text = extract_readable_text(caption_tag)
        
        # Extract and load image as base64-encoded binary string
        image_base64 = ''
        image_found = False
        link_tag = figures[i].find('ce:link')
        if link_tag and link_tag.get('locator'):
            locator = link_tag.get('locator')
            # Construct image path: {pii}-{locator}.jpg
            image_filename = f"{pii}-{locator}.jpg"
            image_path = os.path.join(xml_dir, image_filename)
            # Read image file as binary and encode as base64
            if os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        image_found = True
                except Exception as e:
                    print(f"Warning: Could not read image {image_path}: {e}")
                    image_base64 = ''
        
        # Only add figure to dictionary if image file exists
        if image_found:
            # Initialize dictionary entry for this figure
            if figure_id not in fig_ref_para_dict:
                fig_ref_para_dict[figure_id] = {
                    'caption': caption_text,
                    'descriptions': [],
                    'image': image_base64
                }
            
            # Find paragraphs that reference this figure
            for para in soup.find_all("ce:para"):
                if para.find("ce:cross-ref", {"refid": figure_id}) or para.find("ce:float-anchor", {"refid": figure_id}):
                    fig_ref_para_dict[figure_id]['descriptions'].append(extract_readable_text(para))
    pii_fig_ref_para_dict[pii] = fig_ref_para_dict


# visualise the dictionary and print only first few characters of all strings
verbose = len(pii_fig_ref_para_dict) < 5
if verbose:
    for pii, fig_ref_para_dict in pii_fig_ref_para_dict.items():
        print(pii)
        for figure_id, data in fig_ref_para_dict.items():
            try:
                print('-'*100)
                print(figure_id)
                print(f"Caption: {data['caption'][:100] if data['caption'] else 'No caption'}")
                print(f"Descriptions: {data['descriptions'][0][:100] if data['descriptions'] else 'No descriptions'}")
                print(f"Image (base64, first 100 chars): {data['image'][:100] if data['image'] else 'No image'}")
                print('-'*100)
            except Exception as e:
                print(f"Error: {e}")
        break

with open('pii_fig_ref_para_dict.jsonl', 'w') as f:
    for pii, fig_ref_para_dict in pii_fig_ref_para_dict.items():
        for figure_id, data in fig_ref_para_dict.items():
            f.write(json.dumps({
                'pii': pii,
                'figure_id': figure_id,
                'caption': data['caption'],
                'descriptions': data['descriptions'],
                'image': data['image']
            }) + '\n')