import requests
from bs4 import BeautifulSoup
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




PARENT_DIR = Path(__file__).resolve().parent


APPLICABLE_OCCUPATIONS ={
    "retail_salesperson": {
                        'source_doc': os.path.join(PARENT_DIR, "data/career_profiles/Retail_Sales_WorkersOOH.html")
                        , 'source' : "https://www.bls.gov/ooh/sales/retail-sales-workers.htm"
                    }



    , "driver_sales_worker": {
                    'source_doc': os.path.join(PARENT_DIR, "data/career_profiles/Delivery_Truck_Drivers_and_Driver_Sales_WorkersOOH.html") 
                    , 'source' :"https://www.bls.gov/ooh/transportation-and-material-moving/delivery-truck-drivers-and-driver-sales-workers.htm"    
                }

    , "computer_id_scientist":{
            "source_doc": os.path.join(PARENT_DIR, "data/career_profiles/Computer_and_Information_Research_Scientists_OOH.html")
            , "source": "https://www.bls.gov/ooh/computer-and-information-technology/computer-and-information-research-scientists.htm"

    }

    , "physican_surgeon":{
            "source_doc": os.path.join(PARENT_DIR, "data/career_profiles/Physicians_and_Surgeons_OOH.html")
            , "source": "https://www.bls.gov/ooh/healthcare/physicians-and-surgeons.htm"

    }  

    , "data_scientist":{
            "source_doc": os.path.join(PARENT_DIR, "data/career_profiles/Data_Scientists_OOH.html")
            , "source": "https://www.bls.gov/ooh/math/data-scientists.htm"

    } 
 

}




class DataProcessor:

    @staticmethod
    def extract_content(source, from_url=True, metadata=None):
        """
        Extracts text from <p>, <h3>, <h5> and <table> tags from a URL or local HTML file.

        Args:
            source (str): URL or file path to the HTML content.
            from_url (bool): True if source is a URL; False if it's a local file.

        Returns:
            str: Combined readable content from paragraphs, headers, and tables.

        """

        def is_inside_nav(tag):
            return any(parent.name == "nav" for parent in tag.parents)

        # if True:
        try:
            # Load HTML
            if from_url:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/114.0.0.0 Safari/537.36"
                    )
                }
                response = requests.get(source, headers=headers)
                response.raise_for_status()
                html = response.content
            else:
                if not os.path.exists(source):
                    return f"Error: File not found: {source}"
                with open(source, "r", encoding="utf-8") as f:
                    html = f.read()

            soup = BeautifulSoup(html, "html.parser")
            output = []

            # Get paragraphs and headers
            for tag in soup.find_all(["h1", "h2","h3", "h5", "h6", "p", "br", "li"]): #, "a"
                if tag.name == "p" and "visually-hidden" in tag.get("class", []):
                    continue
                elif tag.name == "li" and not "" in tag.get("class", []):
                    continue
                
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    output.append(text)

            # Get tables
            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    output.append("\n".join(rows))

            # # Extract from <div> with class "order-2 flex-grow-1"
            for div in soup.find_all("div", class_="order-2 flex-grow-1"):

                if not ("visually-hidden" in div.get("class", []) or "dropdown-menu" in div.get("class", [])):
                    if not is_inside_nav(div):
                        text = div.get_text(separator=" ", strip=True)
                        if text:
                            output.append(text)

            for div in soup.find_all("div", class_="reportsection"):
                if not is_inside_nav(div):
                    text = div.get_text(separator=" ", strip=True)
                    if text:
                        output.append(text)

            if metadata is None:
                if from_url:
                    metadata = {'source': source, 'source_doc':None}
                else:
                    metadata = {'source': None, 'source_doc': source}
        
            return Document(page_content="\n\n".join(output[:]), metadata=metadata)

        except requests.exceptions.RequestException as e:
            return f"Error fetching content from URL: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"


    @staticmethod
    def parse_pdf(pdf_pathname, split=True, **kwargs ):
        
        loader = PyPDFLoader(pdf_pathname)  # Replace with your PDF file path
        documents = loader.load()

        info_category=kwargs.get('info_category', None)

        if not info_category is None:
            for d in documents:
                d.metadata['info_category'] = info_category

        if split:
            text_splitter = kwargs.get('text_splitter', None)
            if text_splitter is None:
                text_splitter = kwargs.get('text_splitter', 
                                                RecursiveCharacterTextSplitter(
                                                    chunk_size=500,  # Maximum characters per chunk
                                                    chunk_overlap=100,  # Overlap to maintain context between chunks
                                                    length_function=len
                                                )
                                            )  

            chunks = text_splitter.split_documents(documents)



            return chunks
        else:
            return documents


    @staticmethod
    def collect_and_process_documents(occupations:[str, list]=None, **kwargs):

        if occupations is None:
            occupations = [occ for occ in APPLICABLE_OCCUPATIONS]

        elif isinstance(occupations, str):
            occupations=[occupations]


        text_splitter = kwargs.get('text_splitter', None)

        if text_splitter is None:
            chunk_size = kwargs.get('chunk_size', 500)
            chunk_overlap = kwargs.get('chunk_overlap', 100)
            length_function=kwargs.get('chunk_overlap', len)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,  # Maximum characters per chunk
                chunk_overlap=chunk_overlap,  # Overlap to maintain context between chunks
                length_function=length_function
            )


        occ_metadata_ = {}
        occ_metadata_['career_profile']  = None
        occ_metadata_['state_standards'] = None
        # occ_metadata_['iep_goals_and_transition_templates'] = None
        occ_metadata_['idea'] = None

        career_docs = []

        for occupation in occupations:
            assert occupation in APPLICABLE_OCCUPATIONS, f"The occupation you provided is not supported. It must be one of the following: {list(APPLICABLE_OCCUPATIONS.keys())}"

            

            ## Retrieve Job occupation data from BLS
            jobpro =  DataProcessor.extract_content(source=APPLICABLE_OCCUPATIONS[occupation]['source_doc'], 
                metadata = {
                            'source_doc': APPLICABLE_OCCUPATIONS[occupation]['source_doc']
                            ,  'source': APPLICABLE_OCCUPATIONS[occupation]['source']
                            , 'info_category': 'career_profile'

                        }
                    , from_url=False)

            # print("jobpro: ", jobpro)
            career_docs.append(jobpro)
        # print("Career Docs: ", len(career_docs) )
        # print(career_docs[-1])
        occ_metadata_['career_profile'] = text_splitter.split_documents(career_docs)
        # print(len(occ_metadata_['career_profile']))


        ## Retrieve State educational standards for employment skills
        occ_metadata_['state_standards'] = DataProcessor.parse_pdf( os.path.join(PARENT_DIR, "data/State_Educational_Standards_IOWA _k-12.pdf") , info_category = 'state_standards')
        # for d in occ_metadata_['state_standards']:
        #     d['metatada']['info_category'] = 'state_standards'


        ## Retrieve a sample for IEP goals and transition planning        
        # occ_metadata_['iep_goals_and_transition_templates'] = DataProcessor.parse_pdf( os.path.join(PARENT_DIR, "data/Sample_IEP_Transition_Plan_Understood.pdf") , info_category='iep_goals_and_transition_templates')

        ## Retrieve Sec. 300.320 (b) from the Indiviual with Diabilities Education Act
        idea_ = DataProcessor.extract_content(source="https://sites.ed.gov/idea/regs/b/d/300.320/b", from_url=True)
        # print(f"idea_ {idea_.__class__}", idea_)

        if isinstance(idea_, Document):
            idea_.metadata['info_category'] = 'idea'
            occ_metadata_['idea'] = [idea_]
        elif isinstance(idea_, list):
            for k in idea_:
                k.metadata['info_category'] = 'idea'
                occ_metadata_['idea'] = idea_


        # occ_metadata_['idea'].extend([idea_])

        return occ_metadata_


        




