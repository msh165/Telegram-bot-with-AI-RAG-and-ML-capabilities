import pickle
import pandas as pd
def kyphosis_predict(age:int, number:int,start:int)->int:
    """This tool would be used to predict whether kyphosis is absent or present in a patient.
    It takes Age, Number and start values from user
    age: age of the child in MONTHS.
    number: number of vertebrae involved in the operation.
    start: number of the first vertebra operated on."""
    model = pickle.load(open("model.pkl", "rb"))
    preprocessing = pickle.load(open("preprocessing.pkl", "rb"))

    patient = pd.DataFrame({"Age": [15], "Number": [12], "Start": [7]})
    patient[['Age']] = preprocessing["age_scaler"].transform(patient[['Age']])
    prediction = model.predict(patient)
    
    return prediction


from rag import search 

def search_private_documents(query: str) -> str:
    """Use this tool to search through the user's private documents, PDFs, text files, and images.
    Always use this tool if the user asks a question about their files, documents, or data.
    Input should be a clear search query."""
    
    try:
        #RETRIEVAL
        # Call the search function from rag.py
        results = search(query, k=4)
        
        if not results:
            return "No relevant information found in the documents."
        
        #AUGMENTATION
        # Format the retrieved chunks into a single string for the LLM to read
        formatted_results = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown file")
            content = doc.page_content
            formatted_results.append(f"--- Document {i+1} (Source: {source}) ---\n{content}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"An error occurred while searching the documents: {str(e)}"
search_private_documents('what website would be used for EMEA market in place of DV360 Reference from Nestle doc')