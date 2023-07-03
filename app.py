import streamlit as st
import requests
import json

def main():
    st.title("API Frontend - POST-GET Debugger")
    url_API =st.text_input("inserisci url dell'api","http://localhost:8000/predict")
    rd_spend= st.number_input("Inserisci R&D Spend")
    administration = st.number_input("Inserisci Administration")
    marketing = st.number_input("Inserisci Marketing")

    ############## GET REQUEST #################
    if st.button("Predict with GET"):
        url = url_API
        url2 = f"?R&D Spend={rd_spend}&Administration={administration}&Marketing={marketing}"
        link = url+url2
        st.write('"{}"'.format(link))
        response = requests.get(link)
        result =response.json()
        st.success(f"The result is: {result['prediction']}")

    ############## POST REQUEST #################
    if st.button("Predict with POST"):
        url = url_API
        response =requests.post(url,
                                headers={"Content-Type": "application/json"},
                                data = json.dumps({
                                                   "R&D Spend":rd_spend,
                                                   "Administration":administration,
                                                   "Marketing":marketing,
                                                   })
                                )
        result =response.json()
        st.success(f"The result is: {result['prediction']}")

if __name__ == '__main__':
    main()