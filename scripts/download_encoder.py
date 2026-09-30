from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    SentenceTransformer('keepitreal/vietnamese-sbert')
    print('Encoder cached. Start the FastAPI backend next.')
