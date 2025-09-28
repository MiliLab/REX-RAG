TOPK=3

# ! Change it
INDEX_PATH=/xx/wiki/e5_Flat.index
CORPUS_PATH=/xx/wiki/wiki-18.jsonl

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python search_r1/search/retrieval_server.py \
                    --topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --retriever_name e5 \
                    --retriever_model intfloat/e5-base-v2 \
                    --faiss_gpu
