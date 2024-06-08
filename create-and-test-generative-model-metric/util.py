from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
from metric_dataset import MetricDataset 
import distance
import embedding

def compute_embeddings_for_ref(img_dir, embedding_model, generate, sample_size, effect, power, batch_size):
    dataset = MetricDataset(img_dir, embedding_model.input_image_size,
                            generate, sample_size, effect, power)
    count = len(dataset)
    print(f"Calculating embeddings for {count} images from {img_dir}.")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()
        image_batch = image_batch / 255.0
        cur_embs = np.asarray(embedding_model.get_embedding(image_batch))  
        embeddings.append(cur_embs)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def compute_metric(encoder, class_type, formula, sample_size, effect, power, ref_dir, eval_dir, batch_size):
    
    if encoder == 'InceptionV3':
        embedding_model = embedding.InceptionEmbedding()
    elif encoder == 'CLIP':
        embedding_model = embedding.ClipEmbedding()
    else:
        embedding_model = embedding.DinoEmbedding()
    
    class_bool = True if class_type == 'Yes' else False
    
    ref_embs = compute_embeddings_for_ref(ref_dir, embedding_model, False, sample_size, effect, power, batch_size).astype("float32")
    eval_embs = compute_embeddings_for_ref(eval_dir, embedding_model, True, sample_size, effect, power,  batch_size).astype("float32")
    
    if formula == 'FD':
        val = distance.fd(ref_embs, eval_embs)
    else:
        val = distance.mmd(ref_embs, eval_embs)
        
    return val.numpy()