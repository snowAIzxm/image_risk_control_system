from search_service.mivlus_repository import MilvusRepository
from search_service.mongo_repository import MongoRepository


class EmbeddingGenerator:
    def __init__(self, host, port, collection_name, dim, index_metric, embed_model=None):
        self.milvus_repository = MilvusRepository(
            host=host,
            port=port,
            collection_name=collection_name,
            dim=dim,
            index_metric=index_metric
        )
        self.mongo_repository = MongoRepository()
        if embed_model is None:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

    def process_image_list_to_milvus(self, image_list):
        for image in image_list:
            image_feature = self.get_image_feature(image)
            self.milvus_repository.save_image_list(image_feature)
