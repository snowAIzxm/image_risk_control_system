from dataclasses import dataclass
from typing import List
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


@dataclass
class MilvusImage:
    id: str
    url: str
    feature: List[float] = None
    distance: float = None


class MilvusRepository:
    def __init__(self, collection_name,
                 dim: int,
                 host: str,
                 port: int = 19530,  # milvus default port
                 index_metric='COSINE',

                 ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.index_metric = index_metric
        self.dim = dim
        self._connect()
        if not utility.has_collection(self.collection_name):
            self._create()
        self.collection = Collection(name=self.collection_name)

    def _connect(self):
        connections.connect(host=self.host, port=self.port)

    def _create(self, is_drop=False):  # if need new data, set is_drop=True
        collection_name = self.collection_name
        dim = self.dim
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, descrition='ids', is_primary=True, auto_id=False,
                        max_length=100),
            FieldSchema(name="url", dtype=DataType.VARCHAR, descrition="url", max_length=100),
            FieldSchema(name='feature', dtype=DataType.FLOAT_VECTOR, descrition='feature vectors', dim=dim)
        ]

        self._connect()
        if is_drop and utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        schema = CollectionSchema(fields=fields, description=collection_name)
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type': self.index_metric,
            'index_type': "IVF_FLAT",
            'params': {"nlist": 512}
        }
        collection.create_index(field_name="feature", index_params=index_params)
        # 为 'clue_id' 字段创建索引

    def save_image_list(self, image_feature_list: List[MilvusImage]):
        collection = self.collection
        id_list = []
        url_list = []
        feature_list = []

        for image_feature in image_feature_list:
            id_list.append(image_feature.id)
            url_list.append(image_feature.url)
            feature_list.append(image_feature.feature)

        data = [id_list, url_list, feature_list]
        mr = collection.insert(data)
        return mr

    def start_load(self):
        self.collection.load()
        print('Total number of inserted data is {}.'.format(self.collection.num_entities))

    def search(self, embedding, metric_type=None, offset=0, limit=50) -> List[MilvusImage]:
        if metric_type is None:
            metric_type = self.index_metric
        # metric_type COSINE,L2,IP
        search_params = {
            "metric_type": metric_type,
            "offset": offset,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }
        search_results = self.collection.search(
            data=[embedding],
            anns_field="feature",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=limit,
            expr=None,
            # set the names of the fields you want to
            # retrieve from the search result.
            output_fields=['id', "url"],
        )
        result = []
        for hit in search_results[0]:
            result.append(MilvusImage(
                id=hit.id,
                url=hit.entity.get("url"),
                distance=hit.distance,
            ))

        return result

    def delete_by_id_list(self, id_list):
        expr = f"id in {id_list}"
        self.collection.delete(expr)

    def query_by_id_list(self, id_list):
        expr = f"id in {[str(x) for x in id_list]}"
        # 执行查询
        results = self.collection.query(expr, output_fields=["id", "url", "feature"])
        # 将查询结果转换为 CleanImage 对象列表
        clean_images = [MilvusImage(id=result["id"], url=result["url"], feature=result["feature"]) for result in
                        results]
        return clean_images

    def query_by_id(self, id_):
        expr = f"id in {[str(id_)]}"
        # 执行查询
        results = self.collection.query(expr, output_fields=["id", "url", "feature"])
        # 将查询结果转换为 CleanImage 对象列表
        clean_images = [MilvusImage(id=result["id"], url=result["url"], feature=result["feature"]) for result in
                        results]
        if len(clean_images) > 0:
            return clean_images[0]
        return None
