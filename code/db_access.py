from pymilvus import *

# Fill in your database ip address
client = MilvusClient(
    uri="http://XXX.XXX.X.XXX:XXXX"
)
collection = 'image'

def create_collection():
    client.create_collection(
        collection_name=collection
    )

def load_collection():
    client.load_collection(
        collection_name=collection
    )

def drop_collection():
    client.drop_collection(
        collection_name=collection
    )

def list_partitions():
    res = client.list_partitions(collection_name=collection)
    print(res)

def load_partitions(partitions: list):
    client.load_partitions(
        collection_name=collection,
        partition_names=partitions
    )

def create_partition(partition):
    client.create_partition(
        collection_name=collection,
        partition_name=partition
    )

def release_partitions(partitions: list):
    client.release_partitions(
        collection_name=collection,
        partition_names=partitions
    )

# Before dropping a partition, you need to release it from memory.
def drop_partition(partition):
    release_partitions(partition)
    client.drop_partition(
        collection_name=collection,
        partition_name=partition
    )

def get_sizeof_partition(partition:list):
    load_partitions(partition)
    res = client.query(
        collection_name=collection,
        filter="",
        partition_names=partition,
        output_fields=["count(*)"]
    )
    size = res[0]["count(*)"]
    release_partitions(partition)
    return size

def load_data_from_partition(partition):
    load_partitions([partition])
    res = client.query(
        collection_name=collection,
        filter="",
        partition_names=[partition],
        limit=16384                 # max limit of Milvus
    )
    release_partitions([partition])
    return res                      # dict


def insert_data_to_partition(partition, data):
    '''
    #####   Schema of collection    #####
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="image_url", datatype=DataType.VARCHAR, max_length=200)
    schema.add_field(field_name="image_description", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="ai_description", datatype=DataType.VARCHAR, max_length=10000)
    schema.add_field(field_name="keywords", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=4096, max_length=1000)
    schema.add_field(field_name="group", datatype=DataType.INT64)
    schema.add_field(field_name="subgroup", datatype=DataType.INT64)
    '''
    load_partitions([partition])
    res = client.insert(
        collection_name=collection,
        data=data,
        partition_name=partition
    )
    release_partitions([partition])


def similarity_search(partition, query_vector, limit):
    load_partitions([partition])
    res = client.search(
        collection_name=collection,
        data=[query_vector],
        limit=limit,
        partition_names=[partition]
    )
    release_partitions([partition])
    return res

def get_ids_by_similarity_search_result(result):
    ids = []
    for dict in result[0]:
        ids.append(dict['id'])
    return ids

def get_entities_by_ids(partition, ids):
    load_partitions([partition])
    res = client.get(
        collection_name=collection,
        ids=ids,
        partition_names=[partition]
    )
    release_partitions([partition])
    return res
